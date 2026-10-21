"""The 32-blade radiation-tolerant trainer: sanitizer, TMR voting, ECC/OBC checkpoint healing, SAA mode switching, thermal hibernation, latchup,
the PyTorch / NumPy engines, and the protected-vs-unprotected comparison."""
import math
import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

import radiation_pytorch_scheduler as rps
import seu_injector as si

ROOT = Path(__file__).resolve().parent.parent
NOMINAL = dict(lat_deg=0.0, lon_deg=-150.0, alt_km=600.0)


def make(protected=True, engine="numpy", seed=1, **cfg):
    task = rps.SyntheticTask(seed)
    return rps.RadiationTolerantTrainer(rps.get_engine(engine), task, rps.init_weights(seed), protected, rps.TrainerConfig(**cfg))


def env(step, dt=20.0, temp=45.0, eta=None, flare=1.0, **where):
    return rps.StepEnv(step * dt, dt, **{**NOMINAL, **where}, core_temp_c=temp, saa_eta_s=eta, flare=flare)


def hit(blade, bit, when=0.1, target="weights", elem=0.3):
    """A hand-made upset: flips `bit` of the element at fraction `elem` of the tensor."""
    return si.Upset(blade, "SEU", when, target, elem, (bit + 0.5) / 32)


def run(tr, n, hits_at=None, **kw):
    reps = []
    for s in range(n):
        reps.append(tr.step(env(s, **kw), (hits_at or {}).get(s, [])))
    return reps


def blades_match_reference(tr, blades=None):
    ref = tr.obc.reference()
    return all(np.array_equal(tr.weights_np(i), ref) for i in (blades if blades is not None else range(rps.N_BLADES)))


# ---- layout ---------------------------------------------------------------------------------------------------------------------------
def test_topology_matches_the_alpha_sigma_design():
    assert rps.N_BLADES == 32 and rps.BANK_A == tuple(range(16)) and rps.BANK_B == tuple(range(16, 32))
    assert len(rps.TRIADS) == 10 and sorted(m for t in rps.TRIADS for m in t) == [i for i in range(32) if i not in (15, 31)]   # blades 16 and 32 are the spares
    assert rps.blade_name(0) == "A01" and rps.blade_name(31) == "B16"
    assert rps.N_PARAMS == 2516 and rps.SLICES[-1][3] == rps.N_PARAMS


# ---- engines --------------------------------------------------------------------------------------------------------------------------
def test_torch_and_numpy_engines_agree():
    pytest.importorskip("torch")
    task = rps.SyntheticTask(2)
    w = rps.init_weights(2)
    X, y = task.batch(1, 0)
    ln, gn = rps.NumpyEngine().loss_grad(w, X, y)
    lt, gt = rps.TorchEngine().loss_grad(rps.TorchEngine().native(w), X, y)
    assert ln == pytest.approx(lt, rel=1e-5)
    assert np.abs(gn - gt.numpy()).max() < 1e-5 * max(1.0, np.abs(gn).max())


def test_numpy_gradient_matches_finite_differences():
    task = rps.SyntheticTask(3)
    w = rps.init_weights(3).astype(np.float64)
    X, y = task.batch(1, 0)
    eng = rps.NumpyEngine()
    _, g = eng.loss_grad(w.astype(np.float32), X, y)
    for i in (0, 100, 900, 2400, 2515):
        e = np.zeros_like(w)
        e[i] = 1e-3
        lp, _ = eng.loss_grad((w + e).astype(np.float32), X, y)
        lm, _ = eng.loss_grad((w - e).astype(np.float32), X, y)
        assert g[i] == pytest.approx((lp - lm) / 2e-3, abs=2e-3)


def test_engine_selection_and_fallback(monkeypatch):
    assert isinstance(rps.get_engine("numpy"), rps.NumpyEngine)
    monkeypatch.setattr(rps, "TORCH_AVAILABLE", False)
    assert isinstance(rps.get_engine("auto"), rps.NumpyEngine)
    with pytest.raises(RuntimeError, match="not installed"):
        rps.get_engine("torch")


def test_works_with_torch_blocked_in_a_fresh_interpreter(tmp_path):
    """The lightweight environment: no torch at all. The import must succeed and training, healing and export must work on the NumPy engine."""
    code = (
        "import sys; sys.modules['torch'] = None\n"
        "import radiation_pytorch_scheduler as r, seu_injector as si, numpy as np\n"
        "assert not r.TORCH_AVAILABLE and r.get_engine('auto').name == 'numpy'\n"
        "task = r.SyntheticTask(1); tr = r.RadiationTolerantTrainer(r.get_engine('auto'), task, r.init_weights(1))\n"
        "for s in range(30):\n"
        "    ups = [si.Upset(3, 'SEU', 0.1, 'weights', 0.3, 30.5/32)] if s == 10 else []\n"
        "    rep = tr.step(r.StepEnv(s*20.0, 20.0, 0.0, -150.0, 600.0), ups)\n"
        "assert tr.totals['scrubs'] == 1 and np.isfinite(rep.loss) and rep.loss < 1.0\n"
        "import tempfile, os; p = os.path.join(sys.argv[1], 'w.pt')\n"
        "o = r.RadiationOrchestrator(r.TrackOrbit(np.array([0.0, 1e5]), np.zeros(2), np.zeros(2), np.full(2, 600.0)))\n"
        "assert o.engine.name == 'numpy'; out = o.export_weights(p); assert str(out).endswith('.npz') and os.path.exists(out)\n"
        "print('NO-TORCH OK')\n")
    r = subprocess.run([sys.executable, "-c", code, str(tmp_path)], cwd=ROOT, capture_output=True, text=True, timeout=120)
    assert "NO-TORCH OK" in r.stdout, r.stderr


def test_training_converges_on_both_engines():
    engines = ["numpy"] + (["torch"] if rps.TORCH_AVAILABLE else [])
    for e in engines:
        tr = make(engine=e)
        reps = run(tr, 120)
        assert reps[0].loss > 1.0 and reps[-1].loss < 0.4 and not any(r.quarantined for r in reps)
        loss, acc = tr.evaluate(tr.consensus_weights())
        assert acc > 0.9 and loss < 0.4


# ---- layer 1: the sanitizer -----------------------------------------------------------------------------------------------------------
def _grads(norms):
    rng = np.random.default_rng(0)
    out = []
    for n in norms:
        v = rng.standard_normal(100).astype(np.float32)
        out.append(v / np.linalg.norm(v) * n)
    return out


def test_sanitizer_accepts_normal_and_quarantines_nan_inf_and_large_norms():
    g = _grads([1.0, 1.1, 0.9, 1.05, 0.95])
    g[1][7] = np.nan
    g[2][3] = np.inf
    g[3] = g[3] * 6.0                                                                # norm 6.3 > 5 x median
    rep = rps.sanitize_gradients(g)
    assert rep.accept == [True, False, False, False, True]
    assert rep.reasons[1] == "nan/inf" and rep.reasons[2] == "nan/inf" and "norm >" in rep.reasons[3]
    assert not math.isfinite(rep.norms[2]) and math.isfinite(rep.norms[0])
    assert rep.median_norm == pytest.approx(1.0, abs=0.06)


def test_sanitizer_norm_threshold_is_five_times_the_median():
    base = [1.0] * 6
    assert rps.sanitize_gradients(_grads(base + [4.5])).accept[-1]                    # 4.5x: kept
    assert not rps.sanitize_gradients(_grads(base + [5.5])).accept[-1]                # 5.5x: quarantined
    assert not rps.sanitize_gradients(_grads(base + [1e30])).accept[-1]               # an exponent blow-up that stayed finite
    assert rps.sanitize_gradients([None, _grads([1.0])[0]]).accept == [False, True]
    assert rps.sanitize_gradients([np.zeros(10, np.float32)] * 3).accept == [True] * 3 # all-zero gradients: median 0, nothing is "5x" it


def test_sanitizer_hash_is_a_fingerprint_of_the_bytes():
    g = _grads([1.0, 1.0])
    g2 = g[0].copy()
    rep = rps.sanitize_gradients([g[0], g2, g[1]])
    assert rep.hashes[0] == rep.hashes[1] != rep.hashes[2]
    g2.view(np.uint32)[5] ^= 1                                                       # a single flipped mantissa bit changes the hash
    assert rps.tensor_hash(g2) != rps.tensor_hash(g[0])


# ---- layer 2: TMR voting --------------------------------------------------------------------------------------------------------------
def test_tmr_vote_cases():
    a = np.arange(10, dtype=np.float32)
    b = a.copy()
    c = a.copy()
    assert rps.tmr_vote([a, b, c]).status == "unanimous" and rps.tmr_vote([a, b, c]).agree == [True] * 3
    c.view(np.uint32)[2] ^= 1 << 30                                                  # one member corrupted
    v = rps.tmr_vote([a, b, c])
    assert v.status == "majority" and v.agree == [True, True, False] and v.winner in (0, 1)
    v = rps.tmr_vote([c, a, b])                                                      # the corrupted one first: the majority still wins
    assert v.agree == [False, True, True] and v.winner in (1, 2)
    d = a.copy()
    d.view(np.uint32)[4] ^= 1 << 20
    assert rps.tmr_vote([a, c, d]).winner is None and rps.tmr_vote([a, c, d]).status == "no-majority"
    assert rps.tmr_vote([a, b, None]).status == "duplex-agree"
    assert rps.tmr_vote([a, c, None]).winner is None
    assert rps.tmr_vote([None, None, a]).status == "single" and rps.tmr_vote([None, None, None]).winner is None
    nan = a.copy()
    nan[0] = np.nan
    assert rps.tmr_vote([nan, nan.copy(), a]).agree == [True, True, False]           # NaNs compare by bytes, not by value


# ---- layer 3: OBC checkpoint, ECC parity, healing -------------------------------------------------------------------------------------
def test_obc_journal_replay_is_bit_exact_and_checkpoints_are_taken_every_n_steps():
    tr = make(checkpoint_interval=5)
    run(tr, 13)
    obc = tr.obc
    assert obc.checkpoints_taken == 1 + 13 // 5 and obc.ckpts[0]["step"] == 10 and len(obc.journal) == 3
    obc.live.view(np.uint32)[100] ^= 1 << 12                                          # corrupt the live copy in OBC memory
    assert obc.live_sha != rps.sha256_hex(obc.live)
    rebuilt = obc.reference().copy()
    assert obc.rebuilds == 1 and np.array_equal(rebuilt, tr.weights_np(0))           # checkpoint + journal reproduces the blades' weights bit for bit


def test_obc_mirror_failover_and_double_failure():
    tr = make(checkpoint_interval=4)
    run(tr, 6)
    obc = tr.obc
    good = tr.weights_np(0).copy()
    obc.ckpts[0]["w"].view(np.uint32)[7] ^= 1 << 29                                   # copy A of the checkpoint is damaged ...
    obc.live.view(np.uint32)[9] ^= 1 << 3                                              # ... and so is the live copy
    assert np.array_equal(obc.reference(), good) and obc.mirror_failovers == 1        # copy B saved the day
    obc.ckpts[0]["w"].view(np.uint32)[7] ^= 1
    obc.ckpts[1]["w"].view(np.uint32)[7] ^= 1
    obc.live.view(np.uint32)[9] ^= 1
    with pytest.raises(RuntimeError, match="both OBC checkpoint"):
        obc.reference()


def test_exponent_strike_on_weights_is_scrubbed_before_the_forward_pass():
    tr = make()
    reps = run(tr, 5, {4: [hit(5, 30)]})
    r = reps[4]
    assert len(r.flips) == 1 and r.flips[0][1].blowup and r.flips[0][1].field == "exponent"
    assert [s[0] for s in r.scrubbed] == [5] and r.accepted_units == 32 and r.updated and math.isfinite(r.loss)
    assert tr.display_state(5) == rps.QUARANTINED                                      # red flash: NaN-class strike caught
    assert blades_match_reference(tr)                                                  # blade 5 is back in bit-exact agreement with the OBC
    assert tr.blades[5].scrubs == 1 and tr.blades[5].restores == 1


def test_mantissa_flip_is_scrubbed_with_a_yellow_flash_and_only_the_damaged_tensor_moves():
    tr = make()
    reps = run(tr, 4, {3: [hit(9, 4, elem=0.9)]})                                      # element 0.9 * 2516 lies in a weight matrix
    r = reps[3]
    assert r.flips[0][1].field == "mantissa" and not r.flips[0][1].blowup
    assert len(r.scrubbed) == 1 and r.scrubbed[0][1].endswith("weight") and r.scrubbed[0][0] == 9
    assert tr.display_state(9) == rps.SCRUBBED and blades_match_reference(tr)                 # yellow flash
    assert max(rep.max_restore_ms for rep in reps) < rps.RESTORE_BUDGET_MS


def test_hit_after_the_parity_check_is_caught_at_the_next_step():
    tr = make()
    reps = run(tr, 3, {2: [hit(7, 4, when=0.9)]})                                      # 'late': lands after this step's parity check
    assert not reps[2].scrubbed and not np.array_equal(tr.weights_np(7), tr.obc.reference())    # blade 7 is wrong for the moment ...
    r = tr.step(env(3), [])
    assert [x[0] for x in r.scrubbed] == [7] and tr.display_state(7) == rps.SCRUBBED       # ... and scrubbed one step later
    assert blades_match_reference(tr)


def test_late_exponent_strike_gives_a_quarantined_gradient_and_the_step_still_updates():
    tr = make()
    reps = run(tr, 3, {2: [hit(7, 30, when=0.9)]})
    r = reps[2]
    assert [q[0] for q in r.quarantined] == [7] and r.accepted_units == 31 and r.updated and math.isfinite(r.loss)
    assert tr.blades[7].quarantines == 1 and tr.display_state(7) == rps.QUARANTINED
    assert not np.isfinite(tr.weights_np(7)).all() or np.abs(tr.weights_np(7)).max() > 1e30     # the weights really are blown up ...
    tr.step(env(3), [])
    assert blades_match_reference(tr)                                                    # ... until the parity check restores them


def test_gradient_strike_is_quarantined_and_does_not_touch_the_weights():
    tr = make()
    reps = run(tr, 3, {2: [hit(11, 30, target="grads")]})
    r = reps[2]
    assert [q[0] for q in r.quarantined] == [11] and r.updated and not r.scrubbed
    assert "nan/inf" in r.quarantined[0][1] or "norm" in r.quarantined[0][1]
    assert blades_match_reference(tr) and tr.display_state(11) == rps.QUARANTINED


def test_restore_time_model_stays_inside_the_50_ms_budget():
    assert rps.backplane_ms(0) == rps.BACKPLANE_LATENCY_MS
    assert rps.backplane_ms(rps.N_PARAMS * 4) == pytest.approx(0.05 + rps.N_PARAMS * 4 * 8 / 100e9 * 1e3)
    assert rps.backplane_ms(rps.N_PARAMS * 4) < 0.1 < rps.RESTORE_BUDGET_MS
    assert rps.backplane_ms(int(0.7e9)) > rps.RESTORE_BUDGET_MS > rps.backplane_ms(int(0.5e9))     # the 50 ms budget holds for shards up to ~0.6 GB at 100 Gbps


def test_unprotected_system_is_destroyed_by_one_exponent_strike():
    prot, base = make(True), make(False)
    for s in range(6):
        ups = [hit(5, 30)] if s == 3 else []
        rp, rb = prot.step(env(s), ups), base.step(env(s), ups)
    assert math.isfinite(rp.loss) and prot.evaluate(prot.consensus_weights())[1] > 0.5
    assert math.isnan(rb.loss) or not math.isfinite(rb.loss)
    assert base.poisoned and all(not np.isfinite(base.weights_np(i)).all() for i in (0, 31))     # every blade's weights are NaN: the update was poisoned
    assert not np.isfinite(base.evaluate(base.consensus_weights())[0])


def test_unprotected_system_keeps_mantissa_corruption_forever():
    base = make(False)
    run(base, 4, {1: [hit(6, 22, when=0.1)]})
    assert not np.array_equal(base.weights_np(6), base.weights_np(7))


# ---- TMR mode -------------------------------------------------------------------------------------------------------------------------
def test_tmr_mode_runs_ten_triads_and_outvotes_a_corrupted_member():
    tr = make()
    run(tr, 3, eta=100.0)
    r = tr.step(env(3, eta=100.0), [hit(4, 30, when=0.9)])                              # blade index 4 sits in the triad (3, 4, 5)
    assert r.mode == rps.TMR and r.unique_batches == 10 and r.rel_throughput == pytest.approx(10 / 32)
    assert r.voted_out == [4] and r.accepted_units == 10 and r.updated and math.isfinite(r.loss)
    assert tr.display_state(4) == rps.QUARANTINED and tr.display_state(0) == rps.TMR_STATE
    assert tr.display_state(15) == rps.TMR_STATE and tr.display_state(31) == rps.TMR_STATE        # idle hot spares are shielded too
    r2 = tr.step(env(4, eta=100.0), [])
    assert r2.scrubbed and r2.scrubbed[0][0] == 4 and blades_match_reference(tr)


def test_tmr_wins_where_data_parallel_loses_a_gradient():
    """The same single-bit gradient hit: DP must drop that blade's batch (31/32 kept); TMR keeps all 10 batches by outvoting it."""
    dp, tm = make(), make()
    run(dp, 2), run(tm, 2, eta=50.0)
    rd = dp.step(env(2), [hit(4, 30, target="grads")])
    rt = tm.step(env(2, eta=50.0), [hit(4, 30, target="grads")])
    assert rd.accepted_units == 31 and rt.accepted_units == 10 and rt.voted_out == [4]


def test_tmr_without_majority_drops_that_triad_and_a_dead_blade_is_replaced_by_the_spare():
    tr = make()
    run(tr, 2, eta=50.0)
    r = tr.step(env(2, eta=50.0), [hit(3, 30, target="grads"), hit(4, 22, target="grads"), hit(5, 29, target="grads")])
    assert r.vote_failed == 1 and r.accepted_units == 9 and r.updated
    tr2 = make()
    run(tr2, 2, eta=50.0)
    tr2.step(rps.StepEnv(40.0, 10.0, **NOMINAL, saa_eta_s=50.0), [si.Upset(1, "SEL", 0.0, "", 0.0, 0.0)])     # blade index 1 (triad 0) latches up until t = 55 ...
    assert tr2._units([i for i in range(32) if tr2.blades[i].online(50.0)])[0] == [0, 2, 15]    # ... so hot spare 16 joins its triad
    r2 = tr2.step(rps.StepEnv(50.0, 10.0, **NOMINAL, saa_eta_s=50.0), [])
    assert r2.mode == rps.TMR and r2.accepted_units == 10 and r2.voted_out == []             # the repaired-by-substitution triad still delivers its batch


def test_mode_switches_to_tmr_before_the_saa_and_back_after():
    tr = make()
    r = tr.step(env(0), [])
    assert r.mode == rps.DP
    r = tr.step(env(1, eta=240.0), [])                                                    # the orbit engine predicts the SAA in 4 minutes
    assert r.mode == rps.TMR and any("TMR" in e for e in r.events)
    r = tr.step(env(2, lat_deg=-25.0, lon_deg=-45.0, eta=0.0), [])                         # inside the SAA
    assert r.mode == rps.TMR and r.zone == si.SAA and r.flux_mult == pytest.approx(100.0)
    r = tr.step(env(3), [])                                                               # out and no SAA predicted: back to data-parallel
    assert r.mode == rps.DP and r.unique_batches == 32
    r = tr.step(env(4, flare=1000.0), [])                                                  # a solar flare also triggers shielding
    assert r.mode == rps.TMR


def test_unprotected_trainer_never_uses_tmr():
    assert make(False).step(env(0, lat_deg=-25.0, lon_deg=-45.0, eta=0.0), []).mode == rps.DP


# ---- thermal and power coupling -------------------------------------------------------------------------------------------------------
def test_soft_throttle_hibernates_bank_b_and_consolidates_onto_bank_a():
    tr = make()
    run(tr, 3)
    r = tr.step(env(3, temp=76.0), [])
    assert r.throttled and any("SOFT THROTTLE" in e for e in r.events)
    assert all(tr.display_state(i) == rps.HIBERNATED for i in rps.BANK_B) and all(tr.display_state(i) == rps.HEALTHY for i in rps.BANK_A)
    assert r.unique_batches == 32 and r.accepted_units == 16 and r.rel_throughput == pytest.approx(0.5)     # 16 blades x 2 micro-batches
    assert r.updated and math.isfinite(r.loss)
    r = tr.step(env(4, temp=72.0), [])                                                     # between 70 and 75: hysteresis keeps it throttled
    assert r.throttled
    r = tr.step(env(5, temp=69.0), [])                                                     # cooled: Bank B wakes, restored from the OBC checkpoint
    assert not r.throttled and r.accepted_units == 32 and any("back online" in e for e in r.events)
    assert blades_match_reference(tr)


def test_hibernated_blades_are_not_hit_and_consolidated_tmr_covers_all_batches():
    tr = make()
    r = tr.step(env(0, temp=80.0, eta=50.0), [hit(20, 30), hit(2, 4, target="grads")])
    assert r.mode == rps.TMR and r.unique_batches == 10 and r.accepted_units == 5 and r.rel_throughput == pytest.approx(10 / 2 / 32)
    assert r.hits == 1 and r.flips[0][0] == 2                                              # the hit on hibernated blade 21 never happened


# ---- latchup ----------------------------------------------------------------------------------------------------------------------------
def test_latchup_takes_a_blade_offline_for_15_seconds_then_heals_it():
    tr = make(sel_offline_s=15.0)
    run(tr, 3)
    r = tr.step(env(3, dt=10.0), [si.Upset(9, "SEL", 0.0, "", 0.0, 0.0)])
    assert r.latchups == 1 and any("latchup" in e.lower() for e in r.events) and tr.display_state(9) == rps.RESET
    assert tr.blades[9].offline_until == pytest.approx(30.0 + 15.0)
    assert np.all(tr.weights_np(9) == 0.0)                                                 # the power cycle wiped its HBM
    r = tr.step(rps.StepEnv(40.0, 10.0, **NOMINAL), [])                                    # t = 40..50: still down until t = 45
    assert r.accepted_units == 31 and tr.display_state(9) == rps.RESET
    r = tr.step(rps.StepEnv(50.0, 10.0, **NOMINAL), [])
    assert r.accepted_units == 32 and any("back online" in e for e in r.events) and blades_match_reference(tr)
    assert r.max_restore_ms < rps.RESTORE_BUDGET_MS


def test_nothing_online_skips_the_step_without_crashing():
    tr = make()
    ups = [si.Upset(i, "SEL", 0.0, "", 0.0, 0.0) for i in range(32)]
    r = tr.step(env(0), ups)
    r2 = tr.step(env(1, dt=1.0), [])
    assert r.latchups == 32 and not r2.updated and r2.accepted_units == 0 and any('no blade online' in e for e in r2.events)


# ---- the orchestrator -----------------------------------------------------------------------------------------------------------------
class FixedOrbit:
    """A ground track for tests: constant position (and optionally a sweep in longitude)."""

    def __init__(self, lat=0.0, lon=-150.0, dlon=0.0, t_end=1e5):
        self.t = np.array([0.0, t_end])
        self.lat, self.lon, self.dlon = lat, lon, dlon

    def position(self, t):
        return self.lat, si.wrap_lon(self.lon + self.dlon * t), 600.0

    def saa_eta(self, t, lookahead_s):
        return si.predict_saa_entry_s(self.position, t, lookahead_s)


def drive(orch, seconds, step=20.0):
    t = orch.t
    reps = []
    while t < orch.t + seconds - 1e-9 and len(reps) < 100000:
        t += step
        reps += orch.advance_to(t, max_jump_s=1e9)
        if len(reps) >= int(seconds / step):
            break
    return reps


def test_solar_flare_kills_the_baseline_but_the_protected_system_keeps_learning():
    o = rps.RadiationOrchestrator(FixedOrbit(), cfg=rps.OrchestratorConfig(engine="numpy", seed=3))
    o.set_flare(True)
    drive(o, 20 * 80)
    s = o.summary()
    assert s["baseline_poisoned"] and s["first_baseline_nan_step"] is not None and not np.isfinite(s["baseline_val_loss"])
    assert np.isfinite(s["protected_val_loss"]) and s["protected_val_acc"] > 0.85
    assert s["protected_totals"]["hits"] > 200 and s["protected_totals"]["tmr_steps"] > 0


def test_unprotected_baseline_usually_explodes_in_the_saa_and_protected_never_does():
    """Whether one particular pass kills the baseline is chance (a bit 30 hit is 1 in 32), so look at several seeds: the protected system never fails."""
    dead = 0
    for seed in (5, 6, 7, 8):
        orbit = FixedOrbit(lat=-20.0, lon=-130.0, dlon=0.05)                                # reaches the SAA (lon -90) at t = 800 s, leaves at t = 3600 s
        o = rps.RadiationOrchestrator(orbit, cfg=rps.OrchestratorConfig(engine="numpy", seed=seed))
        drive(o, 3600)
        h = o.history
        first_tmr = next(t for t, m in zip(h["t"], h["mode"]) if m == rps.TMR)
        assert 480.0 <= first_tmr <= 520.0                                                   # shielding starts the 300 s look-ahead before the SAA
        assert all(z != si.SAA for z, t in zip(h["zone"], h["t"]) if t < 800.0 - 1)
        nan_steps = [t for t, x in zip(h["t"], h["loss_baseline"]) if not math.isfinite(x)]
        dead += bool(nan_steps)
        assert all(t >= 780.0 for t in nan_steps)                                            # nothing dies in the quiet zone before it
        assert all(np.isfinite(h["loss_protected"])) and h["loss_protected"][-1] < 0.5 and o.summary()["protected_val_acc"] > 0.9
    assert dead >= 2


def test_protection_toggle_makes_the_protected_line_behave_like_the_baseline():
    o = rps.RadiationOrchestrator(FixedOrbit(), cfg=rps.OrchestratorConfig(engine="numpy", seed=5))
    o.set_protection(False)
    o.set_flare(True)
    drive(o, 20 * 30)
    assert o.protected.poisoned and o.baseline.poisoned
    assert o.protected.mode == rps.DP and not any(m == rps.TMR for m in o.history["mode"])


def test_turning_protection_back_on_resyncs_the_obc_from_the_blades():
    o = rps.RadiationOrchestrator(FixedOrbit(), cfg=rps.OrchestratorConfig(engine="numpy", seed=5))
    o.set_protection(False)
    drive(o, 20 * 20)
    o.set_protection(True)
    ref = o.protected.obc.reference()
    assert np.array_equal(ref, o.protected.weights_np(0)) and np.isfinite(ref).all()
    drive(o, 20 * 10)
    assert np.isfinite(o.history["loss_protected"][-1]) and o.protected.display_state(0) in (rps.HEALTHY, rps.SCRUBBED, rps.QUARANTINED)


def test_the_two_systems_see_identical_radiation_events():
    o = rps.RadiationOrchestrator(FixedOrbit(lat=-25.0, lon=-45.0), cfg=rps.OrchestratorConfig(engine="numpy", seed=6))
    drive(o, 20 * 15)
    p, b = o.protected.totals, o.baseline.totals
    assert p["hits"] == b["hits"] > 0 and p["latchups"] == b["latchups"]


def test_clock_handling_catches_up_in_bounded_steps_and_resyncs_on_jumps():
    o = rps.RadiationOrchestrator(FixedOrbit(), cfg=rps.OrchestratorConfig(engine="numpy", max_steps_per_call=4))
    assert o.advance_to(10.0) == []                                                          # less than one step of mission time has passed
    reps = o.advance_to(400.0)
    assert len(reps) == 4 and o.t == pytest.approx(400.0)   # capped at 4 steps that cover the whole 400 s
    assert o.advance_to(100.0) == [] and o.t == 100.0                                         # the clock went backwards: resync, no training
    assert o.advance_to(100.0 + 7200.0) == [] and o.t == 7300.0                               # a seek of 2 h: resync


def test_chaos_latchup_and_hot_core():
    o = rps.RadiationOrchestrator(FixedOrbit(), cfg=rps.OrchestratorConfig(engine="numpy"), thermal=lambda t: 45.0)
    b = o.force_latchup(12)
    assert b == 12
    o.temp_offset_c = 35.0
    rp, rb = o.step(20.0)
    assert rp.latchups == 1 and rp.throttled and "SOFT THROTTLE" in " ".join(o.drain_log()) and o.drain_log() == []
    grid = o.blade_grid()
    assert len(grid) == 32 and grid[20][0] == rps.HIBERNATED and grid[0][0] == rps.HEALTHY and "Blade A13" in grid[12][1]


def test_export_roundtrip_pt_and_npz(tmp_path):
    o = rps.RadiationOrchestrator(FixedOrbit(), cfg=rps.OrchestratorConfig(engine="numpy"))
    drive(o, 20 * 10)
    sha = rps.sha256_hex(o.protected.consensus_weights())
    npz = o.export_weights(tmp_path / "w.npz")
    meta = rps.load_checkpoint_meta(npz)
    assert meta["sha256"] == sha and "fc1.weight" in meta["names"] and meta["step"] == 10
    d = np.load(npz)
    assert d["fc1.weight"].shape == (48, 16) and d["fc3.bias"].shape == (4,)
    if rps.TORCH_AVAILABLE:
        import torch
        pt = o.export_weights(tmp_path / "w.pt")
        assert pt.suffix == ".pt"
        blob = torch.load(pt, weights_only=True)
        assert blob["meta"]["sha256"] == sha and blob["state_dict"]["fc1.weight"].shape == (48, 16)
        assert torch.equal(blob["state_dict"]["fc2.bias"], torch.from_numpy(d["fc2.bias"]))
        assert rps.load_checkpoint_meta(pt)["names"][0] == "fc1.weight"


def test_track_orbit_and_core_temp_from_a_real_simulation():
    import simulation as sim
    from helpers import direct_cfg
    res = sim.run_simulation(direct_cfg(duration_h=3, step_s=30))
    orbit = rps.TrackOrbit.from_result(res)
    lat, lon, alt = orbit.position(1000.0)
    tr = res.tracks[res.target]
    i = res.index_at(1000.0)
    assert abs(lat - tr.lat_deg[i]) < 1.0 and 560 < alt < 640 and -180 <= lon <= 180
    ct = rps.CoreTemp(res)
    assert 30.0 < ct(1000.0) < 90.0
    zones = {si.classify_zone(*orbit.position(t)[:2]) for t in np.arange(0, 3 * 3600, 60.0)}
    assert si.SAA in zones and si.POLAR_N in zones                                           # three hours of a 97.8 degree orbit cross both
    etas = [orbit.saa_eta(t, 300.0) for t in np.arange(0, 3 * 3600, 30.0)]
    assert any(e is not None and e > 0 for e in etas) and any(e == 0.0 for e in etas)


def test_command_line_demo(capsys):
    assert rps.main(["--hours", "1.2", "--engine", "numpy", "--flare"]) == 0
    out = capsys.readouterr().out
    assert "PROTECTED" in out and "UNPROTECTED" in out and "bit flips" in out and "went NaN" in out
