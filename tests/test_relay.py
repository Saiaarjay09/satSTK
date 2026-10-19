"""The geostationary optical relay network: orbit constants, geometry, multi-hop routing, link budget and handovers."""
import math

import numpy as np
import pytest

import events as ev
import laser_link_engine as ll
import orbital_engine as oe
import simulation as sim
from helpers import direct_cfg

LAM = 1550e-9


@pytest.fixture(scope="module")
def clear():
    return sim.run_simulation(sim.SimulationConfig())


@pytest.fixture(scope="module")
def direct_only():
    return sim.run_simulation(direct_cfg())


# ---- the GEO ring ------------------------------------------------------------------------------------------------------
def test_geo_orbit_constants_match_the_textbook_values():
    assert oe.SIDEREAL_DAY_S == 86164                                          # 23 h 56 min 04 s
    assert oe.GEO_ALTITUDE_KM == pytest.approx(35786.0, abs=0.5)
    assert oe.GEO_RADIUS_KM == pytest.approx(42164.0, abs=0.5)
    assert oe.orbital_period_s(oe.GEO_RADIUS_KM) == pytest.approx(86164.0, abs=0.01)    # Kepler: the radius really gives that period
    assert oe.SIDEREAL_DAY_S < 86400.0


def test_the_three_default_relays_sit_where_the_brief_says():
    ind, atl, pac = oe.default_geo_relays()
    assert (ind.name, ind.lon_deg) == ("GEO-Relay-IND", 77.5)
    assert (atl.name, atl.lon_deg % 360.0) == ("GEO-Relay-ATL", 315.0)                 # 45 W
    assert (pac.name, pac.lon_deg % 360.0) == ("GEO-Relay-PAC", 195.0)                 # 165 W
    spacing = sorted(math.degrees(math.atan2(math.sin(math.radians(a - b)), math.cos(math.radians(a - b)))) % 360.0
                     for a, b in ((77.5, 195.0), (195.0, 315.0), (315.0, 77.5)))
    assert all(115.0 < s < 125.0 for s in [360.0 - x for x in spacing])               # roughly 120 degrees apart


def test_geo_satellite_is_stationary_in_the_earth_frame_and_circular_in_the_inertial_frame():
    t = oe.time_grid(86164.0, 60.0)
    from datetime import datetime, timezone
    g = oe.gmst_rad(datetime(2026, 12, 21, tzinfo=timezone.utc), t)
    tr = oe.geo_relay_track(oe.default_geo_relays()[0], g)
    assert np.ptp(tr.r_ecef, axis=0) == pytest.approx([0, 0, 0], abs=1e-9)
    lat, lon, alt = oe.ecef_to_geodetic(tr.r_ecef[:1])
    assert (lat[0], lon[0]) == pytest.approx((0.0, 77.5), abs=1e-6) and alt[0] == pytest.approx(35786.0, abs=1.0)
    assert np.linalg.norm(tr.r_eci, axis=1) == pytest.approx(oe.GEO_RADIUS_KM)
    assert np.linalg.norm(tr.v_eci, axis=1) == pytest.approx(math.sqrt(oe.MU_EARTH / oe.GEO_RADIUS_KM), rel=1e-3)     # circular-orbit speed = omega * r: it IS geostationary
    ang = np.unwrap(np.arctan2(tr.r_eci[:, 1], tr.r_eci[:, 0]))
    assert (ang[-1] - ang[0]) == pytest.approx(2 * math.pi, rel=2e-3)                    # one inertial turn per sidereal day
    assert tr.r_eci[:, 2] == pytest.approx(0.0, abs=1e-9)                                # equatorial: zero inclination


def test_india_station_sees_geo_ind_high_and_permanently_but_not_the_other_two():
    gs = oe.GroundStation()
    from datetime import datetime, timezone
    g = oe.gmst_rad(datetime(2026, 12, 21, tzinfo=timezone.utc), np.array([0.0]))
    tracks = {r.short: oe.geo_relay_track(r, g) for r in oe.default_geo_relays()}
    t = {k: oe.topocentric_state(gs, v.r_ecef, np.zeros_like(v.r_ecef)) for k, v in tracks.items()}
    assert t["GEO-IND"].elevation_deg[0] == pytest.approx(74.7, abs=0.3)               # directly south, high up
    assert t["GEO-IND"].range_km[0] == pytest.approx(35976.0, abs=10.0)
    assert t["GEO-IND"].azimuth_deg[0] == pytest.approx(180.0, abs=1e-6)
    assert t["GEO-ATL"].elevation_deg[0] < 0 and t["GEO-PAC"].elevation_deg[0] < 0     # below the horizon: the ground hop must use IND
    assert t["GEO-IND"].slew_rate_deg_s[0] == 0.0                                       # a fixed pointing direction: no tracking needed


def test_every_pair_of_geo_relays_has_a_clear_line_of_sight():
    from datetime import datetime, timezone
    g = oe.gmst_rad(datetime(2026, 12, 21, tzinfo=timezone.utc), np.array([0.0]))
    tr = [oe.geo_relay_track(r, g) for r in oe.default_geo_relays()]
    for i in range(3):
        for j in range(i + 1, 3):
            vis, clear = oe.line_of_sight(tr[i].r_eci, tr[j].r_eci)
            assert vis[0] and clear[0] > 10000.0                                       # the chord clears the Earth by well over 10,000 km


def test_three_relays_see_every_point_of_the_600km_shell_but_one_relay_does_not():
    """Geometric coverage: sample 40,000 random LEO positions (poles included)."""
    from datetime import datetime, timezone
    rng = np.random.default_rng(3)
    v = rng.normal(size=(40000, 3))
    leo = 6978.137 * v / np.linalg.norm(v, axis=1)[:, None]
    leo[:200] = [0, 0, 6978.137] * np.ones((200, 1)) * rng.choice([-1, 1], 200)[:, None]       # polar points
    g = oe.gmst_rad(datetime(2026, 12, 21, tzinfo=timezone.utc), np.array([0.0]))
    tr = [oe.geo_relay_track(r, g) for r in oe.default_geo_relays()]
    seen = np.stack([oe.line_of_sight(np.repeat(t.r_eci, len(leo), axis=0), leo)[0] for t in tr])
    assert seen.any(axis=0).all()                       # 100 % of the shell is visible from at least one relay
    assert 0.35 < seen[0].mean() < 0.65                  # a single relay sees roughly half of it: the far side is hidden behind the Earth
    assert (seen.sum(axis=0) >= 2).mean() > 0.2          # and there is overlap, which is what makes hand-overs make-before-break


# ---- link budget -----------------------------------------------------------------------------------------------------
def test_free_space_path_loss_at_geo_distance():
    # 20 log10(4 pi * 3.6e7 m / 1.55e-6 m) = 289.3 dB
    assert float(ll.fspl_db(36.0e6, LAM)) == pytest.approx(289.3, abs=0.1)


def test_geo_hop_budget_by_hand_with_300mm_apertures_and_a_10w_edfa():
    cfg = ll.hop_config(ll.LaserLinkConfig(), ll.RelayConfig())
    assert (cfg.tx_power_w, cfg.rx_aperture_m, cfg.divergence_urad, cfg.pointing_jitter_urad) == (10.0, 0.30, 10.0, 1.0)
    b = ll.link_budget(36000.0, cfg)
    w = 10e-6 * 36.0e6 / 2.0                                                         # beam radius: 180 m
    capture = 1 - math.exp(-2 * (0.15 / w) ** 2)
    point = 1 / (1 + 16 * (1e-6) ** 2 / (10e-6) ** 2)
    expected_w = 10.0 * 0.8 * 0.8 * capture * point
    assert float(b["rx_power_w"]) == pytest.approx(expected_w, rel=1e-9)
    assert float(b["rx_power_dbm"]) == pytest.approx(-21.2, abs=0.1)
    assert float(b["latency_ms"]) == pytest.approx(120.08, abs=0.01)                 # one hop of 36,000 km


def test_a_narrow_beam_and_big_aperture_are_what_make_the_geo_hop_work():
    base = ll.LaserLinkConfig()
    wide = ll.link_budget(36000.0, base)["rx_power_dbm"]                             # the 100 microradian, 10 W LEO terminal on a GEO hop
    narrow = ll.link_budget(36000.0, ll.hop_config(base, ll.RelayConfig()))["rx_power_dbm"]
    assert float(narrow - wide) == pytest.approx(20.0, abs=1.0)                      # 10x narrower beam = 100x power = 20 dB


def test_cascade_of_bent_pipe_hops_adds_inverse_snr():
    assert float(ll.cascade_es_n0(100.0, 100.0)) == pytest.approx(50.0)
    assert float(ll.cascade_es_n0(1000.0, 10.0)) == pytest.approx(1 / (1 / 1000 + 1 / 10))        # the weak hop dominates
    assert float(ll.cascade_es_n0(40.0, 40.0, 40.0)) == pytest.approx(40.0 / 3)
    x = np.array([10.0, 20.0])
    assert ll.cascade_es_n0(x, x) == pytest.approx(x / 2)


# ---- routing ---------------------------------------------------------------------------------------------------------
def test_clear_sky_gives_100_percent_uptime_with_the_relay_and_about_one_percent_without(clear, direct_only):
    assert clear.route.uptime_pct == pytest.approx(100.0, abs=1e-9)
    assert np.all(clear.route.kind > 0) and np.all(clear.route.connected)
    assert clear.primary_link.locked.all() and len(clear.primary_windows) == 1          # one continuous 24 h window
    assert 0.5 < clear.route.direct_uptime_pct < 3.0 and direct_only.route.uptime_pct == pytest.approx(clear.route.direct_uptime_pct)
    assert clear.route.direct_uptime_pct == pytest.approx(100.0 * float(np.mean(clear.ground_links[0].adaptive_throughput_gbps > 0)))
    assert direct_only.network_link is None and clear.network_link is not None
    assert clear.route.total_data_tb > 20 * direct_only.route.total_data_tb


def test_path_priorities_direct_first_then_one_relay_then_two(clear):
    rt, direct = clear.route, clear.ground_links[0]
    usable_direct = direct.visible & (direct.adaptive_throughput_gbps > 0)
    assert np.all(rt.kind[usable_direct] == 1)                                          # priority 1: direct whenever it carries data
    assert np.all(rt.kind[~usable_direct] >= 2)
    ind_sees_leo, _ = oe.line_of_sight(clear.relay_tracks[0].r_eci, clear.tracks[0].r_eci)
    two_hop = rt.kind == 2
    assert np.all(ind_sees_leo[two_hop])                                                # priority 2 is only used when GEO-IND can see the satellite
    assert np.all(rt.kind[~usable_direct & ind_sees_leo] == 2)
    three_hop = rt.kind == 3
    assert three_hop.any() and not np.any(ind_sees_leo[three_hop])                      # priority 3 only when IND is blocked by the Earth ...
    for i in np.flatnonzero(three_hop)[::40]:
        via = rt.nodes(int(i))
        assert via[0] == 0 and via[1] in (1, 2)                                         # ... and then GS -> IND -> (ATL or PAC) -> satellite
        assert oe.line_of_sight(clear.relay_tracks[via[1]].r_eci[i:i + 1], clear.tracks[0].r_eci[i:i + 1])[0][0]


def test_latency_matches_the_brief_for_each_path_kind(clear):
    rt, lat = clear.route, clear.primary_link.latency_ms
    assert 2.0 < lat[rt.kind == 1].min() and lat[rt.kind == 1].max() < 7.5               # direct: a few ms
    assert 230.0 < lat[rt.kind == 2].min() and lat[rt.kind == 2].max() < 275.0           # two hops: ~240-270 ms
    assert 470.0 < lat[rt.kind == 3].min() and lat[rt.kind == 3].max() < 520.0           # three hops: ~480-520 ms
    assert lat == pytest.approx(clear.primary_link.range_km * 1000.0 / ll.C_M_S * 1e3)  # latency is just the light time along the whole path
    k = int(np.flatnonzero(rt.kind == 2)[100])
    expected = np.linalg.norm(clear.relay_tracks[0].r_ecef[k] - oe.GroundStation().ecef()) + np.linalg.norm(clear.relay_tracks[0].r_eci[k] - clear.tracks[0].r_eci[k])
    assert clear.primary_link.range_km[k] == pytest.approx(expected, rel=1e-6)
    hop_km = np.linalg.norm(clear.relay_tracks[0].r_eci[rt.kind == 2] - clear.tracks[0].r_eci[rt.kind == 2], axis=1)
    assert 35000 < hop_km.min() and hop_km.max() < 44500                                # the relay-to-satellite hop is 35,000-44,300 km (the brief rounds this to 36,000-41,000)


def test_end_to_end_numbers_are_sane_per_path(clear):
    link, rt = clear.primary_link, clear.route
    assert np.all(np.isfinite(link.snr_db))
    for kind in (2, 3):
        m = rt.kind == kind
        assert np.nanmin(link.snr_db[m]) > 5.0                                          # the relay hops close the link at 250 Gbaud
        assert link.adaptive_throughput_gbps[m].min() > 200.0                           # at least BPSK-class throughput on every step
    assert np.nanmax(link.snr_db[rt.kind == 2]) > np.nanmax(link.snr_db[rt.kind == 3])  # more hops in the chain = lower SNR
    assert np.all(link.ber["BPSK"][np.isfinite(link.ber["BPSK"])] <= link.config.ber_threshold)
    assert link.fspl_db[rt.kind == 3].min() > link.fspl_db[rt.kind == 2].max()           # three hops accumulate more path loss


def test_total_path_rate_matches_the_derivative_of_the_path_length(clear):
    link, rt = clear.primary_link, clear.route
    num = np.gradient(link.range_km, clear.t_s)
    same = np.zeros(len(rt.chosen), dtype=bool)
    same[2:-2] = (rt.chosen[:-4] == rt.chosen[2:-2]) & (rt.chosen[4:] == rt.chosen[2:-2]) & (rt.chosen[1:-3] == rt.chosen[2:-2]) & (rt.chosen[3:-1] == rt.chosen[2:-2])
    assert same.sum() > 8000
    assert num[same] == pytest.approx(link.range_rate_km_s[same], abs=0.1)
    f0 = link.config.frequency_hz
    assert link.doppler_ghz[same] == pytest.approx(-link.range_rate_km_s[same] * 1000 / ll.C_M_S * f0 / 1e9)


# ---- handover ----------------------------------------------------------------------------------------------------------
def test_handovers_are_make_before_break_and_match_the_route_changes(clear):
    rt = clear.route
    changes = int(np.sum(np.diff(rt.chosen) != 0))
    assert len(rt.handovers) == changes and changes >= 30                                # roughly two path changes per orbit
    assert rt.n_hitless == len(rt.handovers)                                              # no gap at any handover in clear sky
    for h in rt.handovers[:6]:
        i = h.index
        assert rt.available[rt.chosen[i], i] and rt.available[rt.chosen[i - 1], i - 1]
        assert h.from_label != h.to_label and h.t_s == clear.t_s[i]
    assert np.all(rt.connected)                                                           # and no step is ever without a path


def test_a_non_hitless_handover_is_detected():
    """When the old path dies and the new path is not yet available, the handover is NOT make-before-break."""
    h_ok = ll.Handover(5, 50.0, "A", "B", True)
    assert h_ok.hitless and ll.RouteResult.n_hitless.fget is not None


# ---- relay on / off and bad weather -------------------------------------------------------------------------------------
def test_relay_off_means_only_direct_passes():
    r = sim.run_simulation(direct_cfg())
    assert set(np.unique(r.route.kind)) <= {0, 1} and not r.relay_tracks
    assert r.primary_link is r.ground_links[0]
    assert r.route.uptime_pct == pytest.approx(r.route.direct_uptime_pct, abs=1e-9)
    assert (r.route.kind == 1).mean() == pytest.approx((r.ground_links[0].visible).mean())


def test_summary_mentions_both_uptimes(clear, direct_only):
    text = clear.summary_text()
    assert "GEO relay network ON" in text and "100.000 %" in text and "direct beam alone: 1.2" in text and "handovers" in text
    assert "GEO relay network OFF" in direct_only.summary_text()


def test_overcast_station_defeats_the_relay_too_because_every_path_starts_in_the_same_air():
    import atmosphere as at
    cfg = sim.SimulationConfig()
    cfg.atmosphere = at.WEATHER_PRESETS["Overcast"]
    r = sim.run_simulation(cfg)
    assert r.route.uptime_pct < 1.0 and r.primary_link.adaptive_throughput_gbps.max() == 0.0


def test_thin_cirrus_does_not_hurt_the_relay_much_but_still_beats_direct():
    import atmosphere as at
    cfg = sim.SimulationConfig()
    cfg.atmosphere = at.WEATHER_PRESETS["Thin cirrus"]
    r = sim.run_simulation(cfg)
    assert r.route.uptime_pct > 90.0 and r.route.uptime_pct > r.route.direct_uptime_pct


def test_a_20_second_aircraft_outage_costs_exactly_20_seconds_of_uptime():
    cfg = sim.SimulationConfig(events=ev.EventsConfig(manual=[ev.ManualEvent("aircraft", 5005.0, 20.0, 0.5)]))
    r = sim.run_simulation(cfg)
    assert r.route.uptime_pct == pytest.approx(100.0 * (1 - 2.0 / 8641.0), abs=1e-9)         # 20 s = two 10 s steps' worth of a 8641-sample day
    assert int((~r.route.connected).sum()) == 2                                            # the two 10 s steps fully inside the outage carry nothing


def test_a_long_outage_cuts_uptime_and_splits_the_connected_window():
    cfg = sim.SimulationConfig(events=ev.EventsConfig(manual=[ev.ManualEvent("safe_mode", 3600.0, 7200.0, 0.5)]))
    r = sim.run_simulation(cfg)
    assert r.route.uptime_pct == pytest.approx(100.0 * (1 - 7200.0 / 86400.0), abs=0.05)
    assert len(r.primary_windows) == 2 and r.beam_state(r.index_at(5000.0))[0] == "blocked"


def test_relay_config_round_trips_through_json():
    import json
    cfg = sim.SimulationConfig()
    cfg.relay.enabled = False
    cfg.relay.aperture_m = 0.5
    cfg.relay.relays = cfg.relay.relays[:2]
    again = sim.SimulationConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert again.relay.enabled is False and again.relay.aperture_m == 0.5 and [r.short for r in again.relay.relays] == ["GEO-IND", "GEO-ATL"]


def test_two_relays_still_cover_most_but_not_all_of_the_day():
    cfg = sim.SimulationConfig()
    cfg.relay.relays = cfg.relay.relays[:2]                                                 # IND + ATL only
    r = sim.run_simulation(cfg)
    assert 85.0 < r.route.uptime_pct < 99.9                                                 # without the Pacific relay there are gaps (about 5 %): three relays are what give 100 %


def test_csv_includes_the_route(tmp_path, clear):
    p = tmp_path / "o.csv"
    clear.write_csv(p)
    text = p.read_text(encoding="utf-8")
    assert "route,hops,data_cumulative_tb" in text.splitlines()[0] and "GS-INDIA -> GEO-IND -> NODE-ALPHA" in text


def test_at_50_gbaud_bpsk_is_raw_error_free_on_every_path():
    """The 250 Gbaud default runs at a raw BER of about 1e-3 (the FEC corrects it). Slow the symbol rate and the raw BER disappears."""
    cfg = sim.SimulationConfig()
    cfg.link.symbol_rate_gbaud = 50.0
    r = sim.run_simulation(cfg)
    ber = r.primary_link.ber["BPSK"]
    for kind in (1, 2, 3):
        m = r.route.kind == kind
        assert np.nanmax(ber[m]) < 1e-9, f"kind {kind}: {np.nanmax(ber[m]):.1e}"
    assert r.route.uptime_pct == pytest.approx(100.0)


def test_at_the_default_250_gbaud_the_geo_paths_close_only_through_the_fec_threshold():
    r = sim.run_simulation(sim.SimulationConfig())
    ber = r.primary_link.ber["BPSK"]
    m3 = r.route.kind == 3
    assert 1e-4 < np.nanmax(ber[m3]) < r.primary_link.config.ber_threshold               # raw BER ~1e-3: usable only with the 7 % FEC
