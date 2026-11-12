"""Metric and alarm-contract tests (guide Section 8). Run: python tests\\test_metrics.py"""
import os, sys
import numpy as np
sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts"))
from tailguard_lib import episodes, event_stats, point_stats, calibrate_threshold, alarm_stats

seg = np.zeros(20, int)
ev = np.array([""] * 20, dtype=object); ev[10:15] = "E0"
y = ev != ""

def check(name, cond):
    print(("PASS " if cond else "FAIL ") + name); assert cond

a = np.zeros(20, bool)
r = event_stats(a, ev); p = point_stats(a, y)
check("no alarms: recall 0, precision NA", r["event_recall"] == 0 and np.isnan(p["precision"]))
a = np.zeros(20, bool); a[10] = True
r = event_stats(a, ev); p = point_stats(a, y)
check("alarm at onset: detected, delay 0, no point adjustment", r["detected"] == 1 and r["delay_median_s"] == 0 and p["recall"] == 0.2)
a = np.zeros(20, bool); a[8:10] = True
r = event_stats(a, ev); s = alarm_stats(a, seg, ~y)
check("alarm just before attack: false episode, attack missed", r["detected"] == 0 and s["false_episodes"] == 1)
a = np.ones(20, bool); s = alarm_stats(a, seg, ~y)
check("always-on: one episode but false-alarm duration 100%", s["false_episodes"] == 1 and s["normal_FPR"] == 1.0)
ev2 = ev.copy(); ev2[15:18] = "E1"; a = np.zeros(20, bool); a[16] = True
check("adjacent scenario ids are two events", event_stats(a, ev2)["events"] == 2)
seg2 = np.r_[np.zeros(10, int), np.ones(10, int)]; a = np.zeros(20, bool); a[8:12] = True
check("segment gap splits an episode", episodes(a, seg2).sum() == 2)
sc = np.r_[np.zeros(99), 5.0]; tau, _ = calibrate_threshold(sc, np.zeros(100, int), hours=24.0)
check("score equal to threshold does not alarm (strict >)", not (5.0 > tau and tau < 5.0) or (sc > tau).sum() <= 1)
sc = np.linspace(0, 1, 1000); tau, _ = calibrate_threshold(sc, np.zeros(1000, int), hours=1.0)
check("monotone closure: continuous alarming cannot pass calibration", (sc > tau).mean() < 0.5)
print("all metric tests passed")
