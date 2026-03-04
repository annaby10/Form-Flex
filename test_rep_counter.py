"""
test_rep_counter.py — Unit test for the squat rep-counting algorithm.
Simulates angle data that mimics a person doing 3 complete squats:
  standing (160°) → squat (70°) → standing (160°) → ...
Verifies that exactly 3 reps are counted.
"""
import numpy as np

# ── Replicate the exact rep-counting logic from app.py ──────────────────────
state = {"count": 0, "dir": 0}

def process_angle(knee_angle):
    depth_pct = float(np.interp(knee_angle, (70, 160), (100, 0)))
    depth_pct = max(0.0, min(100.0, depth_pct))

    if depth_pct >= 80 and state["dir"] == 0:
        state["dir"] = 1              # descended to squat depth
    if depth_pct <= 20 and state["dir"] == 1:
        state["count"] += 1           # full rep completed
        state["dir"] = 0
    return depth_pct

# ── Simulate 3 squat cycles ────────────────────────────────────────────────
# Each cycle: ease down from 160° → 70°, then back up to 160°
FRAMES_PER_SQUAT = 60

for rep in range(3):
    # Going DOWN: 160 → 70 (standing to squatting)
    for a in np.linspace(160, 70, FRAMES_PER_SQUAT):
        pct = process_angle(a)

    # Coming UP: 70 → 160 (squatting to standing)
    for a in np.linspace(70, 160, FRAMES_PER_SQUAT):
        pct = process_angle(a)

print(f"\n✅ Rep count after 3 simulated squats: {state['count']}")
assert state["count"] == 3, f"❌ Expected 3 reps, got {state['count']}"
print("✅ Rep counting algorithm is CORRECT\n")

# ── Test depth percentages at key angles ─────────────────────────────────
print("Depth % at key knee angles:")
for angle in [160, 140, 120, 100, 90, 80, 70]:
    pct = float(np.interp(angle, (70, 160), (100, 0)))
    pct = max(0, min(100, pct))
    bar = "█" * int(pct / 5)
    print(f"  Knee {angle:3}° → depth {pct:5.1f}%  {bar}")
