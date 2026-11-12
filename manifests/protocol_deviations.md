# TailGuard protocol deviations log (append-only)

## D1 - 2026-10-13 - HAI test described as official split, not chronological
Official test files are interleaved in time with training files (audit note 5). Wording changed with author approval.
Test outcomes visible: NO.

## D2 - 2026-10-13 - Label join for hai-test2 is positional
label-test2.csv timestamps are minute-truncated; positional join reproduces all 38 official intervals.
Test outcomes visible: NO (only interval boundaries from the official summary files were compared).

## D3 - 2026-10-14 - Calibration threshold uses monotone closure
"Smallest threshold with <= 1 episode/24 h" admits a continuously alarming detector (one long episode).
Rule now: scan down from the maximum calibration score and stop at the first violation.
Test outcomes visible: NO.

## D4 - 2026-10-14 - Common score policy replaces mean of median-scaled errors and the no-smoothing rule
Per-channel tail surprise (ECDF of normal validation errors), mean of the top-3 channels, 30 s causal mean.
Reason: with the original rule every method detected 0/36 development injections because legitimate set-point
transients (e.g. P1_PCV02D) dominate calibration. Chosen on development injections in validation data and validation
false alarms only; applied identically to all methods. Test outcomes visible: NO.

## D5 - 2026-10-14 - Training environments are operating regimes; invariance penalty is selective (author approval)
Environments = P4_ST_PS regime bands within fitting data (day-based environments kept as an ablation).
Risk-variance penalty applies to process-constraint residual risk (selective invariance); whole-forecast V-REx kept
as an ablation. Test outcomes visible: NO.

## D6 - 2026-10-14 - Development injections on validation data
36 sensor manipulations (bias, freeze, ramp; 60-600 s; seed 17) injected into a copy of validation data provide a
detection signal for design decisions. Never reported as results. Test outcomes visible: NO.

## D7 - 2026-10-14 - Tail surprise extended beyond the observed maximum
The ECDF surprise saturated at log10(n), so a constraint violation 3x larger than anything in normal data scored the
same as a marginal one. Above the 99th percentile an exponential tail (mean excess, floor 1e-3) is used. Applied to
every method; baselines re-run. Test outcomes visible: NO.

## D8 - 2026-10-14 - Constraint registry uses level forms and an identifiability rule
Autoregressive forms (x_t on x_{t-1}) cancel sensor bias and freeze; relations are written as level forms
(position vs recent command; output vs recent demand; 30 s mass balance). Retained only if fitting R2 >= 0.5, signs
agree with documented physics, and regime q95 ratio <= 2 (fitting data only). 11 of 14 retained. Test outcomes visible: NO.

## D9 - 2026-10-14 - Ablations A4, A5, A9 computed from saved score streams; A6, A8 (ratio) moot
A4/A5 (beta = 0 or 1) and A9 (alarm budget) need no retraining. The stability screen rejected no relation, so A6 equals
the full model; screening-ratio sensitivity (A8) is likewise moot. A7 (randomised tags) is deferred to a follow-up run.
Test outcomes visible: NO.

## D10 - 2026-10-14 - Model selection by development-injection detection (author approval)
The prespecified rule (minimum worst-regime validation forecast loss) systematically selects lambda_phys = 0, because
the physics term trades forecast accuracy for physical consistency; after 15 of 27 search runs the loss-optimal setting
was the plain GRU (W60, inv 0, phys 0), which detected 19% of development faults versus 33-42% with lambda_phys > 0.
New rule: maximise event recall on the 36 development injections in validation data at the calibrated budget of one false
alarm per 24 h (beta = 0.5); ties broken by lower validation forecast loss. Validation data and synthetic faults only.
Test outcomes visible: NO.
