# BATADAL confirmation protocol (frozen 2026-11-19, before any model was fitted on BATADAL)

Purpose: test the TailGuard construction on a second, different industrial control system (C-Town water
distribution network, BATADAL competition data) in response to the review of manuscript mathematics-4731067.
Everything below was fixed before any BATADAL model was fitted and before any detection result was seen.
Only the structure of the three files (columns, row counts, dates, constant columns) was inspected beforehand.

## Data roles
- `BATADAL_dataset03.csv` (8,761 hourly rows, normal operation): split in time order 60/20/20 into
  fitting, validation and calibration.
- `BATADAL_dataset04.csv` (4,177 rows) and `BATADAL_test_dataset.csv` (2,089 rows): evaluation only.
- Attack intervals: the 14 intervals published with the competition (7 per evaluation file), as listed in
  `batadal_attacks.json`. The `ATT_FLAG` column is dropped at preparation and is never read for fitting,
  selection, calibration or scoring; it is used once, in the audit, to cross-check the intervals of dataset04.
- Tags constant in dataset03 are dropped. Levels, flows and pressures are forecast targets; pump and valve
  statuses are discrete inputs. Scaling: median and interquartile range of the fitting data, as on HAI.

## Carried over from HAI 23.05 without change (sample units)
- Model: two-layer GRU (64 units), window W = 30, lambda_inv = 1.0 (on constraint risk), lambda_phys = 0.1,
  environment-balanced sampling, batch 256, Adam 1e-3, at most 100 epochs, patience 10, seeds 17, 29, 43, 71, 101.
- No hyperparameter search and no model selection on BATADAL.
- Comparators trained the same way: matched GRU (both penalties off, pooled sampling) and A1 (lambda_phys = 0),
  five seeds each; ridge autoregression (alpha = 1.0), PCA reconstruction (k = 20) and persistence.
- Score: tail surprise per channel (empirical ranks to the 99th percentile, exponential excess above), mean of the
  three largest surprises per stream, causal mean over 30 samples, fused score with beta = 0.5 (prespecified score);
  the channel-maximum score is reported beside it, as on HAI 22.04.
- Threshold: monotone-closure rule on calibration data; block-maximum order-statistic level of Proposition 5.

## Adaptations required by hourly data (fixed here)
1. Reference sample: every eligible validation row (step 1 instead of 5); validation loss on every row.
2. Alarm budget: one false alarm episode per 30 days (720 h) of normal operation.
3. Proposition 5: day-long blocks (24 samples) on the calibration record; k = max(1, floor((B + 1)/30)).
4. History requirement: 30 samples (the window) instead of 120. Ridge lags: 1, 2, 5, 10, 30.
5. Training environments for the invariance penalty: three equal consecutive time bands of the fitting data
   (BATADAL documents no operating set point).
6. Constraint registry: relations documented in the C-Town network definition (`ctown_map.inp` of the epanetCPA
   toolbox: which pumps and valve feed which tank through the control rules; the junctions on either side of each
   pump and of valve V2). Coefficients by Huber regression on fitting data; screening rule of HAI unchanged
   (95th-percentile residual in every environment at most twice the pooled value, R2 at least 0.5, documented signs).
   Subsampling of fitting rows for the regression is removed (every row is used).

## Reported quantities
- At the calibrated threshold: event recall over the 14 attacks, false alarm episodes per 30 days of normal
  evaluation data, share of normal hours in alarm; block-level share of alarmed normal days against k/(B + 1).
- Published metrics of the competition on the official test file: S score (time-to-detection and classification
  components, gamma = 0.5) and F1, at the same calibrated threshold.
- Sensitivity, fixed here and labelled as secondary: causal mean over 1 and 6 samples instead of 30, each with
  its own closure threshold on calibration data. The primary result is the 30-sample score.
