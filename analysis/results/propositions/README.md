# Propositions 4 and 5: evaluation and simulation

- `prop5_eval.py <root> <prefix> <runs|runs_max> <out.csv> [block_hours]`: block-maximum threshold of Proposition 5 (k = 2) against the monotone-closure threshold on the frozen scores; recomputes every closure threshold and checks it against the saved one (difference 0.0 on all runs). Outputs `p5_*` (1 h blocks) and `p5h_*` (0.5 h blocks) for HAI 23.05 and HAI 22.04, prespecified and max scores.
- `sim.py <scripts_dir> <out.csv>`: AR(1) simulation of Appendix B (66 channels, lag-1 correlation 0, 0.5, 0.9, 0.99, 0.999; three repetitions). `sim_blocks.csv`: 20 further repetitions at 0.99 (repetitions 3-22) for the block guarantee.
- `figB1.py`: Figure B1 from `sim.csv`.
- Model-implied threshold of Proposition 4: tau_A = log10(e * 86400 * m), m = 66 (HAI 23.05) or 67 (HAI 22.04).
