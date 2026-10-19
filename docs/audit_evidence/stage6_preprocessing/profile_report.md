# Stage 6 preprocessing profile validation

- Profile: `preprocessing-v1`
- Profile fingerprint: `5b4f5849a9bb63f9016dd79969ff5bfb9e37006acbff1e26bbfee4ee2bad68f2`
- Code commit: `6e1718547756ef04744ce0b3b1b73bb839920052`
- Git dirty: `false`
- FoR rows processed: 17721
- Audio scope: FoR only; DEEP-VOICE not accessed

## Checks
- f2_manifest_verified: True
- all_sample_hashes_verified: True
- all_rows_processed: True
- outputs_finite: True
- padding_measurements_valid: True
- loudness_decision_rule_pass: True
- invariance_pass: True
- invariance_rows_complete: True

## Shortcut re-screen
| scope | comparison | feature | rows | auc_separability | flag |
|---|---|---|---|---|---|
| train | FAKE_vs_REAL | rms_dbfs | 12404 | 0.6986 | LOW |
| train | FAKE_vs_REAL | silence_frac | 12404 | 0.6128 | LOW |
| train | FAKE_vs_REAL | bw99_hz | 12404 | 0.6549 | LOW |
| train | FAKE_vs_REAL | padding_fraction | 12404 | 0.5034 | LOW |
| train | FAKE_mp3_vs_REAL | rms_dbfs | 11474 | 0.6987 | LOW |
| train | FAKE_mp3_vs_REAL | silence_frac | 11474 | 0.5888 | LOW |
| train | FAKE_mp3_vs_REAL | bw99_hz | 11474 | 0.666 | LOW |
| train | FAKE_mp3_vs_REAL | padding_fraction | 11474 | 0.505 | LOW |
| train | FAKE_other_vs_REAL | rms_dbfs | 7090 | 0.6983 | LOW |
| train | FAKE_other_vs_REAL | silence_frac | 7090 | 0.7499 | LOW |
| train | FAKE_other_vs_REAL | bw99_hz | 7090 | 0.5916 | LOW |
| train | FAKE_other_vs_REAL | padding_fraction | 7090 | 0.5059 | LOW |
| all | FAKE_vs_REAL | rms_dbfs | 17721 | 0.6976 | LOW |
| all | FAKE_vs_REAL | silence_frac | 17721 | 0.6124 | LOW |
| all | FAKE_vs_REAL | bw99_hz | 17721 | 0.6562 | LOW |
| all | FAKE_vs_REAL | padding_fraction | 17721 | 0.5037 | LOW |
| all | FAKE_mp3_vs_REAL | rms_dbfs | 16392 | 0.697 | LOW |
| all | FAKE_mp3_vs_REAL | silence_frac | 16392 | 0.5883 | LOW |
| all | FAKE_mp3_vs_REAL | bw99_hz | 16392 | 0.6659 | LOW |
| all | FAKE_mp3_vs_REAL | padding_fraction | 16392 | 0.5052 | LOW |
| all | FAKE_other_vs_REAL | rms_dbfs | 10129 | 0.7006 | LOW |
| all | FAKE_other_vs_REAL | silence_frac | 10129 | 0.7501 | MODERATE |
| all | FAKE_other_vs_REAL | bw99_hz | 10129 | 0.6004 | LOW |
| all | FAKE_other_vs_REAL | padding_fraction | 10129 | 0.5049 | LOW |

## Post-trim padding by class and subgroup
| scope | grouping | value | rows | mean_padding_samples | median_padding_samples | p95_padding_samples | max_padding_samples | mean_padding_fraction |
|---|---|---|---|---|---|---|---|---|
| train | label | FAKE | 6244 | 118 | 0 | 320 | 14080 | 0.003688 |
| train | label | REAL | 6160 | 147.8 | 0 | 320 | 15360 | 0.004619 |
| train | subgroup | FAKE_mp3 | 5314 | 122.8 | 0 | 320 | 14080 | 0.003839 |
| train | subgroup | FAKE_other | 930 | 90.49 | 0 | 320 | 7040 | 0.002828 |
| train | subgroup | REAL | 6160 | 147.8 | 0 | 320 | 15360 | 0.004619 |
| all | label | FAKE | 8921 | 119.9 | 0 | 320 | 15360 | 0.003747 |
| all | label | REAL | 8800 | 146.1 | 0 | 320 | 15360 | 0.004567 |
| all | subgroup | FAKE_mp3 | 7592 | 126.6 | 0 | 320 | 15360 | 0.003955 |
| all | subgroup | FAKE_other | 1329 | 81.87 | 0 | 320 | 7040 | 0.002558 |
| all | subgroup | REAL | 8800 | 146.1 | 0 | 320 | 15360 | 0.004567 |

## Invariance summary
| rows | max_relative_l2_error | min_cosine_similarity |
|---|---|---|
| 180 | 0.002124 | 1 |

All checks pass: **true**.
