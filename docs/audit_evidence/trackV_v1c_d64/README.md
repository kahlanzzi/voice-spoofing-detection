# Track V V1c D64 recipe re-freeze evidence

This directory records the one-time D64 re-freeze of the complete 6,650-row selection recipe under
the declared training environment. It is recipe evidence only: no audio, generated waveforms,
selection-set tensors or resource cache are committed here.

## Run identity

| Item | Value |
|---|---|
| Reviewed code commit | `41ad8bc8b95ef8d9d178e8b6986fcf9a192e48aa` |
| Colab runtime / GPU | `2026.10` / Tesla T4 |
| Python / torch | `3.13.16` / `2.11.0+cu130` |
| numpy / scipy / soundfile | `2.5.3` / `1.18.1` / `0.14.0` |
| librosa | `1.0.0` |
| FFmpeg | `6.1.1-3ubuntu5` |
| Result | passed |

The repository was a clean detached checkout of the reviewed commit. The official runtime and
dependency guards passed before the output directory was created.

## Result

- Rows and sources: 6,650 rows for 1,330 val-A sources.
- Unchanged waveform hashes: 5,320.
- Changed waveform hashes: 1,330, all and only the `reverberation` condition.
- Sources, conditions, parameters, seeds, resources, resource halves and row order are identical.
- `recorded_noise_10db` and `reverberation` both continue to use the selection resource half.

The complete row-by-row proof is in
[`selection_recipe_refreeze_comparison.json`](selection_recipe_refreeze_comparison.json).

## Files and hashes

| File | Bytes | SHA-256 | Purpose |
|---|---:|---|---|
| `selection_set_s.old.csv` | 2,563,916 | `71a340f009b5b0a20e295f4031c15fe2471b1174444b4b0fa2d3014f34c72533` | V1b recipe, preserved byte for byte |
| `selection_set_s_report.old.json` | 1,588 | `48594032d70fec52430908690e5bd279f16a322335a2514e32a1d309059cdf1f` | V1b report, preserved byte for byte |
| `selection_set_s.csv` | 2,563,916 | `70a2105425838c64b9bf674fbdbf3f3e121fc97ae2d6d3bf647c9aceff12d0a1` | D64 recipe; canonical copy is in `data/manifests/augmentation/` |
| `selection_set_s_report.json` | 2,145 | `3eab3192e7a24695f9713cc0ec7541216cf89f2fb4b12d931d6aace1c55b936c` | D64 report; canonical copy is in `data/manifests/augmentation/` |
| `selection_recipe_refreeze_comparison.json` | 466,318 | `acc3ea3acd719fa3a552628bca69e5a642865db5c1c549dd26568f2bb885c3db` | Old/new invariant proof and every changed hash |

The old and new semantic recipe hashes are respectively
`8cf4858c51f78a1fd5bce78042f7d490529eb1230a149fe52a0a1f856e9046a7` and
`95dfe79e0b2fb86c1e055beb793aeccbd6664505b71ec8c52c7689c918835783`.

## Remaining V1c work

This evidence closes the D64 recipe re-freeze only. V1c still requires the real selection-set
materialization/verification and the deterministic training-half MUSAN/RIRS resource-cache build;
those large outputs remain outside Git.
