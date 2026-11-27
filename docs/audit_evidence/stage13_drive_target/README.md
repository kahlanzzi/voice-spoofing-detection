# Stage 13 approved Drive target — pre-run audit

This note records a read-only preparation audit. It does **not** authorize the
Stage 13 claim, access external audio, or establish that the durability and
sharing gate has passed.

## Target observed on 27 November 2026

| Item | Observed value |
|---|---|
| Google account | `khalil.dridi.dk@gmail.com` |
| Folder | `voice-spoofing-detection-stage13` |
| Folder ID | `17BxyuWYp2DUTnHbyUEUuyDA0Gv8HJuJD` |
| Registry path | `/content/drive/MyDrive/voice-spoofing-detection-stage13/registry/` |
| Output path | `/content/drive/MyDrive/voice-spoofing-detection-stage13/output/` |
| Registry folder ID | `13PXxUfwXCPJp_bcI0Kz3eg4OA-BYYTJp` |
| Output folder ID | `143PlXlHp3uKl4qgOjngkz0WhmTWXSvpb` |
| Probe file | `stage13-approved-drive-rehearsal.txt` |
| Probe size | 259 bytes |
| Probe SHA-256 | `3fb7482a130a19c99730655001ec73aeefbc545f631aacfed47024d03c52430b` |

The probe states that it is harmless dummy data, records repository commit
`768b75a3154f20503ba4742225ba80ae55f000a6`, and states that no dataset audio
was accessed and no official claim was created. A byte-identical copy was
observed in `registry/`. The root folder, `registry/`, `output/`, and probe were
read only; no Drive bytes were changed during this audit.

## Checks and limitations

- [verified] The folder and both required subfolders exist at the exact paths
  above.
- [verified] Reconstructing the probe's observed UTF-8 bytes, including its
  trailing line feed, produced 259 bytes and the SHA-256 above.
- [verified limitation] The folder metadata reported `shared: false`. The
  required Viewer access for Donia, Iheb, and Malak therefore was not
  established.
- [verified limitation] The saved notebook
  `Stage 13 - Drive persistence read-back.ipynb` (Drive file
  `1Fw8Jv29sxA0ny-8tu91uQLpL6eGIR7MA`) proves only Entry 198's personal
  `stage13-scratch` read-back. Its output verifies probe
  `929558cd046241f2962b5de89be68b7b` with SHA-256
  `70520a98873a6685ff395b6bf6458bf1f72a4accf04dce864d42a974e3d82915`;
  it does not read this approved target.
- [pending] Read the approved probe from a new, separate Colab runtime mounted
  read-only and record the full observed path, byte count, SHA-256, UTC time,
  notebook/file reference, and successful output.
- [pending] Share the target folder as Viewer with the other three team members
  and record their exact approval/review references. Do not infer email
  addresses or approval from names.

The folder must not be renamed, moved, or emptied before the official registry
is committed and independently preserved. Even after these two pending items
pass, the official run remains blocked on the D48 F5-or-cutoff model set, the
run-session D58 recheck, reviewed invocation/configuration, and all-four review.
