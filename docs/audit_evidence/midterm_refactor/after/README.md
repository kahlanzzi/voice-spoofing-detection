# Part 6 after measurements

Khalil ran these checks in the same local Windows Python environment as the before
capture. The two measured source edits are committed in `de943333c9f8e51416666b84f37d319cd8cac4b9`.
The report records the commands, outputs and limitations. No governed data was read.

- `coverage.json`, `complexity.json`, `lint.json`: byte-identical copies of the
  original temporary captures; Git attributes preserve their bytes.
- `console.txt`: user-pasted coverage/static-tool console transcript, normalized to
  UTF-8/LF with trailing whitespace removed; content values unchanged.
- `tool-versions.txt`: same normalization; identical to the stored before capture.
- `summary.json`: calculated comparison, source hashes, exact reported test results
  and provenance; raw and normalized file hashes are distinguished.

The uninstrumented suite (937 passed in 57.15s) and governance run (3 passed in
0.18s) were supplied as screenshots; they are not included in the pasted console
transcript. The instrumented run is 937 passed in 85.53s. No speedup is claimed.
Original local folder: `voice-spoofing-part6-after-20261031-233811`.
The `.coverage` database, local environment, audio and models are excluded.
