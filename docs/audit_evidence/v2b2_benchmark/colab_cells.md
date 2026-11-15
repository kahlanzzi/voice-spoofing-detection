# V2b-2 — D49 batch-size benchmark on Colab T4 (runtime 2026.10)

Runtime: **T4 GPU, runtime version 2026.10**. Laptop plugged in, sleep off, no key presses during long cells.
Paste each block into its own cell and run them in order. If any `assert` fails, stop and send Donia's
reviewer the full cell output.

## Cell 1 — settings
```python
REPO_URL = 'https://github.com/kahlanzzi/voice-spoofing-detection.git'
COMMIT = '9c46bafba7e220867f3b5fdd7b4d66ebc0573991'  # main after PR #129 (V2b-1)
DRIVE_ROOT = '/content/drive/MyDrive/voice-spoofing-detection'
OUT = '/content/v2b2'  # outside the repo (run_benchmark refuses the repo)
```

## Cell 2 — Drive, clone, clean checkout
```python
from datetime import datetime, timezone
from pathlib import Path
import hashlib, json, os, shutil, subprocess, sys
from google.colab import drive
drive.mount('/content/drive')

STAMP = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
DEST = Path(DRIVE_ROOT) / 'processed' / 'metadata' / 'v2b2_benchmark' / STAMP
assert not DEST.exists(), DEST
REPO = Path('/content/voice-spoofing-detection')
if REPO.exists():
    shutil.rmtree(REPO)
subprocess.run(['git', 'clone', '-q', REPO_URL, str(REPO)], check=True)
subprocess.run(['git', 'checkout', '-q', '--detach', COMMIT], cwd=REPO, check=True)

def git(*a):
    return subprocess.run(['git', *a], cwd=REPO, check=True, capture_output=True, text=True).stdout

def assert_clean():
    assert git('rev-parse', 'HEAD').strip() == COMMIT
    assert not git('status', '--porcelain'), git('status', '--porcelain')

def sha256(p):
    h = hashlib.sha256()
    with open(p, 'rb') as f:
        for chunk in iter(lambda: f.read(1 << 20), b''):
            h.update(chunk)
    return h.hexdigest()

assert_clean()
os.chdir(REPO)
print('HEAD', COMMIT, '| Drive dest', DEST)
```

## Cell 3 — environment gate (D58) and packages
```python
import torch
GPU = subprocess.run(['nvidia-smi', '--query-gpu=name,memory.total', '--format=csv,noheader'],
                     check=True, capture_output=True, text=True).stdout.strip()
assert 'T4' in GPU, GPU
assert torch.__version__ == '2.11.0+cu130', torch.__version__
subprocess.run([sys.executable, '-m', 'pip', 'install', '-q', '-r', 'requirements.txt',
                '-r', 'requirements-cnn.txt'], check=True)
import platform
assert platform.python_version() == '3.13.16', platform.python_version()
assert_clean()
print(GPU, '| python', platform.python_version(), '| torch', torch.__version__,
      '| cpus', os.cpu_count())
```

## Cell 4 — tests of the code being run (v2 + config + models), torch required
```python
import re
r = subprocess.run([sys.executable, '-m', 'pytest', '-q', '-rs', '-p', 'no:cacheprovider',
                    'tests/training', 'tests/tuning/test_v2_benchmark.py', 'tests/models'],
                   env={**os.environ, 'REQUIRE_TORCH_TESTS': '1'}, capture_output=True, text=True)
print(r.stdout[-2000:])
assert r.returncode == 0, r.stdout[-3000:] + r.stderr[-3000:]
skips = [l for l in r.stdout.splitlines() if l.startswith('SKIPPED')]
assert all('AASIST_REF_DIR' in l for l in skips), skips
assert_clean()
```
(One skip is acceptable: the AASIST faithfulness test, which needs `AASIST_REF_DIR`.)

## Cell 5 — GPU benchmark, run 1 and run 2 (decisive part; no codec, no data)
```python
def bench(name, *extra):
    out = Path(OUT) / name
    r = subprocess.run([sys.executable, '-m', 'src.tuning.v2_benchmark', '--out', str(out), *extra],
                       capture_output=True, text=True)
    print(r.stdout)
    assert r.returncode == 0, r.stderr[-3000:]
    return json.loads((out / 'benchmark.json').read_text())

runs = [bench(f'gpu_run{i}', '--device', 'cuda', '--throughput-items', '0') for i in (1, 2)]
for rep in runs:
    for row in rep['rows']:
        print(row['candidate'], row['batch_size'], row['status'], row['peak_allocated_bytes'],
              None if row['mean_step_seconds'] is None else round(row['mean_step_seconds'], 4))
    print('chosen', rep['chosen_batch_size'], '| errors', rep['choice_errors'],
          '| vram', rep['total_vram_bytes'])
assert runs[0]['chosen_batch_size'] == runs[1]['chosen_batch_size'], 'runs disagree: stop'
assert not runs[0]['choice_errors'] and all(runs[0]['chosen_batch_size'].values())
assert_clean()
print('CHOSEN', runs[0]['chosen_batch_size'])
```

## Cell 6 — CPU throughput (planning only; codec unpinned until V1b)
```python
print(subprocess.run(['ffmpeg', '-hide_banner', '-version'], capture_output=True, text=True).stdout.splitlines()[0])
enc = subprocess.run(['ffmpeg', '-hide_banner', '-encoders'], capture_output=True, text=True).stdout
print('libopus encoder:', 'libopus' in enc, '| libmp3lame encoder:', 'libmp3lame' in enc)
cpu = bench('cpu_throughput', '--device', 'cpu', '--candidates', 'cnn-aug', '--batch-sizes', '16',
            '--throughput-items', '20', '--codec-unpinned-timing')
t = cpu['throughput']
print('codec_timing', cpu['codec_timing'], '|', cpu['ffmpeg_version_observed'])
for fam, s in t['families'].items():
    print(f"{fam:15s} mean {s['mean_seconds']:.4f}s  p95 {s['p95_seconds']:.4f}s")
for stage, s in t['stages'].items():
    print(f"{stage:15s} mean {s['mean_seconds']:.4f}s")
print('estimated epoch seconds', json.dumps(t['estimated_epoch_seconds'], indent=1))
assert_clean()
```
If this cell fails (e.g. no `libopus`), skip it and send the error: the GPU results of Cell 5 are
already complete and are what V2b-2 needs.

## Cell 7 — copy to Drive and print the hashes
```python
DEST.mkdir(parents=True)
for p in sorted(Path(OUT).glob('*/benchmark.json')):
    target = DEST / f'{p.parent.name}.json'
    shutil.copy2(p, target)
    assert sha256(target) == sha256(p)
    print(target.name, sha256(target))
print('saved to', DEST)
```
Then download the three JSON files from that Drive folder (`gpu_run1.json`, `gpu_run2.json`,
`cpu_throughput.json`) and send them, plus the printed output of Cells 3, 5, 6 and 7.
