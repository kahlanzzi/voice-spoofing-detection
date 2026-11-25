"""Run inside the built API image (CI e2e job; by hand: docs/setup.md), read from stdin so it is not in the image:

    docker compose exec -T -e VSD_CHECK_PROFILE="$(cat configs/preprocessing.yaml)" api python - < deploy/image_check.py

Checks that the image holds the scoring modules and none of the research code, that torch is the CPU build, and that
one analysis of a synthetic bundle works under the container's own settings (read-only root, numba cache folder).
The profile is public and not in the image, so it comes in as text. Prints one line per check; any failure exits 1.
"""
import hashlib
import importlib
import importlib.util
import os
import sys
import tempfile
from pathlib import Path

ABSENT = ("src.tuning", "src.evaluation", "src.training.train_cnn", "src.audit", "src.features", "src.augmentation",
          "pandas", "wandb", "pytest")
SCORING = ("api.bundle_analyzer", "src.bundle.model_loader", "src.preprocessing.waveform", "src.preprocessing.features",
           "src.risk.policy", "src.training.normalization")
NEVER_LOADED = ("src.preprocessing.validation", "src.preprocessing.benchmark")
LEDGER = "c9fc2e230c0883028459d0b2b8de4f6ee537676cd5a4e259e9d0f545c60943cc"


def check(ok, what):
    print(("ok    " if ok else "FAIL  ") + what, flush=True)
    if not ok:
        sys.exit(1)


def not_loaded(when):
    loaded = [name for name in NEVER_LOADED if name in sys.modules]
    check(not loaded, f"validation and benchmark not imported {when} {loaded}")


def synthetic_bundle(folder: Path) -> Path:
    import numpy as np
    import torch
    from src.bundle.bundle import build_bundle
    from src.models.cnn import build_cnn
    from src.models.cnn_checkpoint import checkpoint_sha256, save_checkpoint
    from src.preprocessing.profile import load_profile
    from src.risk.calibration import PlattCalibration
    from src.risk.policy import RiskPolicy

    profile = folder / "preprocessing.yaml"
    profile.write_text(os.environ["VSD_CHECK_PROFILE"] + "\n", encoding="utf-8")
    torch.manual_seed(2026)
    meta = {"schema_version": 1, "version": "cnn-check", "n_blocks": 3, "channels": [16, 32, 64], "dropout": 0.3,
            "input_shape": [1, 64, 197], "learning_rate": 0.001, "seed": 0, "best_epoch": 1, "val_a_eer": 0.5,
            "code_commit": "0" * 40, "environment": {"python": "check"},
            "profile_fingerprint": load_profile(profile).fingerprint(), "f2_manifest_sha256": "0" * 64,
            "train_cache_npz_sha256": "0" * 64, "val_a_cache_npz_sha256": "0" * 64, "determinism": {"seed": 0}}
    ckpt = save_checkpoint(build_cnn(3).eval(), np.full(64, -6, "float32"), np.full(64, 2, "float32"), meta,
                           folder / "ckpt")
    policy = folder / "risk_policy.json"
    policy.write_text(RiskPolicy(
        version="policy-check", calibration=PlattCalibration(a=1.0, b=0.0, n_spoof=10, n_bonafide=10), threshold=0.5,
        half_width=0.1, target_deferral=0.1, n_spoof_fit=10, n_bonafide_fit=10, deferral_on_fit=0.1).to_json(),
        encoding="utf-8")
    build_bundle(folder / "bundle", checkpoint_dir=ckpt, expected_checkpoint_sha256=checkpoint_sha256(ckpt),
                 profile=profile, policy=policy, bundle_version="bundle-check", model_version="cnn-check",
                 profile_version="preprocessing-v1", code_commit="0" * 40, protocol_version="1.0",
                 f2_ledger_sha256=LEDGER, created_at="2026-11-24T12:00:00Z")
    return folder / "bundle"


def main():
    print("python", sys.version.split()[0], "uid", os.getuid(), "cwd", os.getcwd())
    for path in ("/app", "/app/src", os.path.dirname(os.__file__)):
        check(not os.access(path, os.W_OK), f"{path} is not writable")
    for name in ABSENT:
        parent = name.rpartition(".")[0]
        check(importlib.util.find_spec(name) is None, f"{name} is not in the image" + (f" ({parent} is)" if parent else ""))
    for name in SCORING:
        importlib.import_module(name)
    check(True, f"imported {', '.join(SCORING)}")
    import torch
    check(torch.__version__ == "2.11.0+cpu", f"torch {torch.__version__}")
    not_loaded("by the imports")

    import numpy as np
    import soundfile as sf
    from api.bundle_analyzer import BundleAnalyzer
    with tempfile.TemporaryDirectory() as tmp:
        bundle = synthetic_bundle(Path(tmp))
        pin = hashlib.sha256((bundle / "bundle_manifest.json").read_bytes()).hexdigest()
        analyzer = BundleAnalyzer(str(bundle), pin)
        clip = Path(tmp) / "clip.wav"
        sf.write(clip, np.random.default_rng(2026).uniform(-0.2, 0.2, (3 * 44100, 2)), 44100, subtype="PCM_16")
        outcome = analyzer.analyze(clip)
    check(outcome.model_version == "cnn-check", f"one analysis: {outcome.prediction} {outcome.risk_level}")
    check(torch.get_num_threads() == 1, "one torch thread")
    not_loaded("by the analysis")
    print("image check OK")


main()
