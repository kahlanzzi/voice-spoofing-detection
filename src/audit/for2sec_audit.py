"""Stage 1 — full audit of the FoR-2sec development dataset.

Usage (Colab, after copying the dataset to local disk):
    python -m src.audit.for2sec_audit --root /content/data/for_2sec \
        --out /content/audit_out/for2sec --revision <HF commit sha>

Reads every file. Writes a per-file inventory, duplicate and near-duplicate
tables, and a markdown summary. It never modifies the dataset.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np
import pandas as pd
import soundfile as sf

from .common import (AuditError, Thresholds, auc_separability, df_to_md, environment_info,
                     filename_signature, filename_tokens, find_audio_files, header_info,
                     infer_label, infer_split_dir, sha256_file, write_json)

FP_SR, FP_MELS, FP_TBINS = 16_000, 40, 25  # near-duplicate fingerprint geometry
NUMERIC_SHORTCUT_FEATURES = ["duration_s", "size_bytes", "samplerate", "channels",
                             "rms_dbfs", "peak_dbfs", "clip_frac", "dc_offset",
                             "lead_silence_ms", "trail_silence_ms", "silence_frac", "bw99_hz"]
CATEGORICAL_SHORTCUT_FEATURES = ["ext", "format", "subtype", "samplerate", "channels",
                                 "filename_signature"]


# --------------------------------------------------------------------------- per-file
def signal_stats(x: np.ndarray, sr: int, th: Thresholds) -> dict:
    """Signal-level statistics on the decoded (mono-mixed) waveform."""
    mono = x.mean(axis=1) if x.ndim == 2 else x
    eps = 1e-12
    rms = float(np.sqrt(np.mean(mono ** 2)) + eps)
    peak = float(np.max(np.abs(mono)) + eps)
    hop = max(1, int(0.02 * sr))  # 20 ms frames
    n = len(mono) // hop
    if n == 0:
        frame_db = np.array([20 * np.log10(rms)])
    else:
        fr = mono[: n * hop].reshape(n, hop)
        frame_db = 20 * np.log10(np.sqrt(np.mean(fr ** 2, axis=1)) + eps)
    voiced = np.flatnonzero(frame_db > th.silence_dbfs)
    lead = (voiced[0] if voiced.size else len(frame_db)) * 20.0
    trail = ((len(frame_db) - 1 - voiced[-1]) if voiced.size else len(frame_db)) * 20.0
    spec = np.abs(np.fft.rfft(mono)) ** 2
    cum = np.cumsum(spec)
    bw99 = float(np.searchsorted(cum, 0.99 * cum[-1]) * sr / len(mono)) if cum[-1] > 0 else 0.0
    return {
        "rms_dbfs": 20 * np.log10(rms), "peak_dbfs": 20 * np.log10(peak),
        "clip_frac": float(np.mean(np.abs(mono) >= th.clip_level)),
        "dc_offset": float(np.mean(mono)),
        "lead_silence_ms": float(lead), "trail_silence_ms": float(trail),
        "silence_frac": float(np.mean(frame_db <= th.silence_dbfs)),
        "bw99_hz": bw99,
    }


def fingerprint(x: np.ndarray, sr: int) -> np.ndarray | None:
    """Compact time-aligned log-mel fingerprint for near-duplicate search.

    This is an audit diagnostic only. It is NOT the project's preprocessing
    profile and must not be reused as model input.
    """
    import librosa
    mono = x.mean(axis=1) if x.ndim == 2 else x
    if sr != FP_SR:
        mono = librosa.resample(mono.astype(np.float32), orig_sr=sr, target_sr=FP_SR)
    if len(mono) < int(0.1 * FP_SR):
        return None
    mel = librosa.feature.melspectrogram(y=mono, sr=FP_SR, n_fft=512, hop_length=160,
                                         n_mels=FP_MELS)
    logmel = np.log(mel + 1e-6)
    pooled = np.stack([c.mean(axis=1) for c in np.array_split(logmel, FP_TBINS, axis=1)], 1)
    v = pooled.flatten()
    v = (v - v.mean()) / (v.std() + 1e-8)
    return (v / (np.linalg.norm(v) + 1e-8)).astype(np.float32)


def audit_file(path: Path, root: Path, th: Thresholds, want_fp: bool):
    row = {"rel_path": path.relative_to(root).as_posix(), "label": infer_label(path, root),
           "split_dir": infer_split_dir(path, root), "ext": path.suffix.lower(),
           "stem": path.stem, "size_bytes": path.stat().st_size,
           "sha256_bytes": sha256_file(path)}
    row["filename_signature"] = filename_signature(path.stem)
    row.update(header_info(path))
    fp = None
    try:
        x, sr = sf.read(str(path), dtype="float32", always_2d=False)
        if x.size == 0:
            raise ValueError("decoded zero samples")
        row["decode_ok"], row["decode_error"] = True, ""
        row["decoded_samples"] = int(x.shape[0])
        q = np.clip(np.round(x * 32767), -32768, 32767).astype(np.int16)
        import hashlib
        row["pcm_sha256"] = hashlib.sha256(f"{sr}|".encode() + q.tobytes()).hexdigest()
        row.update(signal_stats(x, sr, th))
        if want_fp:
            fp = fingerprint(x, sr)
    except Exception as exc:  # noqa: BLE001 - recorded as corrupt, never silently skipped
        row["decode_ok"], row["decode_error"] = False, f"{type(exc).__name__}: {exc}"
        row["pcm_sha256"] = None
    return row, fp


# --------------------------------------------------------------------------- dataset-level
class _UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def neardup_col(sim: float) -> str:
    """Inventory column name for the group id at one threshold, e.g. 0.95 -> neardup_group_095."""
    return f"neardup_group_{round(sim * 100):03d}"


def near_duplicates(F: np.ndarray, valid: np.ndarray, th: Thresholds,
                    chunk: int = 1024, max_pairs: int = 2_000_000):
    """Chunked all-pairs cosine similarity.

    Returns (pairs_df, max_sim, groups_by_sim, truncated). groups_by_sim maps every
    threshold in th.neardup_sims to connected-component group ids (transitive, so
    A~B and B~C put A and C in one group even if A is not ~ C). Grouping uses every
    pair, even when the stored pairs table is truncated at max_pairs.
    """
    n = F.shape[0]
    idx = np.flatnonzero(valid)
    G = F[idx]
    max_sim = np.full(n, np.nan, dtype=np.float32)
    lo = min(th.neardup_sims)
    pairs, truncated = [], False
    ufs = {s: _UnionFind(n) for s in th.neardup_sims}
    for s in range(0, len(idx), chunk):
        S = G[s:s + chunk] @ G.T
        rows = np.arange(s, min(s + chunk, len(idx)))
        S[np.arange(len(rows)), rows] = -np.inf  # ignore self
        max_sim[idx[rows]] = S.max(axis=1)
        S[np.tril(np.ones_like(S, dtype=bool), k=s)] = -np.inf  # keep j > i only
        r, c = np.nonzero(S >= lo)
        for a, b, v in zip(idx[rows[r]], idx[c], S[r, c]):
            for sim, uf in ufs.items():
                if v >= sim:
                    uf.union(int(a), int(b))
            if len(pairs) < max_pairs:
                pairs.append((int(a), int(b), float(v)))
            else:
                truncated = True
    groups = {sim: np.array([uf.find(i) for i in range(n)]) for sim, uf in ufs.items()}
    return pd.DataFrame(pairs, columns=["i", "j", "sim"]), max_sim, groups, truncated


def neardup_group_stats(groups: np.ndarray, decodable: np.ndarray, th: Thresholds) -> dict:
    """Group-size statistics for the D14 chaining check, over decodable clips only.

    Corrupt files are excluded (as §0 excludes them). d14_pass is True when the largest
    group holds at most th.d14_max_group_pct % of decodable clips.
    """
    sizes = pd.Series(groups[decodable]).value_counts()
    pct = round(100.0 * int(sizes.max()) / int(decodable.sum()), 4)
    return {"n_groups": int(len(sizes)), "multi_file_groups": int((sizes > 1).sum()),
            "largest_group": int(sizes.max()), "largest_group_pct": pct,
            "d14_pass": bool(pct <= th.d14_max_group_pct)}


def d14_sanity_failure(sanity: dict, sim: float) -> str | None:
    """D14 sanity check at one threshold: None if it passes, else the reason it fails.

    Passes when there are no exact-PCM-duplicate pairs, or when their minimum
    fingerprint similarity is >= sim. Pairs that exist but cannot be checked (no
    fingerprint for either file) fail: an unverifiable check is not a pass.
    """
    if sanity["pairs"] == 0:
        return None
    if sanity["min_sim"] is None:
        return (f"{sanity['pairs']} exact-PCM-duplicate pairs exist but none has "
                "fingerprints to check")
    if sanity["min_sim"] < sim:
        return (f"min similarity among exact-PCM-duplicate pairs is "
                f"{sanity['min_sim']:.4f} < {sim}")
    return None


def d14_select_threshold(by_threshold: dict, th: Thresholds):
    """D14: first candidate threshold that is eligible (chaining check AND PCM sanity
    check both pass), else 'pcm_hash_only'."""
    for s in th.d14_candidate_sims:
        if by_threshold[str(s)]["d14_eligible"]:
            return s
    return "pcm_hash_only"


def pcm_dup_min_sim(F: np.ndarray, valid: np.ndarray, pcm_hash: pd.Series) -> dict:
    """D14 sanity check: minimum fingerprint similarity among exact-PCM-duplicate pairs.

    Every such pair must reach the chosen grouping threshold. Pairs where a file has
    no fingerprint cannot be checked and are counted separately, not skipped silently.
    """
    h = pcm_hash.reset_index(drop=True)
    min_sim, n_pairs, unchecked = None, 0, 0
    for _, members in h[h.notna()].groupby(h[h.notna()]).groups.items():
        m = np.asarray(members)
        if len(m) < 2:
            continue
        k = len(m)
        n_pairs += k * (k - 1) // 2
        ok = m[valid[m]]
        unchecked += k * (k - 1) // 2 - len(ok) * (len(ok) - 1) // 2
        if len(ok) < 2:
            continue
        S = F[ok] @ F[ok].T
        v = float(S[np.triu_indices(len(ok), k=1)].min())
        min_sim = v if min_sim is None else min(min_sim, v)
    return {"pairs": n_pairs, "pairs_without_fingerprint": unchecked, "min_sim": min_sim}


def dup_group_report(inv: pd.DataFrame, key: str) -> dict:
    d = inv[inv[key].notna()]
    g = d.groupby(key)
    multi = g.filter(lambda x: len(x) > 1)
    if multi.empty:
        return {"groups": 0, "files_in_groups": 0, "cross_label_groups": 0,
                "cross_split_groups": 0}
    gg = multi.groupby(key)
    return {"groups": int(gg.ngroups), "files_in_groups": int(len(multi)),
            "cross_label_groups": int((gg["label"].nunique() > 1).sum()),
            "cross_split_groups": int((gg["split_dir"].nunique() > 1).sum())}


def shortcut_tables(inv: pd.DataFrame, th: Thresholds):
    ok = inv[inv["decode_ok"]]
    rows = []
    for f in NUMERIC_SHORTCUT_FEATURES:
        if f in ok and ok[f].notna().any():
            s = auc_separability(ok[f], ok["label"])
            flag = ("HIGH" if s >= th.shortcut_auc_high else
                    "MODERATE" if s >= th.shortcut_auc_moderate else "")
            med = ok.groupby("label")[f].median()
            rows.append({"feature": f, "separability_auc": s,
                         "median_REAL": med.get("REAL", np.nan),
                         "median_FAKE": med.get("FAKE", np.nan), "flag": flag})
    num = pd.DataFrame(rows).sort_values("separability_auc", ascending=False)

    cat_rows, n = [], len(inv)
    for f in CATEGORICAL_SHORTCUT_FEATURES:
        ct = pd.crosstab(inv[f].astype(str), inv["label"])
        for val, r in ct.iterrows():
            tot = int(r.sum())
            if tot / n < th.min_support_frac:
                continue
            purity = r.max() / tot
            if purity >= th.category_purity:
                cat_rows.append({"feature": f, "value": val, "n": tot,
                                 "majority_label": r.idxmax(), "purity": purity})
    return num, pd.DataFrame(cat_rows, columns=["feature", "value", "n",
                                                "majority_label", "purity"])


def token_table(inv: pd.DataFrame, th: Thresholds, top: int = 30) -> pd.DataFrame:
    rec = [(t, lab) for stem, lab in zip(inv["stem"], inv["label"]) for t in set(filename_tokens(stem))]
    if not rec:
        return pd.DataFrame(columns=["token", "n", "frac_FAKE", "label_pure"])
    t = pd.DataFrame(rec, columns=["token", "label"])
    ct = pd.crosstab(t["token"], t["label"])
    for lab in ("REAL", "FAKE"):
        if lab not in ct:
            ct[lab] = 0
    ct["n"] = ct["REAL"] + ct["FAKE"]
    ct["frac_FAKE"] = ct["FAKE"] / ct["n"]
    ct["label_pure"] = (ct[["REAL", "FAKE"]].max(axis=1) / ct["n"]) >= th.category_purity
    return ct.sort_values("n", ascending=False).head(top).reset_index()[
        ["token", "n", "frac_FAKE", "label_pure"]]


def describe_by_label(inv: pd.DataFrame, col: str) -> pd.DataFrame:
    q = inv.groupby("label")[col].describe(percentiles=[0.01, 0.5, 0.99])
    return q.reset_index()[["label", "count", "min", "1%", "50%", "99%", "max", "mean"]]


# --------------------------------------------------------------------------- main
def run(root: Path, out: Path, revision: str, limit: int | None = None,
        do_fingerprint: bool = True, th: Thresholds = Thresholds()) -> dict:
    t0 = time.time()
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    files = find_audio_files(root)[: limit or None]
    rows, fps = [], []
    for k, p in enumerate(files, 1):
        r, fp = audit_file(p, root, th, do_fingerprint)
        rows.append(r)
        fps.append(fp)
        if k % 1000 == 0:
            print(f"  audited {k}/{len(files)} files ({time.time() - t0:.0f}s)")
    inv = pd.DataFrame(rows)

    report = {"dataset": "FoR-2sec", "root": str(root), "revision": revision,
              "limit": limit, "thresholds": th.as_dict(), "environment": environment_info()}
    report["counts"] = {
        "files": len(inv),
        "by_label": inv["label"].value_counts().to_dict(),
        "by_split_dir_label": {f"{a}/{b}": int(v) for (a, b), v in
                               inv.groupby(["split_dir", "label"]).size().items()},
        "header_failures": int((~inv["header_ok"]).sum()),
        "decode_failures": int((~inv["decode_ok"]).sum()),
        "ext": inv["ext"].value_counts().to_dict(),
    }
    ok = inv[inv["decode_ok"]]
    report["duration_not_2s_pm50ms"] = int((ok["duration_s"].sub(2.0).abs() > 0.05).sum())
    report["exact_duplicates_bytes"] = dup_group_report(inv, "sha256_bytes")
    report["exact_duplicates_pcm"] = dup_group_report(inv, "pcm_sha256")

    nd_pairs = pd.DataFrame(columns=["i", "j", "sim"])
    if do_fingerprint:
        valid = np.array([f is not None for f in fps])
        dim = next((len(f) for f in fps if f is not None), 1)
        F = np.stack([f if f is not None else np.zeros(dim, np.float32) for f in fps])
        nd_pairs, max_sim, groups, truncated = near_duplicates(F, valid, th)
        inv["fp_max_sim"] = max_sim
        for s in th.neardup_sims:
            inv[neardup_col(s)] = groups[s]
        np.save(out / "for2sec_fingerprints.npy", F)
        lab, spl = inv["label"].to_numpy(), inv["split_dir"].to_numpy()
        decodable = inv["decode_ok"].to_numpy(dtype=bool)
        sanity = pcm_dup_min_sim(F, valid, inv["pcm_sha256"])
        nd, sanity_failures = {}, {}
        for s in th.neardup_sims:
            p = nd_pairs[nd_pairs["sim"] >= s]
            nd[str(s)] = {"pairs": int(len(p)),
                          "cross_label_pairs": int((lab[p.i] != lab[p.j]).sum()),
                          "cross_split_pairs": int((spl[p.i] != spl[p.j]).sum()),
                          **neardup_group_stats(groups[s], decodable, th)}
            reason = d14_sanity_failure(sanity, s)
            nd[str(s)]["d14_sanity_pass"] = reason is None
            nd[str(s)]["d14_eligible"] = nd[str(s)]["d14_pass"] and reason is None
            if reason:
                sanity_failures[str(s)] = reason
        report["near_duplicates"] = {
            "fingerprint_invalid": int((~valid).sum()), "pairs_truncated": truncated,
            "by_threshold": nd,
            "d14_selected_threshold": d14_select_threshold(nd, th),
            "pcm_dup_sanity": sanity,
            "d14_sanity_failures": sanity_failures,
            "max_sim_quantiles_by_label": {
                lb: inv.loc[inv.label == lb, "fp_max_sim"].quantile(
                    [0.5, 0.9, 0.99, 1.0]).round(4).to_dict()
                for lb in ("REAL", "FAKE")},
        }
        nd_pairs = nd_pairs.assign(a=inv.rel_path.to_numpy()[nd_pairs.i],
                                   b=inv.rel_path.to_numpy()[nd_pairs.j])

    num_sc, cat_sc = shortcut_tables(inv, th)
    tokens = token_table(inv, th)
    report["shortcut_flags"] = {
        "numeric_high": num_sc.loc[num_sc.flag == "HIGH", "feature"].tolist(),
        "numeric_moderate": num_sc.loc[num_sc.flag == "MODERATE", "feature"].tolist(),
        "categorical_pure_values": len(cat_sc),
        "label_pure_filename_tokens_in_top30": int(tokens["label_pure"].sum()),
    }
    report["runtime_s"] = round(time.time() - t0, 1)

    inv.to_csv(out / "for2sec_inventory.csv", index=False)
    nd_pairs.to_csv(out / "for2sec_neardup_pairs.csv", index=False)
    num_sc.to_csv(out / "for2sec_shortcut_numeric.csv", index=False)
    cat_sc.to_csv(out / "for2sec_shortcut_categorical.csv", index=False)
    write_json(report, out / "for2sec_summary.json")
    _write_markdown(out / "for2sec_summary.md", report, inv, num_sc, cat_sc, tokens)
    print(f"FoR-2sec audit done: {len(inv)} files in {report['runtime_s']}s -> {out}")
    return report


def _write_markdown(path, report, inv, num_sc, cat_sc, tokens):
    ok = inv[inv["decode_ok"]]
    c = report["counts"]
    md = ["# FoR-2sec audit — machine summary",
          f"Revision: `{report['revision']}` · files: {c['files']} · "
          f"generated {report['environment']['timestamp_utc']}",
          "\n## 1. Inventory",
          df_to_md(pd.DataFrame([{"split_dir/label": k, "files": v}
                                 for k, v in c["by_split_dir_label"].items()])),
          f"\nHeader failures: {c['header_failures']} · decode failures: "
          f"{c['decode_failures']} · extensions: {c['ext']}",
          "\n## 2. Format by label",
          df_to_md(pd.crosstab([inv["samplerate"], inv["channels"], inv["subtype"]],
                               inv["label"]).reset_index()),
          "\n## 3. Duration by label (s)", df_to_md(describe_by_label(ok, "duration_s")),
          f"\nClips outside 2.0 s ± 50 ms: {report['duration_not_2s_pm50ms']}",
          "\n## 4. Shortcut screen — numeric (univariate separability AUC)",
          df_to_md(num_sc),
          "\n## 5. Shortcut screen — categorical values ≥ purity threshold",
          df_to_md(cat_sc) if len(cat_sc) else "_none_",
          "\n## 6. Filename tokens (top 30 by frequency)", df_to_md(tokens),
          "\n## 7. Exact duplicates",
          df_to_md(pd.DataFrame([{"key": "bytes", **report["exact_duplicates_bytes"]},
                                 {"key": "decoded PCM", **report["exact_duplicates_pcm"]}]))]
    if "near_duplicates" in report:
        nd = report["near_duplicates"]
        if nd["d14_sanity_failures"]:
            sanity_md = ["\n**D14 sanity check FAILED** (threshold not eligible):"] + \
                [f"- {k}: {v}" for k, v in nd["d14_sanity_failures"].items()]
        else:
            sanity_md = ["\nD14 sanity check passed at all thresholds."]
        md += ["\n## 8. Near-duplicates (log-mel fingerprint cosine)",
               df_to_md(pd.DataFrame([{"threshold": k, **v}
                                      for k, v in nd["by_threshold"].items()])),
               "\n_Groups are transitive (connected components). largest_group_pct is "
               "% of decodable clips; d14_pass = largest_group_pct ≤ "
               f"{report['thresholds']['d14_max_group_pct']}% (D14 chaining check)._",
               f"\n**D14 selected threshold: {nd['d14_selected_threshold']}** "
               f"(candidates in order: {report['thresholds']['d14_candidate_sims']}, "
               "else pcm_hash_only)",
               "\n_d14_eligible = d14_pass AND d14_sanity_pass (every exact-PCM-duplicate "
               "pair has similarity ≥ the threshold)._",
               f"\nD14 sanity — exact-PCM-duplicate pairs: {nd['pcm_dup_sanity']}",
               *sanity_md,
               f"\nMax-sim quantiles: "
               f"{nd['max_sim_quantiles_by_label']}",
               "\n_Limitation: aligned fingerprints do not detect time-shifted "
               "overlapping excerpts of the same source recording._"]
    md += ["\n## 9. Requires human review (not automatable)",
           "- Speaker / source-recording metadata: inspect §6 tokens and the dataset card.",
           "- Generator (TTS engine) metadata: inspect §6 tokens and the dataset card.",
           "- License and redistribution terms."]
    Path(path).write_text("\n".join(md), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--revision", required=True,
                    help="Exact dataset revision (HF commit sha). Required for traceability.")
    ap.add_argument("--limit", type=int, default=None, help="Dry run on first N files only")
    ap.add_argument("--no-fingerprint", action="store_true")
    a = ap.parse_args()
    if not a.revision.strip():
        raise AuditError("--revision must not be empty")
    run(a.root, a.out, a.revision, a.limit, not a.no_fingerprint)


if __name__ == "__main__":
    main()
