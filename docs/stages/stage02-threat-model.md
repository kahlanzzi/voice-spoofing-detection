# Stage 2 — Threat model

| | |
|---|---|
| **Stage** | 2 — Threat model |
| **Owner** | Donia Mabrouk (D26, Part A) |
| **Reviewer(s)** | Khalil Dridi (his Stages 13, 15 and 16 test and report against the threat limits; D26) |
| **Status** | Accepted (PR #40) |
| **Date** | 2026-10-18 |
| **Related PRs** | #3 (`stage-2-threat-model`, approved by Malak, merged into `stage-1-audit-run` by mistake, 2026-10-05) · #4 (`stage-2-into-main`, re-delivery to `main`, approved by Iheb, 2026-10-06) · #5 (`docs/stage2-closeout`, sign-offs, Entry 14) — source: Entry 14, Entry 15 |

## 1. In one paragraph

Before we measure anything, we must say what we are protecting, against whom, and what our results
will be allowed to claim. Stage 2 writes this down in [`docs/threat_model.md`](../threat_model.md):
a bank call center sends only high-risk phone requests (large transfer, phone-number change,
credential reset) to the detector, and an analyst checks the result before the action runs. It
names the attackers (a fraudster using text-to-speech, an impersonator using voice conversion), the
attacks we cover and the ones we do not, and maps each threat T1–T7 to the experiment that will
test it and to the most that experiment can prove. It also lists threats to the service itself,
such as attackers using the detector as a free testing tool. Nothing in this stage touches audio
or trains a model; it is a document that later stages are checked against.

## 2. Inputs → outputs

| | What | From / to | Path |
|---|---|---|---|
| In | Audit facts (bandwidth gap, MP3 imbalance, no speaker metadata, DEEP-VOICE metadata) | Stage 1 (Donia) | `docs/dataset_audit.md`, `docs/audit_evidence/` |
| In | Scope, requirements, attacker assumptions | SRS v3.0 (§1.2, WH-01–04, NFRs), Proposal §2.3, SDD v1.0 | `docs/reference/` |
| In | Approved decisions D2, D8, D9, D11, D12, D14, D16–D20; H5, H6 | Register | `docs/decisions/README.md` |
| Out | Scenario, scope, threats T1–T7 with their claim limits | Stage 3 protocol (Donia), Stages 11, 13–16 (Donia, Khalil, Malak) | `docs/threat_model.md` §3, §7, §8 |
| Out | Error policy (false positive / false negative / uncertain) | Stage 10 risk policy (Donia), Stage 18 dashboard (Iheb) | `docs/threat_model.md` §9 |
| Out | System threats and the oracle-abuse residual risk | Stage 17 API (Iheb) | `docs/threat_model.md` §10, §10.1 |
| Out | D21 (dataset pinning) added to the register | Every stage | `docs/decisions/README.md` (Entry 11) |

No D26 hand-over interface belongs to this stage; those are fixed in Stage 3 (D26).

## 3. How it works, step by step

1. **Stacked branch.** Stage 1 was waiting for sign-off, so `stage-2-threat-model` was branched from
   `stage-1-audit-run`, not from `main` (Entry 9).
2. **Draft v0.1.** Drafted with Claude (chat) from the team's documents and decisions, committed as
   `8b7b679` for team review (Entry 10, corrected by Entry 12).
3. **Citation check** (Claude Code): every FR / NFR / PC / WH / UN ID checked against the SRS text,
   every D / H ID against the register, the audit facts against `docs/dataset_audit.md`, and the
   Mermaid diagram parsed. Two mismatches were reported, not fixed silently (Entry 10).
4. **v0.2.** The two fixes applied (§3.2 REAL filenames; §10 cites D21 instead of D20), FR-11 marked
   as Could priority with residual risk 7, and **D21** (dataset pins) added to the register
   (Entry 11).
5. **Review → v1.0 "reviewed".** Malak approved via PR #3 (2026-10-05), Iheb via PR #4 (2026-10-06);
   Donia is the author (`docs/threat_model.md` §14; Entry 14).
6. **Merge fix.** PR #3 had the wrong base and landed on a branch nothing merges any more; the work
   was re-delivered through PR #4, and the stacked-PR rule was added (Entry 14).
7. **Open questions handed on.** §13 Q1 (scenarios) and Q3 (English only) were answered in Protocol
   v1.0 §0; Q2 (access control) stays open until Stage 17 at the latest (Open items, Stage 2 gate).

## 4. Key decisions and why

- **Primary scenario = triage of high-risk requests** (threat model §3.1, [decision]; confirmed in
  Protocol §0): keeps false alarms reviewable (§3.4).
- **D8** — the research question is FoR (TTS) → DEEP-VOICE (RVC); a cause such as "generation method"
  is a hypothesis only: the two datasets differ in many ways at once.
- **D2, D12** — thresholds and the uncertain band are fitted on FoR validation and frozen before the
  external run (§9): the external test cannot shape the decision rule.
- **D3** — robustness runs on the FoR test split only (T3, T4 in §8): DEEP-VOICE stays a single, clean test.
- **D17, D19** — speech-window filter and per-speaker table for the external run (T2, T7).
- **H5, H6** — the two shortcut hypotheses from the audit become threats T5 and T4.
- **D21** — produced in this stage (Entry 11): pinned datasets back the "Tampering (data)" control in §10.

## 5. Evidence

- All cited requirement and decision IDs exist and match their meaning; Mermaid block parses
  (`mermaid.parse` 11.17.2, with a broken-diagram negative control) [verified] — source: Entry 10.
- Citation check re-run on every changed v0.2 line; the D21 hashes equal `configs/audit.yaml`; D20 no
  longer cited [verified] — source: Entry 11.
- T4 fact: FoR REAL clips carry energy to ~5.1 kHz vs ~3.3 kHz for FAKE (99%-energy medians)
  [verified] — source: threat model §8 T4; exact medians 5,074.5 Hz / 3,264 Hz in
  `docs/dataset_audit.md` §3.2.
- T5 fact: 85% of FAKE and 0% of REAL clips have MP3 history; about 200 FAKE_other test clips
  [verified for the 85% / 0%; "~200" as stated in threat model §8 T5].
- Sign-off: Malak (PR #3, 2026-10-05), Iheb (PR #4, 2026-10-06), Donia author [doc] — source: threat
  model §14; Entry 14. Khalil joined later (Entry 18) and has not reviewed the threat model.
- The §3.4 base-rate table (10,000 vs ~200 calls a day, 1% false alarms) is **illustrative, not
  measured**, as the table itself says [doc] — source: threat model §3.4.

## 6. Limitations

- The threat model is a document, not a test result: each T row is only a plan until its stage
  reports (§8). Until then, every "limit of the claim" is an upper bound on what we may say.
- The base-rate numbers in §3.4 are invented to show the effect; no real call-center volume or fraud
  rate was measured or sourced.
- "The attacker cannot tamper with the service" is an assumption, only partly supported by the
  controls in §10 (§11 item 1).
- Oracle abuse (§10.1) and analyst impersonation (FR-11 is Could priority) are unmitigated until access
  control is decided (§11 items 5 and 7; Open items: before Stage 17).
- Not covered by any experiment: replay, adversarial evasion, partial fakes, live calls, other
  languages, generators absent from both datasets, real phone-network recordings (§8).
- "English speech" is stated as [doc] (dataset papers), not [verified]: DEEP-VOICE audio may not be
  inspected (Protocol §0; D20).

## 7. Questions the professor may ask

1. **Q:** Why only high-risk requests in a call center, and not screening every call?
   **A:** Fraud is rare, so screening every call drowns analysts in false alarms. With illustrative
   (not measured) numbers, 10,000 calls a day at 1% false alarms give ~100 per real attack; triage of
   ~200 high-risk requests gives ~2 (threat model §3.4; scenario §3.1, confirmed in Protocol §0).
2. **Q:** Which attacks are in scope and which are not, and why?
   **A:** In: TTS (FoR is TTS) and voice conversion, RVC only (DEEP-VOICE is RVC). Out: replay and
   partial fakes (in neither dataset), adversarial evasion (SRS WH-02), live calls (WH-01); no claim
   for other languages or newer generators (threat model §7).
3. **Q:** What does "decision support" mean, and what does an "authentic" result not prove?
   **A:** The system flags a recording for the analyst, who decides; "synthetic" triggers out-of-band
   checks, never an automatic block. It is not a biometric authenticator, and "authentic" does not
   prove the request is legitimate (threat model §1, §3.3; SRS §1.2, NFR-05).
4. **Q:** Which threat is the most critical operationally, and how will you test it?
   **A:** T4: genuine callers on narrowband phone lines flagged as synthetic, because FoR REAL clips
   have wider bandwidth than FAKE (§8). H6 in Stage 14: band-limit REAL FoR test clips to 3.4 kHz;
   confirmed if false "synthetic" calls rise ≥ 5 points over clean and the 95% CI excludes 0 (Protocol §9).
5. **Q:** What is "oracle abuse", and how do you mitigate it?
   **A:** An attacker submits a fake, reads the result and adjusts it until it passes, using our
   detector as a free test. FR-06 requires the confidence output, so we limit who can ask: no public
   API, authenticated clients, per-client rate limits, volume in the log (FR-20) (§10.1). Residual risk until Stage 17.
6. **Q:** What is the maximum claim your experiments can support for each threat?
   **A:** The last column of §8: T1 FoR engines only, an upper bound (no speaker metadata); T2 RVC only,
   8 source recordings, confounded (D8); T3 simulated degradation; T4/T5 only via H6/H5, T5 with wide
   CIs; T6 calibration fitted on FoR only; T7 our own pipeline only.
7. **Q:** What happens after a false positive versus a false negative?
   **A:** False negative: the fraud proceeds; the system is one control and analysts still verify
   high-risk requests. False positive: a genuine customer is delayed or suspected; it triggers
   verification, not a block, and T4 tests the main cause. Low confidence → "uncertain" → manual check (§9; FR-05).
8. **Q:** Which real incidents motivate the project, and do you reproduce them?
   **A:** A 2019 case where a UK energy company's CEO transferred €220,000 after an AI-imitated voice
   call, and a 2023 Vice journalist who passed Lloyds Bank's Voice ID with a cloned voice [doc]. They
   are motivation only; our experiments do not reproduce them (threat model §2).

## 8. How to verify or re-run it

Stage 2 is a document; "re-running" it means re-checking its references and the guard that keeps
the external set untouched. No data or tokens needed:

```bash
grep -o "D[0-9]\+\|H[56]" docs/threat_model.md | sort -u                      # every cited decision / hypothesis ID
grep -o "^| D[0-9]\+\|^| H[56]" docs/decisions/README.md | tr -d '| ' | sort -u  # IDs that exist in the register
grep -o "\(FR\|NFR\|PC\|WH\|UN\)-[0-9]\+" docs/threat_model.md | sort -u          # compare by hand with the SRS PDF
python -m pytest -q tests/governance                                          # DEEP-VOICE guard
```

Expected: every ID from the first command appears in the second; the requirement IDs exist in SRS
v3.0 (checked in Entries 10–11). The governance test printed `3 passed` on this branch (Python
3.14.7 local venv). The Mermaid diagram in §4 renders on GitHub.
