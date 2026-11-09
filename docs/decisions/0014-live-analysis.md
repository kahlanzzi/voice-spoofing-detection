# 0014. Live analysis: scope, capture paths and decision rule

- **Status:** Proposed (agreed in principle by the team on 2026-11-09, user-reported; formal
  approval by all four in this PR)
- **Date:** 2026-11-09
- **Register:** D51 (changes SRS v3.0 WH-01 for SRS v4.0), D52
- **Deciders:** Iheb Zemzemi (Part E owner, D51), Khalil Dridi (Stage 13 conventions reused by D52),
  Donia Mabrouk (author), Malak Ben Salem
- **AI assistance:** Claude (claude.ai, Claude Opus 5.5) drafted this ADR and checked the platform
  facts below against public sources (2026-11-09); the team reviews it.

## Context

- SRS v3.0 lists real-time or streaming detection as *Won't have* (WH-01), and the SDD rejected an
  event-driven style because streaming was out of scope. The professor asked for a live check that an
  analyst or a user can run during a call, ideally without opening the web app.
- **Platform facts that bound the design:**
  - Android removed third-party access to call audio step by step: recording through the microphone
    during calls was blocked in Android 10, and a Play policy from 11 May 2022 forbids using the
    Accessibility API to capture call audio. Only preloaded, first-party dialers can read the
    `VOICE_CALL` source. iOS has never given third-party apps call audio.
  - WhatsApp calls are end-to-end encrypted. No official interface gives a third party the audio of a
    personal call; the only consented access point is the user's own device output.
  - WhatsApp Web supports voice calls in the browser (rolled out in 2026), as do Google Meet and other
    web call tools. A Chrome Manifest V3 extension can capture a tab's audio with `chrome.tabCapture`
    after a user gesture, in an offscreen document. Capturing mutes the tab unless the stream is routed
    back to the speakers, which is a known, documented pattern.
- Our evaluation unit is already the 2 s window (D5). Stage 13 defines non-overlapping 2 s windows, a
  speech-window filter (D17) and a file-level rule: the frozen threshold and band applied to the mean
  calibrated p of a file (D34). DEEP-VOICE files are 79–600 s long, so they look like calls.

## Decision

- **D51 (scope):** Must: live microphone mode in the web app and a streaming endpoint
  (`/api/live`, WebSocket). Should: a Chrome extension that captures the active tab (WhatsApp Web,
  Meet) and uses the same endpoint. Won't this semester: a mobile app, and capture of native phone or
  in-app mobile call audio (a platform restriction, documented as such). Audio is processed per window
  in memory and never stored or logged; capture starts only after an explicit user action and shows a
  visible indicator.
- **D52 (rule):** server-side F3 on non-overlapping 2 s windows, the D17 speech filter, and a running
  verdict equal to D34's file-level rule applied to the stream so far (mean calibrated p of kept
  windows, frozen threshold and band), after a warm-up of 3 kept windows. No new parameter is fitted.

Full texts: D51 and D52 in [`README.md`](README.md).

## Consequences

- **Easier:** one decision rule for uploads, files and live calls; Stage 13's file-level view (D34)
  measures the end-of-call accuracy of the live rule on real long recordings at no extra test cost.
- **Committed:** SRS v4.0 moves live analysis from *Won't* to the priorities above and adds its
  functional and privacy requirements; SDD v2.0 adds the streaming components; NFR-03 and NFR-04
  (no stored audio, no audio or identifiers in logs) extend to live sessions.
- **Committed:** the extension's authentication needs a design decision and a security review in
  SDD v2.0 before any implementation. D47's cookie is `SameSite=Strict` with `Path=/api`, and every
  state-changing request needs an allowed `Origin`, so an extension origin is not accepted today.
- **Harder:** the API gains long-lived connections (capacity rules in D53); the UI needs a live screen
  and a per-window timeline (D55).
- **Limits stated honestly:** the live verdict is decision support, not proof; the tab capture mixes
  everything the tab plays (mostly the remote speaker); the legal need for the other party's consent
  varies by country, and the UI says so.

## Options considered

1. **Native phone-call capture (React Native or Flutter).** Rejected: blocked by the operating systems
   for third-party apps; a framework choice cannot change that.
2. **Paid telephony or WhatsApp business interfaces.** Rejected: not free, and they do not cover a
   person's own WhatsApp calls.
3. **Overlapping windows with smoothing or per-window alerts.** Rejected: new parameters that Stage 13
   does not validate, and per-window labels are noisy (132 of 1,329 val-B clips, 9.9%, fall in the uncertain band).
4. **Mobile share-to-analyse for voice notes.** Deferred as future work: valuable, but a third client
   in the remaining weeks would cost the core work.

## Links

- [Decision register](README.md): D5, D17, D34, D39, D47, D51–D53, D55.
- [ADR 0007](0007-stage13-reporting-and-access.md), [ADR 0011](0011-analyst-sign-in-sessions.md),
  [ADR 0015](0015-inference-infrastructure-and-environments.md).
- [API design](../api_design.md), [architecture](../architecture.md).
- Sources checked 2026-11-09: Google's 2022 Play policy on Accessibility-based call recording (press
  coverage, e.g. 9to5Google, The Register); WhatsApp Web calling roll-out (2026 press coverage);
  Chrome developer documentation for `chrome.tabCapture` and "Audio recording and screen capture".
