import { describe, expect, it } from 'vitest'
import { resultAnnouncement } from './resultText'

describe('resultAnnouncement', () => {
  it.each([
    ['synthetic', 'HIGH', 'Result ready. Verdict: Likely synthetic voice. High risk. Required action: do not act on the request yet.'],
    ['uncertain', 'MEDIUM', 'Result ready. Verdict: Cannot decide. Medium risk. Manual verification required.'],
    ['authentic', 'LOW', 'Result ready. Verdict: Likely authentic voice. Low risk. Required action: continue your normal verification procedure.'],
  ] as const)('%s, %s', (prediction, risk, text) => {
    expect(resultAnnouncement({ prediction, risk_level: risk })).toBe(text)
  })
})
