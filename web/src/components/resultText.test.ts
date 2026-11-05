import { describe, expect, it } from 'vitest'
import { isMockResult, resultAnnouncement, VERDICT_SHORT } from './resultText'

describe('resultAnnouncement', () => {
  it.each([
    ['synthetic', 'HIGH', 'Result ready. Verdict: Likely synthetic voice. High risk. Required action: do not act on the request yet.'],
    ['uncertain', 'MEDIUM', 'Result ready. Verdict: Cannot decide. Medium risk. Manual verification required.'],
    ['authentic', 'LOW', 'Result ready. Verdict: Likely authentic voice. Low risk. Required action: continue your normal verification procedure.'],
  ] as const)('%s, %s', (prediction, risk, text) => {
    expect(resultAnnouncement({ prediction, risk_level: risk })).toBe(text)
  })
})

describe('the short verdict', () => {
  it('has the words of prototype line 594', () => {
    expect(VERDICT_SHORT).toEqual({ authentic: 'Likely authentic', uncertain: 'Cannot decide', synthetic: 'Likely synthetic' })
  })
})

describe('isMockResult', () => {
  it('is true only for model_version "mock"', () => {
    expect(isMockResult({ model_version: 'mock' })).toBe(true)
    expect(isMockResult({ model_version: 'cnn-v1' })).toBe(false)
  })
})
