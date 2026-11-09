import { cleanup, render, screen, within } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import type { AnalysisResponse, RiskLevel, Verdict } from '../api/types'
import ResultCard from './ResultCard'

afterEach(cleanup)

const ID = `req_${'b'.repeat(32)}`

const ANALYSIS: AnalysisResponse = {
  prediction: 'uncertain',
  confidence: 0.62,
  risk_level: 'MEDIUM',
  model_version: 'mock',
  profile_version: 'profile-under-test',
  policy_version: 'policy-under-test',
  request_id: ID as AnalysisResponse['request_id'],
  notice: 'The notice this response carried.',
}

const VERDICTS: readonly [Verdict, RiskLevel][] = [
  ['synthetic', 'HIGH'],
  ['uncertain', 'MEDIUM'],
  ['authentic', 'LOW'],
]

function show(change: Partial<AnalysisResponse> = {}) {
  render(<ResultCard analysis={{ ...ANALYSIS, ...change }} fileName="call.wav" />)
  return screen.getByRole('article')
}

const details = () => document.querySelector<HTMLDetailsElement>('details.more')!
const pairs = () => {
  const names = [...details().querySelectorAll('dt')].map((dt) => dt.textContent)
  const values = [...details().querySelectorAll('dd')].map((dd) => dd.textContent)
  return Object.fromEntries(names.map((name, index) => [name, values[index]]))
}

describe('verdict and risk', () => {
  it.each([
    ['synthetic', 'Likely synthetic voice', 'r-syn'],
    ['uncertain', 'Cannot decide', 'r-unc'],
    ['authentic', 'Likely authentic voice', 'r-auth'],
  ] as const)('%s: its title and colour class', (prediction, title, className) => {
    const card = show({ prediction })
    expect(screen.getByRole('heading', { level: 2 }).textContent).toBe(title)
    expect(card.classList.contains(className)).toBe(true)
  })

  it('risk follows risk_level, whatever the verdict', () => {
    const card = show({ prediction: 'synthetic', risk_level: 'LOW' })
    expect(card.querySelector('.risk')?.textContent).toBe('LOW RISK')
  })

  it.each([
    ['LOW', 'LOW RISK'],
    ['MEDIUM', 'MEDIUM RISK'],
    ['HIGH', 'HIGH RISK'],
  ] as const)('%s reads %s, with its icon', (risk, text) => {
    const label = show({ risk_level: risk }).querySelector('.risk')!
    expect(label.textContent).toBe(text)
    expect(label.querySelector('svg')).not.toBeNull()
  })
})

describe('required action and explanation', () => {
  it.each([
    ['synthetic', 'Required action: do not act on the request yet', 3],
    ['uncertain', 'Manual verification required', 3],
    ['authentic', 'Required action: continue your normal verification procedure', 2],
  ] as const)('%s: its heading and %s', (prediction, heading, lines) => {
    show({ prediction })
    const todo = screen.getByRole('region', { name: heading })
    expect(within(todo).getAllByRole('listitem')).toHaveLength(lines)
  })

  it.each([
    ['synthetic', 'This does not prove fraud and does not identify the person responsible.'],
    ['uncertain', 'This is not a weak “authentic” result: treat the request as unverified.'],
    ['authentic', 'This does not verify the caller’s identity and does not guarantee that the request is legitimate.'],
  ] as const)('%s: its explanation', (prediction, text) => {
    show({ prediction })
    expect(document.querySelector('.explain')?.textContent).toContain(text)
  })
})

describe('the model output', () => {
  it('Calibrated model output to 2 decimals, labelled, with its range', () => {
    show({ confidence: 0.7 })
    expect(details().querySelector('.score-def p')?.textContent).toBe('Calibrated model output: 0.70 (range 0.50 to 1.00)')
  })

  it('Technical details is closed, and the main card has no number', () => {
    const card = show()
    expect(details().open).toBe(false)
    const clone = card.cloneNode(true) as HTMLElement
    clone.querySelector('details')?.remove()
    expect(clone.textContent).not.toContain('0.62')
  })

  it('the "it is not" lines use the value', () => {
    show()
    expect(details().textContent).toContain('a percentage of certainty (0.62 does not mean “62% certain”).')
  })

  it('the "Cannot decide" clause is for uncertain only', () => {
    const clause = 'which is why a “Cannot decide” result can show a high value'
    show({ prediction: 'uncertain' })
    expect(details().textContent).toContain(clause)
    cleanup()
    show({ prediction: 'authentic' })
    expect(details().textContent).not.toContain(clause)
  })
})

describe('values from the response', () => {
  it('the versions', () => {
    show({ model_version: 'model-under-test' })
    expect(pairs()).toMatchObject({
      'Model version': 'model-under-test',
      'Preprocessing profile': 'profile-under-test',
      'Risk policy': 'policy-under-test',
    })
  })

  it('the API fields line', () => {
    show({ prediction: 'synthetic', confidence: 1, risk_level: 'LOW' })
    expect(pairs()['API fields']).toBe('prediction = synthetic, confidence = 1.00, risk_level = LOW')
  })

  it('the request ID', () => {
    show()
    expect(screen.getByText(ID).tagName).toBe('CODE')
  })

  it("the notice is the response's text", () => {
    show()
    expect(document.querySelector('.api-notice')?.textContent).toBe('Notice from the service: The notice this response carried.')
  })

  it('model_version mock shows the Mock result tag', () => {
    show({ model_version: 'mock' })
    expect(screen.getByText('Mock result')).toBeTruthy()
  })

  it('a non-mock version shows no tag, whatever the other two say', () => {
    show({ model_version: 'cnn-v1', profile_version: 'preprocessing-v1', policy_version: 'risk-v1' })
    expect(screen.queryByText('Mock result')).toBeNull()
  })
})

describe('the rest of the card', () => {
  it('the storage line names the file', () => {
    show()
    const dd = [...document.querySelectorAll('.result-body dd')].at(-1)!
    expect(dd.textContent).toBe('call.wav is visible on this screen only and is not stored. The audio is deleted when processing finishes.')
  })

  it('no Analyzed or Processing time row: the response has neither', () => {
    show()
    const names = [...document.querySelectorAll('dt')].map((dt) => dt.textContent)
    expect(names).not.toContain('Analyzed')
    expect(names).not.toContain('Processing time')
  })

  it.each(VERDICTS)('%s: none of the words HANDOFF line 196 avoids', (prediction, risk) => {
    const text = show({ prediction, risk_level: risk }).textContent
    for (const word of ['Fake detected', 'Real voice', 'Fraud confirmed', 'Safe', 'Verified', 'AI confidence']) {
      expect(text).not.toContain(word)
    }
  })

  it('the fixed order: verdict, risk, action, meaning, request, details, notice', () => {
    const text = show().textContent!
    const order = ['Cannot decide', 'MEDIUM RISK', 'Manual verification required', 'What this result means', 'Request ID', 'Technical details', 'Notice from the service']
    const places = order.map((part) => text.indexOf(part))
    expect(places).toEqual([...places].sort((a, b) => a - b))
    expect(places).not.toContain(-1)
  })
})
