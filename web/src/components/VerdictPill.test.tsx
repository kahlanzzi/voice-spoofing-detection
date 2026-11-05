import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import VerdictPill from './VerdictPill'

afterEach(cleanup)

describe('VerdictPill', () => {
  it.each([
    ['synthetic', 'HIGH', 'Likely synthetic · HIGH', 'p-syn'],
    ['uncertain', 'MEDIUM', 'Cannot decide · MEDIUM', 'p-unc'],
    ['authentic', 'LOW', 'Likely authentic · LOW', 'p-auth'],
  ] as const)('%s: icon, words and colour class', (prediction, risk, text, className) => {
    render(<VerdictPill prediction={prediction} risk={risk} />)
    const pill = screen.getByText(text)
    expect(pill.className).toBe(`pill ${className}`)
    expect(pill.querySelector('svg')).not.toBeNull()
  })

  it('shows the risk it is given, not the one the verdict usually has', () => {
    render(<VerdictPill prediction="uncertain" risk="HIGH" />)
    expect(screen.getByText('Cannot decide · HIGH')).toBeTruthy()
  })
})
