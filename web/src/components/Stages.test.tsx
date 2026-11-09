import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import Stages from './Stages'

afterEach(cleanup)

const item = (name: RegExp) => screen.getAllByRole('listitem').find((li) => name.test(li.textContent ?? ''))!

describe('Stages', () => {
  it('a stage before the current one is done, with a check and a hidden "(done)"', () => {
    render(<Stages names={['One']} current={1} />)
    const stage = screen.getByRole('listitem')
    expect(stage.className).toBe('stage done')
    expect(stage.textContent).toBe('One (done)')
    expect(stage.querySelector('svg')).not.toBeNull()
    expect(stage.getAttribute('aria-current')).toBeNull()
  })

  it('the current stage has aria-current="step" and a spinner', () => {
    render(<Stages names={['One', 'Two']} current={1} />)
    const stage = item(/Two/)
    expect(stage.className).toBe('stage now')
    expect(stage.getAttribute('aria-current')).toBe('step')
    expect(stage.querySelector('.spin')).not.toBeNull()
  })

  it('a stage after the current one only has its name', () => {
    render(<Stages names={['One', 'Two']} current={0} />)
    const stage = item(/Two/)
    expect(stage.className).toBe('stage')
    expect(stage.textContent).toBe('Two')
    expect(stage.children).toHaveLength(0)
  })
})
