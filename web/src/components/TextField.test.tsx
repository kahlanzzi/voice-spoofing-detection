import { cleanup, render, screen } from '@testing-library/react'
import { createRef } from 'react'
import { afterEach, describe, expect, it } from 'vitest'
import TextField from './TextField'

// TextField (Stage 18 PR 8a; HANDOFF line 138). The wiring must stay S1's (Entry 141): `<id>-err`, aria-invalid,
// and describedBy only while there is no error.

afterEach(cleanup)

const noop = () => undefined
const input = () => screen.getByRole('textbox')

describe('TextField', () => {
  it('the label names the input', () => {
    render(<TextField id="si-user" label="Analyst ID" value="" onChange={noop} />)
    expect(screen.getByLabelText('Analyst ID')).toBe(input())
    expect(input().className).toBe('input')
  })

  it('no error: aria-invalid false, no aria-describedby, no type attribute', () => {
    render(<TextField id="si-user" label="Analyst ID" value="" onChange={noop} />)
    expect(input().getAttribute('aria-invalid')).toBe('false')
    expect(input().hasAttribute('aria-describedby')).toBe(false)
    expect(input().hasAttribute('type')).toBe(false)
    expect(document.querySelector('.field-err')).toBeNull()
  })

  it('error: the alert icon and text with id <id>-err, described by it, aria-invalid', () => {
    render(<TextField id="si-user" label="Analyst ID" value="" onChange={noop} error="Enter your analyst ID." />)
    const message = document.getElementById('si-user-err')!
    expect(message.className).toBe('field-err')
    expect(message.textContent).toBe('Enter your analyst ID.')
    expect(message.querySelector('svg')?.getAttribute('aria-hidden')).toBe('true')
    expect(input().getAttribute('aria-describedby')).toBe('si-user-err')
    expect(input().getAttribute('aria-invalid')).toBe('true')
  })

  it('invalid without text: aria-invalid true, described by describedBy', () => {
    render(<TextField id="si-user" label="Analyst ID" value="" onChange={noop} invalid describedBy="si-banner" />)
    expect(input().getAttribute('aria-invalid')).toBe('true')
    expect(input().getAttribute('aria-describedby')).toBe('si-banner')
    expect(document.querySelector('.field-err')).toBeNull()
  })

  it('an error replaces describedBy', () => {
    render(<TextField id="si-user" label="Analyst ID" value="" onChange={noop} invalid describedBy="si-banner" error="Enter your analyst ID." />)
    expect(input().getAttribute('aria-describedby')).toBe('si-user-err')
  })

  it('a hint is described first, before the error', () => {
    render(<TextField id="from" label="From" value="" onChange={noop} hint="Day, month, year." error="Too early." />)
    expect(document.getElementById('from-hint')?.textContent).toBe('Day, month, year.')
    expect(input().getAttribute('aria-describedby')).toBe('from-hint from-err')
  })

  it('action renders after the input inside .pw', () => {
    render(<TextField id="si-pass" label="Password" value="" onChange={noop} action={<button type="button">Show</button>} />)
    const box = document.querySelector('.field > .pw')!
    expect([...box.children].map((child) => child.tagName)).toEqual(['INPUT', 'BUTTON'])
  })

  it('name, type, autoComplete and the ref reach the input', () => {
    const ref = createRef<HTMLInputElement>()
    render(<TextField id="si-pass" label="Password" value="x" onChange={noop} name="password" type="password" autoComplete="current-password" ref={ref} />)
    const field = screen.getByLabelText('Password') as HTMLInputElement
    expect(ref.current).toBe(field)
    expect([field.name, field.type, field.autocomplete, field.value]).toEqual(['password', 'password', 'current-password', 'x'])
  })
})
