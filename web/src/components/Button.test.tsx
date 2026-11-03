import { cleanup, fireEvent, render, screen } from '@testing-library/react'
import { createRef } from 'react'
import { afterEach, describe, expect, it, vi } from 'vitest'
import Button, { type ButtonVariant } from './Button'
import Icon from './Icon'

// Button (Stage 18 PR 8a; HANDOFF line 137). The markup must stay what S1, S10 and the shell rendered before the
// extraction (Entry 141): the prototype's class names, and aria-busy only while busy.

afterEach(cleanup)

const button = () => screen.getByRole('button')

describe('Button', () => {
  it('is a native button of type button by default', () => {
    render(<Button variant="primary">Go</Button>)
    expect(button().tagName).toBe('BUTTON')
    expect(button().getAttribute('type')).toBe('button')
  })

  it('type submit submits its form', () => {
    const onSubmit = vi.fn((event: SubmitEvent) => event.preventDefault())
    render(
      <form onSubmit={(event) => onSubmit(event.nativeEvent as SubmitEvent)}>
        <Button type="submit" variant="primary">
          Send
        </Button>
      </form>,
    )
    fireEvent.click(button())
    expect(onSubmit).toHaveBeenCalledTimes(1)
  })

  it.each<[ButtonVariant, string]>([
    ['primary', 'btn btn-primary'],
    ['secondary', 'btn btn-secondary'],
    ['quiet', 'btn btn-quiet'],
    ['destructive', 'btn btn-danger'],
    ['link', 'btn-link'],
  ])('variant %s has the class "%s"', (variant, className) => {
    render(<Button variant={variant}>Go</Button>)
    expect(button().className).toBe(className)
  })

  it('size sm adds btn-sm after the variant', () => {
    render(
      <Button variant="quiet" size="sm">
        Check now
      </Button>,
    )
    expect(button().className).toBe('btn btn-quiet btn-sm')
  })

  it('busy: disabled, aria-busy true, an aria-hidden spinner, the label kept', () => {
    render(
      <Button variant="quiet" busy>
        Check now
      </Button>,
    )
    expect(button().hasAttribute('disabled')).toBe(true)
    expect(button().getAttribute('aria-busy')).toBe('true')
    expect(button().querySelector('.spin')?.getAttribute('aria-hidden')).toBe('true')
    expect(button().textContent).toBe('Check now')
  })

  it('busy replaces the icon with the spinner, in the same place', () => {
    const { rerender } = render(
      <Button variant="quiet" icon={<Icon name="refresh" size={14} />}>
        Check now
      </Button>,
    )
    expect(button().firstElementChild?.tagName.toLowerCase()).toBe('svg')
    rerender(
      <Button variant="quiet" busy icon={<Icon name="refresh" size={14} />}>
        Check now
      </Button>,
    )
    expect(button().querySelector('svg')).toBeNull()
    expect(button().firstElementChild?.className).toBe('spin')
  })

  it('not busy: no aria-busy attribute, no spinner, not disabled', () => {
    render(<Button variant="primary">Sign in</Button>)
    expect(button().hasAttribute('aria-busy')).toBe(false)
    expect(button().hasAttribute('disabled')).toBe(false)
    expect(button().querySelector('.spin')).toBeNull()
  })

  it('disabled without busy has no aria-busy and no spinner', () => {
    render(
      <Button variant="primary" disabled>
        Sign in
      </Button>,
    )
    expect(button().hasAttribute('disabled')).toBe(true)
    expect(button().hasAttribute('aria-busy')).toBe(false)
    expect(button().querySelector('.spin')).toBeNull()
  })

  it('a busy button does not call onClick', () => {
    const onClick = vi.fn()
    render(
      <Button variant="primary" busy onClick={onClick}>
        Signing in…
      </Button>,
    )
    fireEvent.click(button())
    expect(onClick).not.toHaveBeenCalled()
  })

  it('forwards the ref and aria-* props to the native button', () => {
    const ref = createRef<HTMLButtonElement>()
    render(
      <Button variant="secondary" ref={ref} aria-controls="x" aria-pressed={false}>
        Retry
      </Button>,
    )
    expect(ref.current).toBe(button())
    expect(button().getAttribute('aria-controls')).toBe('x')
    expect(button().getAttribute('aria-pressed')).toBe('false')
  })
})
