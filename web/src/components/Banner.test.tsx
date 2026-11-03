import { cleanup, render } from '@testing-library/react'
import { afterEach, describe, expect, it } from 'vitest'
import Banner from './Banner'

// Banner (Stage 18 PR 8a; HANDOFF line 143). The role is the caller's, with no default, so S1's alert and status and
// S10's notice without a role announce exactly as before the extraction (Entry 141).

afterEach(cleanup)

const banner = () => document.body.firstElementChild!.firstElementChild!

describe('Banner', () => {
  it.each(['info', 'warn', 'err'] as const)('tone %s has the class banner-%s', (tone) => {
    render(
      <Banner tone={tone} icon="alert">
        Text
      </Banner>,
    )
    expect(banner().className).toBe(`banner-${tone}`)
  })

  it.each(['alert', 'status'] as const)('role %s is set as given', (role) => {
    render(
      <Banner tone="err" icon="alert" role={role}>
        Text
      </Banner>,
    )
    expect(banner().getAttribute('role')).toBe(role)
  })

  it('no role given: no role attribute', () => {
    render(
      <Banner tone="warn" icon="alert">
        Text
      </Banner>,
    )
    expect(banner().hasAttribute('role')).toBe(false)
  })

  it('an aria-hidden icon, then the text in a span', () => {
    render(
      <Banner tone="info" icon="clock">
        Your session expired.
      </Banner>,
    )
    expect([...banner().children].map((child) => child.tagName.toLowerCase())).toEqual(['svg', 'span'])
    expect(banner().querySelector('svg')?.getAttribute('aria-hidden')).toBe('true')
    expect(banner().querySelector('span')?.textContent).toBe('Your session expired.')
  })

  it('the id is passed, and absent when not given', () => {
    const { rerender } = render(
      <Banner tone="err" icon="alert" id="si-banner">
        Text
      </Banner>,
    )
    expect(banner().id).toBe('si-banner')
    rerender(
      <Banner tone="err" icon="alert">
        Text
      </Banner>,
    )
    expect(banner().hasAttribute('id')).toBe(false)
  })
})
