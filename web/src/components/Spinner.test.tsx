import { cleanup, render } from '@testing-library/react'
import { afterEach, expect, it } from 'vitest'
import Spinner from './Spinner'

// Spinner (Stage 18 PR 8a; HANDOFF line 149): decorative, hidden from screen readers.

afterEach(cleanup)

it('is an aria-hidden span.spin', () => {
  const { container } = render(<Spinner />)
  expect(container.innerHTML).toBe('<span class="spin" aria-hidden="true"></span>')
})
