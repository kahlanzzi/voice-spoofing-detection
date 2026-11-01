import { cleanup, render, screen } from '@testing-library/react'
import { afterEach, expect, it, vi } from 'vitest'
import App from './App'

afterEach(() => {
  cleanup()
  vi.restoreAllMocks()
})

it('renders one h1 and makes no request', () => {
  const fetchSpy = vi.spyOn(globalThis, 'fetch')
  render(<App />)
  expect(screen.getAllByRole('heading', { level: 1 })).toHaveLength(1)
  expect(screen.getByRole('heading', { level: 1 }).textContent).toBe('Voice Check analyst console')
  expect(fetchSpy).not.toHaveBeenCalled()
})
