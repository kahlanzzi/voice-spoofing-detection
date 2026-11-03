// Vitest setup (Stage 18 PR 6b, Entry 137 Open item): jsdom does not implement window.scrollTo and logs "Not
// implemented" for every call. A spy before each test, so a test can still assert the calls (PageChange.test.tsx
// spies on the same method) and vi.restoreAllMocks() in a test's afterEach is undone by the next beforeEach. Files
// in the node environment (@vitest-environment node) have no window and are left alone.
import { beforeEach, vi } from 'vitest'

beforeEach(() => {
  if (typeof window === 'undefined') return
  vi.spyOn(window, 'scrollTo').mockImplementation(() => undefined)
})
