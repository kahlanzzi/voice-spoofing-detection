// The sign-in `next` parameter: where S1 sends the analyst after sign-in (HANDOFF §6 "lands on the requested
// page", §9 "returns to the same place"). An open-redirect defence: the path must be one of the protected app
// routes exactly, so no other origin, scheme or path can get through. The query is kept byte-for-byte (repeated
// verdict= included); with a fixed same-origin path it cannot change the destination. next lives only in the
// URL: never stored, never logged (eslint.config.js bans both in src/routing).
import { appRoute, DEFAULT_ROUTE } from './routes'

// The longest real value, /history with every filter, is about 150 characters.
export const NEXT_MAX_LENGTH = 512

// Control characters, whitespace and '#', anywhere in the value.
function hasForbiddenChar(value: string): boolean {
  return [...value].some((char) => {
    const code = char.codePointAt(0) ?? 0
    return code < 0x20 || code === 0x7f || /[\s#]/.test(char)
  })
}

// raw is URLSearchParams.get('next'). Anything not accepted → /analyze.
export function safeNext(raw: string | null): string {
  if (raw === null || raw === '' || raw.length > NEXT_MAX_LENGTH || hasForbiddenChar(raw)) return DEFAULT_ROUTE
  const mark = raw.indexOf('?')
  const route = appRoute(mark === -1 ? raw : raw.slice(0, mark))
  if (route === null || route === '/signin') return DEFAULT_ROUTE
  return raw
}

// Where the guard sends a request for a protected route without a session.
export function signInPath(pathname: string, search: string): string {
  return `/signin?${new URLSearchParams({ next: pathname + search })}`
}
