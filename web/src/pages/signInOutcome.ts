// What S1 shows for a failed POST /api/session (docs/api_design.md §4.8 "Errors for S1", §6). A variant other
// than unavailable needs its exact status and code; anything else, unknown or malformed included, is
// unavailable, so no unexpected answer can read as a wrong password. The copy is the prototype's
// (docs/design/analyst-console-prototype.html lines 834-837); the server's message is never shown.
import { isHttpError } from '../api/errors'

export type SignInError = 'invalid' | 'denied' | 'locked' | 'unavailable'

export const SIGN_IN_COPY: Record<SignInError, string> = {
  invalid: 'The analyst ID or password is not correct. Check both and try again.',
  denied: 'This account does not have access to the analyst console. Ask your administrator for access.',
  locked: 'This account is temporarily locked. Try again later, or ask your administrator.',
  unavailable: 'Sign-in is temporarily unavailable. Try again in a few minutes.',
}

// 400 invalid_request is answered before the account is looked up and is the same for every username
// (api/routers/session.py lines 87-96), e.g. a password over password_max_chars: a password the server
// refuses to check cannot be correct, and nothing about the account is revealed.
const KNOWN: Record<string, SignInError> = {
  '401 invalid_credentials': 'invalid',
  '400 invalid_request': 'invalid',
  '403 access_denied': 'denied',
  '429 account_locked': 'locked',
}

// nginx's 429 too_many_requests and 503 analysis_unavailable, the API's 503 sign_in_unavailable and a network
// error are all "sign-in service unavailable" (§4.8), like everything not listed above.
export function signInOutcome(error: unknown): SignInError {
  if (!isHttpError(error)) return 'unavailable'
  return KNOWN[`${error.status} ${error.code}`] ?? 'unavailable'
}
