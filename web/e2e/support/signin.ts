// S1 as the e2e specs see it (web/src/pages/SignIn.tsx). The banner copy is SIGN_IN_COPY in
// web/src/pages/signInOutcome.ts, repeated here because the e2e project does not compile src/.
import type { Page, Request, Response } from '@playwright/test'

export const NEXT = '/history?verdict=a&verdict=b'
export const SIGNIN_WITH_NEXT = '/signin?next=%2Fhistory%3Fverdict%3Da%26verdict%3Db'

export const COPY = {
  invalid: 'The analyst ID or password is not correct. Check both and try again.',
  denied: 'This account does not have access to the analyst console. Ask your administrator for access.',
} as const

export function signInForm(page: Page) {
  return {
    heading: page.getByRole('heading', { level: 1, name: 'Sign in to the analyst console' }),
    username: page.locator('#si-user'),
    password: page.locator('#si-pass'),
    submit: page.getByRole('button', { name: 'Sign in', exact: true }),
    alert: page.getByRole('alert'),
  }
}

const isSignInPost = (r: Request | Response) => {
  const request = 'request' in r ? r.request() : r
  return request.method() === 'POST' && new URL(request.url()).pathname === '/api/session'
}

export const signInRequest = (page: Page) => page.waitForRequest(isSignInPost)
export const signInResponse = (page: Page) => page.waitForResponse(isSignInPost)

/** Counts the sign-in POSTs the page sends from now on. */
export function countSignInPosts(page: Page): () => number {
  let count = 0
  page.on('request', (request) => {
    if (isSignInPost(request)) count += 1
  })
  return () => count
}
