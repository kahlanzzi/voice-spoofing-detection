// The app's routes (HANDOFF §12), equal to the nginx app routes (deploy/nginx.conf, team decision A2);
// routes.test.ts fails if the lists differ. Matching is nginx's: exact, case-sensitive, no trailing slash,
// no percent-encoding, and /history/ takes only a full request ID (docs/api_design.md §2).

export const REQUEST_ID = /^req_[0-9a-f]{32}$/

export const APP_ROUTES = ['/signin', '/analyze', '/history', '/history/:requestId', '/overview', '/status'] as const

export type AppRoute = (typeof APP_ROUTES)[number]

// nginx sends / here (302), and a rejected next lands here.
export const DEFAULT_ROUTE = '/analyze' satisfies AppRoute

const DETAILS_PREFIX = '/history/'

export function appRoute(pathname: string): AppRoute | null {
  if (pathname.startsWith(DETAILS_PREFIX)) {
    return REQUEST_ID.test(pathname.slice(DETAILS_PREFIX.length)) ? '/history/:requestId' : null
  }
  return APP_ROUTES.find((route) => route === pathname && !route.includes(':')) ?? null
}
