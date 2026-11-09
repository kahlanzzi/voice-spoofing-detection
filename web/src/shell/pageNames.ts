// Page names for document.title and the page-change announcement (prototype lines 720 and 789; HANDOFF lines 71
// and 174). The h1 of each page says the same. NotFound has no prototype name: its h1.
import { appRoute, type AppRoute } from '../routing/routes'

export const APP_NAME = 'Voice Check analyst console'

export const PAGE_NAMES: Record<AppRoute, string> = {
  '/signin': 'Sign in',
  '/analyze': 'Analyze a recording',
  '/history': 'History',
  '/history/:requestId': 'Analysis details',
  '/overview': 'Overview',
  '/status': 'System status',
}

export const NOT_FOUND_NAME = 'Page not found'

// /analyze while it shows a result (prototype line 721).
export const RESULT_NAME = 'Analysis result'

// New analysis on /analyze (PR 8b): the location state that makes the same page a page change (prototype lines
// 755-759). A constant, no data, so nothing about the analysis reaches the history entry (Entry 143 Q16).
export const RESTART_PAGE = { restartPage: true } as const

export function isPageRestart(state: unknown): boolean {
  return typeof state === 'object' && state !== null && 'restartPage' in state
}

export function pageName(pathname: string, showsResult = false): string {
  const route = appRoute(pathname)
  if (route === '/analyze' && showsResult) return RESULT_NAME
  return route === null ? NOT_FOUND_NAME : PAGE_NAMES[route]
}

// "<name> · Voice Check analyst console" (prototype line 789).
export function documentTitle(pathname: string, showsResult = false): string {
  return `${pageName(pathname, showsResult)} · ${APP_NAME}`
}
