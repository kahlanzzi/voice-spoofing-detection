// The page-change rules (HANDOFF lines 67-71 and 174; prototype lines 759 and 788-793), in one place: the root
// layout renders this once, beside every route, so no page has its own copy.
//
// A page is on screen when an h1 is (the Checking view has none). On the first page of a document load, redirects
// on load included, only the title is set: focus stays at the top, so the skip link is the first Tab (prototype
// line 1520). After that, each move to another pathname, by link, redirect or Back/Forward (HANDOFF line 72),
// moves focus to the h1, announces "<Page> page" politely, sets the title and scrolls to the top. Two exceptions:
// a query-only change (History filters, HANDOFF lines 74-76) is not a page change, and Back/Forward to /history
// leaves the scroll alone, which PR 12 restores from history.state (HANDOFF line 69). New analysis on /analyze
// (RESTART_PAGE, PR 8b) is one, once per entry: the effect also runs when the session changes. A result arriving on
// /analyze changes the name and the title only (prototype lines 721, 789): the result moves focus itself (PR B).
//
// useLayoutEffect: it runs after the commit and before paint, so the new page is never painted at the old scroll
// position. No timer: the message goes in a new node, which is announced even when the text repeats (the
// prototype's 40 ms timer, line 659). Written through a ref, so the effect sets no React state.
import { useLayoutEffect, useRef } from 'react'
import { useLocation, useNavigationType } from 'react-router'
import { useSession } from '../session/useSession'
import { useAnalysis } from '../upload/useAnalysis'
import { documentTitle, isPageRestart, pageName } from './pageNames'

const EXPIRED_MESSAGE = 'Your session expired. Sign in again to continue.'
// Prototype line 1496, after the rail's Sign out (PR 6b).
const SIGNED_OUT_MESSAGE = 'Signed out.'

export default function PageChange() {
  const { key, pathname, state } = useLocation()
  const navigationType = useNavigationType()
  const session = useSession()
  const showsResult = useAnalysis().status === 'result'
  const live = useRef<HTMLDivElement>(null)
  const lastPage = useRef<string | null>(null)
  const lastKey = useRef<string | null>(null)

  useLayoutEffect(() => {
    const h1 = document.querySelector<HTMLElement>('h1')
    if (h1 === null) return
    document.title = documentTitle(pathname, showsResult)
    const previous = lastPage.current
    lastPage.current = pathname
    const restart = key !== lastKey.current && isPageRestart(state)
    lastKey.current = key
    if (previous === null || (previous === pathname && !restart)) return

    if (!(navigationType === 'POP' && pathname === '/history')) window.scrollTo(0, 0)
    h1.focus({ preventScroll: true })
    const reason = pathname === '/signin' && session.status === 'signedOut' ? session.reason : 'none'
    const message = document.createElement('p')
    message.textContent =
      reason === 'expired' ? EXPIRED_MESSAGE : reason === 'signOut' ? SIGNED_OUT_MESSAGE : `${pageName(pathname, showsResult)} page`
    live.current?.replaceChildren(message)
  }, [key, pathname, state, navigationType, session, showsResult])

  return <div className="sr" aria-live="polite" aria-atomic="true" id="page-announcer" ref={live} />
}
