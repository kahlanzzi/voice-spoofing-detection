// The selected recording (Stage 18 PR 8b; A23, Entry 130), read by React through useSyncExternalStore(subscribe,
// getSnapshot) from 8d on. A module store, not component state, so the File survives the guarded page unmounting
// when the session expires (HANDOFF line 211). In memory only: a reload loses it, and neither the File nor its name
// is ever stored or logged (HANDOFF line 280).
//
// The draft belongs to the analyst who chose it (owner), because the session forgets the username on expiry. An
// explicit sign-out clears it (prototype line 1493). Expiry and every other state keep it, unavailable included (A19:
// a failed check never signs the analyst out), and the next sign-in compares usernames: another analyst clears it,
// the same one keeps it (Entry 141 Q4). An expiry during the DELETE wins in session.ts, so markSignedOut never runs
// and the File stays until the next sign-in (Entry 143 Q14).
import { getSnapshot as getSession, subscribe as subscribeSession } from '../session/session'

interface FileDraft {
  readonly file: File
  readonly owner: string
}

let draft: FileDraft | null = null
const listeners = new Set<() => void>()

export function getSnapshot(): File | null {
  return draft === null ? null : draft.file
}

export function subscribe(listener: () => void): () => void {
  listeners.add(listener)
  return () => {
    listeners.delete(listener)
  }
}

// As in session.ts: a copy, so a listener that (un)subscribes does not change this round; a throwing listener does
// not stop the others, and its error is rethrown in a microtask. Not a timer.
function notify(): void {
  let failed = false
  let firstError: unknown
  for (const listener of [...listeners]) {
    try {
      listener()
    } catch (error) {
      if (!failed) {
        failed = true
        firstError = error
      }
    }
  }
  if (failed) {
    queueMicrotask(() => {
      throw firstError
    })
  }
}

// Ignored unless signed in, so a draft always has an owner.
export function setFileDraft(file: File): void {
  const session = getSession()
  if (session.status !== 'signedIn') return
  if (draft !== null && draft.file === file && draft.owner === session.username) return
  draft = { file, owner: session.username }
  notify()
}

export function clearFileDraft(): void {
  if (draft === null) return
  draft = null
  notify()
}

subscribeSession(() => {
  if (draft === null) return
  const session = getSession()
  if (session.status === 'signedOut' && session.reason === 'signOut') clearFileDraft()
  else if (session.status === 'signedIn' && session.username !== draft.owner) clearFileDraft()
})
