// The duration read (Stage 18 PR A; prototype lines 1290-1295): the browser's Audio element on the file's object URL.
// The CSP allows it through media-src blob: (deploy/nginx_security_headers.conf). The first of loadedmetadata, error
// and the fallback wins; a duration that is not finite is one the browser could not read (prototype line 1292). The
// object URL never leaves the browser and is revoked once, whatever ends the read.
export type DurationRead = number | 'unreadable'

const FALLBACK_MS = 3000 // prototype line 1294

// Returns a function that cancels the read; onDone is then never called.
export function readDuration(file: Blob, onDone: (duration: DurationRead) => void): () => void {
  const url = URL.createObjectURL(file)
  const audio = new Audio()
  let open = true

  function stop(): boolean {
    if (!open) return false
    open = false
    clearTimeout(fallback)
    audio.onloadedmetadata = null
    audio.onerror = null
    audio.removeAttribute('src')
    URL.revokeObjectURL(url)
    return true
  }

  function finish(duration: DurationRead): void {
    if (stop()) onDone(duration)
  }

  audio.preload = 'metadata'
  audio.onloadedmetadata = () => finish(Number.isFinite(audio.duration) ? audio.duration : 'unreadable')
  audio.onerror = () => finish('unreadable')
  // eslint-disable-next-line no-restricted-syntax -- One-shot, cleared on settle, not polling (Entry 145 Q9): without it a read that fires no event leaves Analyze disabled for good; 'unreadable' is safe, the service re-checks.
  const fallback = setTimeout(() => finish('unreadable'), FALLBACK_MS)
  audio.src = url
  return () => {
    stop()
  }
}
