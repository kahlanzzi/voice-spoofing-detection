// The security headers nginx adds to every answer (team decision A13), read from the file nginx includes, so the
// e2e check follows deploy/nginx_security_headers.conf instead of a copy of it. Node only (no Playwright import).
import { readFileSync } from 'node:fs'
import { resolve } from 'node:path'

export const SECURITY_HEADERS_CONF = resolve(import.meta.dirname, '../../../deploy/nginx_security_headers.conf')

const ADD_HEADER = /^add_header\s+(\S+)\s+"([^"]*)"\s+always;$/

/** Header name (lower case) → value, one per `add_header … always;` line; any other add_header line throws. */
export function securityHeaders(text = readFileSync(SECURITY_HEADERS_CONF, 'utf8')): Record<string, string> {
  const lines = text.split('\n').map((line) => line.trim()).filter((line) => line.startsWith('add_header'))
  return Object.fromEntries(lines.map((line) => {
    const match = ADD_HEADER.exec(line)
    if (!match) throw new Error(`Unexpected add_header line in ${SECURITY_HEADERS_CONF}: ${line}`)
    return [match[1]!.toLowerCase(), match[2]!]
  }))
}
