// The time on a History row (Stage 18 PR C): the record's created_at in the browser's time zone, as S10 does it
// (status/checkedAt.ts). The prototype's cell format (line 645) has no offset: the column header carries the zone
// (prototype line 655). The header shows today's offset, so across a DST change an older row's time is right and its
// label is not, as in the prototype. parseHistory admits only real dates, so there is no fallback here.
const CELL = new Intl.DateTimeFormat('en-GB', { day: 'numeric', month: 'short', year: 'numeric', hour: '2-digit', minute: '2-digit' })

const OFFSET = new Intl.DateTimeFormat('en-GB', { hour: '2-digit', minute: '2-digit', timeZoneName: 'shortOffset' })

export function formatAnalyzedAt(iso: string): string {
  return CELL.format(new Date(iso))
}

export function timeZoneLabel(): string {
  return OFFSET.formatToParts(new Date()).find((part) => part.type === 'timeZoneName')?.value ?? 'local time'
}
