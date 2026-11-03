// The status time on S10 (Stage 18 PR 7b): the prototype's format (docs/design/analyst-console-prototype.html line
// 644), in the browser's time zone with its offset, absolute only (a relative "x minutes ago" would need a timer).
const FORMAT = new Intl.DateTimeFormat('en-GB', {
  day: 'numeric',
  month: 'short',
  year: 'numeric',
  hour: '2-digit',
  minute: '2-digit',
  timeZoneName: 'shortOffset',
})

// A value that is not a date is shown as given (prototype line 649).
export function formatCheckedAt(iso: string): string {
  const date = new Date(iso)
  return Number.isNaN(date.getTime()) ? iso : FORMAT.format(date)
}
