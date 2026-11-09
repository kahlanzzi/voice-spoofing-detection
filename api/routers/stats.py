"""GET /api/stats (docs/api_design.md §4.7).

Operational figures for an inclusive period of days, computed by AnalysisRepository.stats() in one
snapshot; the web app computes none of them (ADR 0008 §8). Days are UTC days (open decision 6;
create_app refuses another stats_time_zone), read with the §4.4 date rules, and the period filter is
the one history uses, so total and counts equal GET /api/analyses for the same from and to. Mock
and real records are both counted. monitoring_marker comes from the analyzer's info() (§8), never
from a constant here. No accuracy, EER or other quality metric. Bad queries answer 400
invalid_request with request_id null (§6). Nothing is logged here.
"""
from __future__ import annotations

from datetime import date, timedelta
from typing import Annotated

from fastapi import APIRouter, Query, Request

from api.repository import VERDICTS
from api.routers.analyses import invalid_request, parse_day, to_stored
from api.routers.status import utc_now_iso
from api.schemas import AwaitingNote, MonitoringMarker, StatsDay, StatsResponse, StatsToday, VerdictCounts

router = APIRouter()

DEFAULT_PERIOD_DAYS = 7  # §4.7: the last 7 days, including today
MAX_PERIOD_DAYS = 366  # §4.7: a longer period → 400
RECENT_AWAITING = 6  # §4.7: awaiting_note.recent holds the 6 newest


def period(first: date | None, last: date | None, today: date) -> tuple[date, date]:
    """The §4.7 period: a missing to is today, a missing from is 6 days before to. Raises 400 if from
    is after to, if the period passes MAX_PERIOD_DAYS or if it would start before 0001-01-01."""
    last = today if last is None else last
    if first is None:
        try:
            first = last - timedelta(days=DEFAULT_PERIOD_DAYS - 1)
        except OverflowError:
            raise invalid_request() from None
    if first > last or (last - first).days + 1 > MAX_PERIOD_DAYS:
        raise invalid_request()
    return first, last


@router.get("/stats", response_model=StatsResponse, response_model_by_alias=True)
def get_stats(
    request: Request,
    day_from: Annotated[str | None, Query(alias="from")] = None,
    to: str | None = None,
) -> StatsResponse:
    generated_at = utc_now_iso()
    today = generated_at[:10]  # §4.7: today is the day of generated_at
    first, last = period(parse_day(day_from), parse_day(to), date.fromisoformat(today))
    figures = request.app.state.repository.stats(first.isoformat(), last.isoformat(), today, RECENT_AWAITING)
    info = request.app.state.analyzer.info()

    days = []
    for offset in range((last - first).days + 1):
        day = (first + timedelta(days=offset)).isoformat()
        counts = figures.per_day.get(day, dict.fromkeys(VERDICTS, 0))
        days.append(StatsDay(date=day, **counts, total=sum(counts.values())))
    share = None if figures.total == 0 else round(figures.counts["uncertain"] / figures.total, 4)
    return StatsResponse.model_validate({
        "from": first.isoformat(),
        "to": last.isoformat(),
        "time_zone": request.app.state.config.stats_time_zone,
        "generated_at": generated_at,
        "today": StatsToday(date=today, total=figures.today_total),
        "total": figures.total,
        "counts": VerdictCounts(**figures.counts),
        "uncertain_share": share,
        "median_processing_ms": figures.median_processing_ms,
        "awaiting_note": AwaitingNote(total=figures.awaiting_total,
                                      recent=[to_stored(record, note) for record, note in figures.awaiting_recent]),
        "days": days,
        "monitoring_marker": MonitoringMarker(deferral_on_fit=info.deferral_on_fit,
                                              policy_version=info.policy_version, mock=info.mock),
    })
