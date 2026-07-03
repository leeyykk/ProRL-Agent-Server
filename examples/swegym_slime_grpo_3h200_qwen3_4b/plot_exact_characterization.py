#!/usr/bin/env python3
"""Draw an exact POLAR stage timeline from saved per-session UTC marks."""

from __future__ import annotations

import argparse
import html
import json
import re
from datetime import datetime, timezone
from pathlib import Path
from zoneinfo import ZoneInfo

KST = ZoneInfo("Asia/Seoul")
FONT = "Tahoma"
TRAIN_TS_RE = re.compile(r"\[(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})\]")
TIMER_RE = re.compile(r"Timer ([A-Za-z0-9_]+) (start|end)")
HEARTBEAT_RE = re.compile(r"heartbeat_interval_seconds:\s*([0-9.]+)")
GATEWAY_LOG_TS_RE = re.compile(r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),(\d+)")
HEARTBEAT_SENT_RE = re.compile(
    r"Gateway heartbeat sent .*?total_sessions=(\d+)"
)
ROLLOUT_DONE_RE = re.compile(
    r"^(\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2}),\d+\s+.*Task .*?-(\d+)-\d+ completed with \d+ results"
)
DROP_RE = re.compile(r"Dropping Polar group (\d+)")
COLLECT_RE = re.compile(r"Final collected (\d+) samples from rollout to train")

COLORS = {
    "queue": "#c6dbef",
    "init": "#9ecae1",
    "ready": "#d9d9d9",
    "run": "#4c78a8",
    "build": "#b7a1c7",
    "eval": "#8f63b8",
    "teardown": "#777777",
    "train_wait": "#d9d9d9",
    "update_weights": "#9e9e9e",
    "ref_log_probs": "#54a24b",
    "log_probs": "#72b7b2",
    "actor_train": "#f58518",
}


def parse_train_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST).astimezone(timezone.utc)


def parse_log_time(value: str) -> datetime:
    return datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST).astimezone(timezone.utc)


def parse_utc(value: str | None) -> datetime | None:
    if not value:
        return None
    return datetime.fromisoformat(value.replace("Z", "+00:00")).astimezone(timezone.utc)


def seconds(t0: datetime, t: datetime) -> float:
    return (t - t0).total_seconds()


def task_sort_key(task_id: str) -> int:
    try:
        return int(task_id.rsplit("-", 2)[1])
    except Exception:
        return 0


def text(x: float, y: float, value: str, cls: str = "", anchor: str = "start") -> str:
    class_attr = f' class="{cls}"' if cls else ""
    return f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" font-family="{FONT}"{class_attr}>{html.escape(value)}</text>'


def middle_text(x: float, y: float, value: str, cls: str = "", anchor: str = "start") -> str:
    class_attr = f' class="{cls}"' if cls else ""
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" text-anchor="{anchor}" dominant-baseline="middle" '
        f'font-family="{FONT}"{class_attr}>{html.escape(value)}</text>'
    )


def rect(
    x: float,
    y: float,
    w: float,
    h: float,
    fill: str,
    stroke: str = "#000",
    sw: float = 1.6,
    extra: str = "",
) -> str:
    extra_attr = f" {extra}" if extra else ""
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{max(w, 0.8):.1f}" height="{h:.1f}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"{extra_attr} />'
    )


def approx_text_width(value: str, font_size: float = 18.0) -> float:
    width = 0.0
    for char in value:
        if char in "ilI.,' ":
            width += 0.30
        elif char in "mwMW":
            width += 0.95
        elif char.isupper():
            width += 0.68
        else:
            width += 0.55
    return width * font_size


def legend(x: float, y: float, items: list[tuple[str, str, str | None]], cols: int) -> list[str]:
    swatch = 15.0
    row_h = 26.0
    gap = 26.0
    rows = [items[i : i + cols] for i in range(0, len(items), cols)]
    col_widths = []
    for col in range(cols):
        labels = [row[col][0] for row in rows if col < len(row)]
        col_widths.append(swatch + 8 + max((approx_text_width(label, 17.0) for label in labels), default=0.0))
    content_w = sum(col_widths) + gap * (cols - 1)
    width = content_w + 22
    height = row_h * len(rows) + 18
    out = [rect(x, y, width, height, "#fff", "#000", 1.5)]
    base_x = x + 11
    base_y = y + height / 2 - (len(rows) - 1) * row_h / 2
    for row_i, row in enumerate(rows):
        cx = base_x
        cy = base_y + row_i * row_h
        for col_i, (label, color, extra) in enumerate(row):
            out.append(rect(cx, cy - swatch / 2, swatch, swatch, color, "#000", 1.2, extra or ""))
            out.append(middle_text(cx + swatch + 8, cy, label, "legend"))
            cx += col_widths[col_i] + gap
    return out


def legend_width(items: list[tuple[str, str, str | None]], cols: int) -> float:
    swatch = 15.0
    gap = 26.0
    rows = [items[i : i + cols] for i in range(0, len(items), cols)]
    col_widths = []
    for col in range(cols):
        labels = [row[col][0] for row in rows if col < len(row)]
        col_widths.append(swatch + 8 + max((approx_text_width(label, 17.0) for label in labels), default=0.0))
    return sum(col_widths) + gap * (cols - 1) + 22


def parse_timers(train_log: Path) -> dict[str, list[tuple[datetime, datetime]]]:
    starts: dict[str, datetime] = {}
    timers: dict[str, list[tuple[datetime, datetime]]] = {}
    if not train_log.exists():
        return timers
    for raw in train_log.read_text(errors="replace").splitlines():
        ts_match = TRAIN_TS_RE.search(raw)
        timer_match = TIMER_RE.search(raw)
        if not ts_match or not timer_match:
            continue
        ts = parse_train_time(ts_match.group(1))
        name, event = timer_match.groups()
        if event == "start":
            starts[name] = ts
        elif name in starts:
            timers.setdefault(name, []).append((starts.pop(name), ts))
    return timers


def heartbeat_interval_seconds(run_dir: Path) -> float | None:
    topology = run_dir / "topology.yaml"
    if not topology.exists():
        return None
    match = HEARTBEAT_RE.search(topology.read_text(errors="replace"))
    if not match:
        return None
    try:
        interval = float(match.group(1))
    except ValueError:
        return None
    return interval if interval > 0 else None


def parse_gateway_log_time(value: str, millis: str) -> datetime:
    base = datetime.strptime(value, "%Y-%m-%d %H:%M:%S").replace(tzinfo=KST)
    return base.replace(microsecond=int(millis[:3].ljust(3, "0")) * 1000).astimezone(timezone.utc)


def actual_heartbeat_events(run_dir: Path) -> list[tuple[datetime, int | None]]:
    gateway_log = run_dir / "logs" / "polar_gateway.log"
    if not gateway_log.exists():
        return []
    events: list[tuple[datetime, int | None]] = []
    for raw in gateway_log.read_text(errors="replace").splitlines():
        if "Gateway heartbeat sent" not in raw:
            continue
        ts_match = GATEWAY_LOG_TS_RE.search(raw)
        if not ts_match:
            continue
        total_match = HEARTBEAT_SENT_RE.search(raw)
        total = int(total_match.group(1)) if total_match else None
        events.append((parse_gateway_log_time(ts_match.group(1), ts_match.group(2)), total))
    return events


def earliest_dispatch_time(
    sessions: list[tuple[str, str, list[tuple[str, datetime, datetime]], dict[str, object]]],
) -> datetime | None:
    dispatch_times: list[datetime] = []
    for _, _, _, metadata in sessions:
        marks = ((metadata.get("timing_marks") or {}).get("utc") or {})
        dispatch = parse_utc(marks.get("dispatch_started"))
        if dispatch is not None:
            dispatch_times.append(dispatch)
    return min(dispatch_times) if dispatch_times else None


def dispatch_time_for_task_sample(
    sessions: list[tuple[str, str, list[tuple[str, datetime, datetime]], dict[str, object]]],
    *,
    task_group: int,
    sample_number: int,
) -> datetime | None:
    if sample_number <= 0:
        return None
    sample_dispatches: list[tuple[str, datetime]] = []
    for task_id, session_id, _, metadata in sessions:
        if task_sort_key(task_id) != task_group:
            continue
        marks = ((metadata.get("timing_marks") or {}).get("utc") or {})
        dispatch = parse_utc(marks.get("dispatch_started"))
        if dispatch is not None:
            sample_dispatches.append((session_id, dispatch))
    sample_dispatches.sort(key=lambda item: item[0])
    if len(sample_dispatches) < sample_number:
        return None
    return sample_dispatches[sample_number - 1][1]


def infer_training_usage(run_dir: Path) -> tuple[set[int], set[int], dict[datetime, int]]:
    """Infer which rollout groups supplied actor-update samples.

    Raw rollout result JSONs contain exact stage timing, but the accepted-sample
    annotations are added only in memory. We therefore reconstruct the mapping
    from rollout completion order, explicit drop logs, and train collection logs.
    """
    events: list[tuple[datetime, int, str, int | None, int | None]] = []
    order = 0

    rollout_log = run_dir / "logs" / "polar_rollout.log"
    if rollout_log.exists():
        for raw in rollout_log.read_text(errors="replace").splitlines():
            match = ROLLOUT_DONE_RE.search(raw)
            if not match:
                continue
            events.append((parse_log_time(match.group(1)), order, "complete", int(match.group(2)), None))
            order += 1

    train_log = run_dir / "train.log"
    if train_log.exists():
        for raw in train_log.read_text(errors="replace").splitlines():
            ts_match = TRAIN_TS_RE.search(raw)
            if not ts_match:
                continue
            drop_match = DROP_RE.search(raw)
            collect_match = COLLECT_RE.search(raw)
            if drop_match:
                events.append((parse_train_time(ts_match.group(1)), order, "drop", int(drop_match.group(1)), None))
                order += 1
            elif collect_match:
                events.append((parse_train_time(ts_match.group(1)), order, "collect", None, int(collect_match.group(1))))
                order += 1

    events.sort(key=lambda item: (item[0], item[1]))
    completed: list[int] = []
    dropped: set[int] = set()
    trained: set[int] = set()
    samples_by_collect_time: dict[datetime, int] = {}

    for ts, _, kind, group_id, sample_count in events:
        if kind == "complete" and group_id is not None:
            if group_id not in dropped and group_id not in trained and group_id not in completed:
                completed.append(group_id)
        elif kind == "drop" and group_id is not None:
            dropped.add(group_id)
            completed = [item for item in completed if item != group_id]
        elif kind == "collect":
            completed = [item for item in completed if item not in dropped and item not in trained]
            if completed:
                trained.add(completed.pop(0))
            if sample_count is not None:
                samples_by_collect_time[ts] = sample_count

    return trained, dropped, samples_by_collect_time


def session_spans(path: Path) -> tuple[str, str, list[tuple[str, datetime, datetime]], dict[str, object]]:
    payload = json.loads(path.read_text())
    metadata = payload.get("metadata") or {}
    marks = ((metadata.get("timing_marks") or {}).get("utc") or {})
    if not marks:
        raise ValueError(f"{path} has no metadata.timing_marks.utc")

    def span(kind: str, start_key: str, end_key: str) -> tuple[str, datetime, datetime] | None:
        start = parse_utc(marks.get(start_key))
        end = parse_utc(marks.get(end_key))
        if start is None or end is None or end <= start:
            return None
        return kind, start, end

    spans = [
        span("queue", "dispatch_started", "init_started"),
        span("init", "init_started", "init_finished"),
        span("ready", "init_finished", "run_started"),
        span("run", "run_started", "run_finished"),
        span("build", "build_started", "build_finished"),
        span("eval", "eval_started", "eval_finished"),
        span("teardown", "teardown_started", "teardown_finished"),
    ]
    return payload["task_id"], payload["session_id"], [item for item in spans if item is not None], metadata


def load_sessions(run_dir: Path) -> list[tuple[str, str, list[tuple[str, datetime, datetime]], dict[str, object]]]:
    sessions = []
    for path in sorted((run_dir / "rollout_results").glob("task_*/*.json")):
        sessions.append(session_spans(path))
    return sessions


def filter_sessions_by_task_count(
    sessions: list[tuple[str, str, list[tuple[str, datetime, datetime]], dict[str, object]]],
    max_task_groups: int | None,
) -> list[tuple[str, str, list[tuple[str, datetime, datetime]], dict[str, object]]]:
    if max_task_groups is None:
        return sessions
    task_ids = sorted({task_id for task_id, *_ in sessions}, key=task_sort_key)
    keep = set(task_ids[:max_task_groups])
    return [item for item in sessions if item[0] in keep]


def filter_sessions_by_training_relevance(
    run_dir: Path,
    sessions: list[tuple[str, str, list[tuple[str, datetime, datetime]], dict[str, object]]],
) -> list[tuple[str, str, list[tuple[str, datetime, datetime]], dict[str, object]]]:
    trained_groups, dropped_groups, _ = infer_training_usage(run_dir)
    keep_groups = trained_groups | dropped_groups
    if not keep_groups:
        return sessions
    return [item for item in sessions if task_sort_key(item[0]) in keep_groups]


def draw_bar(
    out: list[str],
    x0: float,
    width: float,
    y: float,
    h: float,
    start: datetime,
    end: datetime,
    t0: datetime,
    t1: datetime,
    fill: str,
    sw: float = 1.4,
    extra: str = "",
) -> tuple[float, float] | None:
    s = max(start, t0)
    e = min(end, t1)
    if e <= s:
        return None
    span = max(seconds(t0, t1), 1.0)
    x = x0 + seconds(t0, s) / span * width
    w = seconds(s, e) / span * width
    out.append(rect(x, y, w, h, fill, "#000", sw, extra))
    return x, w


def make_svg(
    run_dir: Path,
    sessions: list[tuple[str, str, list[tuple[str, datetime, datetime]], dict[str, object]]],
    *,
    show_heartbeat_clock: bool = False,
    show_actual_heartbeats: bool = False,
    heartbeat_origin: datetime | None = None,
) -> str:
    by_task: dict[str, list[tuple[str, list[tuple[str, datetime, datetime]], dict[str, object]]]] = {}
    all_times: list[datetime] = []
    for task_id, session_id, spans, metadata in sessions:
        by_task.setdefault(task_id, []).append((session_id, spans, metadata))
        for _, start, end in spans:
            all_times.extend([start, end])
    if not all_times:
        raise SystemExit("No exact timing marks found")

    timers = parse_timers(run_dir / "train.log")
    heartbeat_events = actual_heartbeat_events(run_dir) if show_actual_heartbeats else []
    trained_groups, _, _ = infer_training_usage(run_dir)
    for name in ["train_wait", "update_weights", "ref_log_probs", "log_probs", "actor_train"]:
        for start, end in timers.get(name, []):
            all_times.extend([start, end])
    for ts, _ in heartbeat_events:
        all_times.append(ts)

    t0 = min(all_times)
    t1 = max(all_times)
    pad = max(3.0, seconds(t0, t1) * 0.03)
    t0 = datetime.fromtimestamp(t0.timestamp() - pad, tz=timezone.utc)
    t1 = datetime.fromtimestamp(t1.timestamp() + pad, tz=timezone.utc)

    worker_limits = next((m.get("worker_limits") for *_, m in sessions if m.get("worker_limits")), {}) or {}
    task_ids = sorted(by_task, key=task_sort_key)
    train_rows = [
        ("Train wait", "train_wait"),
        ("Weight sync", "update_weights"),
        ("Ref logp fwd", "ref_log_probs"),
        ("Current logp fwd", "log_probs"),
        ("Actor GRPO update", "actor_train"),
    ]
    heartbeat_row = "Gateway heartbeat"
    rows = (
        ([heartbeat_row] if show_heartbeat_clock or show_actual_heartbeats else [])
        + [name for name, _ in train_rows]
        + [f"Task group {task_sort_key(task_id)}" for task_id in task_ids]
    )

    width = 1960
    x0, plot_w = 335.0, 1510.0
    title_y = 36.0
    legend_y = 62.0
    top_y = 184.0
    row_gap = 64.0
    row_y = {name: top_y + i * row_gap for i, name in enumerate(rows)}
    plot_h = row_gap * (len(rows) - 1) + 62.0
    height = int(top_y + plot_h + 82)

    out = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        "<style>",
        f"text{{font-family:{FONT};fill:#111}}.legend{{font-size:17px}}.tick{{font-size:18px}}.row{{font-size:21px;font-weight:700}}.title{{font-size:32px;font-weight:700}}.xlabel{{font-size:26px;font-weight:700}}.note{{font-size:18px;font-weight:700}}.barlabel{{font-size:14px;font-weight:700}}",
        "</style>",
        '<rect width="100%" height="100%" fill="#fff" />',
    ]
    limits = (
        f"Worker limits: init={worker_limits.get('max_init_workers', '?')}, "
        f"run={worker_limits.get('max_run_workers', '?')}, "
        f"post/eval={worker_limits.get('max_postrun_workers', '?')}"
    )
    out.append(text(x0 + plot_w / 2, title_y, limits, "title", "middle"))
    out.extend(
        legend(
            330,
            legend_y,
            [
                ("Queue", COLORS["queue"], None),
                ("Init", COLORS["init"], None),
                ("Ready wait", COLORS["ready"], None),
                ("Run", COLORS["run"], None),
                ("Build", COLORS["build"], None),
                ("Eval", COLORS["eval"], None),
                ("Teardown", COLORS["teardown"], None),
                ("Not trained", COLORS["run"], 'opacity="0.48" stroke-dasharray="5 3"'),
            ],
            cols=4,
        )
    )
    train_legend_items = [
        ("Train wait", COLORS["train_wait"], None),
        ("Weight sync", COLORS["update_weights"], None),
        ("Ref logp", COLORS["ref_log_probs"], None),
        ("Current logp", COLORS["log_probs"], None),
        ("Actor GRPO", COLORS["actor_train"], None),
    ]
    out.extend(legend(x0 + plot_w - legend_width(train_legend_items, 3), legend_y, train_legend_items, cols=3))
    out.append(rect(x0, top_y - 34, plot_w, plot_h, "#fff", "#000", 2.4))

    for name in rows:
        out.append(text(x0 - 28, row_y[name] + 11, name, "row", "end"))

    if show_heartbeat_clock or show_actual_heartbeats:
        interval = heartbeat_interval_seconds(run_dir)
        clock_origin = heartbeat_origin or earliest_dispatch_time(sessions) or t0
        y = row_y[heartbeat_row]
        span = max(seconds(t0, t1), 1.0)
        origin_x = x0 + seconds(t0, clock_origin) / span * plot_w
        out.append(
            f'<line x1="{origin_x:.1f}" y1="{y:.1f}" x2="{x0 + plot_w:.1f}" y2="{y:.1f}" '
            'stroke="#000" stroke-width="2" />'
        )
        if show_actual_heartbeats and heartbeat_events:
            visible_events = [(ts, total) for ts, total in heartbeat_events if t0 <= ts <= t1]
            for event_i, (ts, total) in enumerate(visible_events):
                x = x0 + seconds(t0, ts) / span * plot_w
                out.append(
                    f'<line x1="{x:.1f}" y1="{y - 20:.1f}" x2="{x:.1f}" y2="{y + 20:.1f}" '
                    'stroke="#000" stroke-width="2.2" />'
                )
                out.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.0" fill="#111" '
                    'stroke="#000" stroke-width="2" />'
                )
                if event_i == 0 or event_i == len(visible_events) - 1:
                    label = f"{seconds(t0, ts):.0f}s"
                    if total is not None:
                        label += f" / {total}"
                    out.append(text(x, y - 28, label, "tick", "middle"))
            out.append(text(x0 + plot_w + 18, y + 6, "actual", "note"))
        elif interval is not None:
            clock = 0.0
            while clock <= max(seconds(clock_origin, t1), 0.0) + 0.1:
                x = origin_x + clock / span * plot_w
                out.append(
                    f'<line x1="{x:.1f}" y1="{y - 20:.1f}" x2="{x:.1f}" y2="{y + 20:.1f}" '
                    'stroke="#000" stroke-width="2.2" />'
                )
                out.append(
                    f'<circle cx="{x:.1f}" cy="{y:.1f}" r="5.0" fill="#fff" '
                    'stroke="#000" stroke-width="2" />'
                )
                out.append(text(x, y - 28, f"{int(clock)}s", "tick", "middle"))
                clock += interval
            out.append(text(x0 + plot_w + 18, y + 6, f"{interval:g}s", "note"))
        else:
            out.append(text(x0 + 10, y - 16, "heartbeat interval unavailable", "note"))

    for label, timer_name in train_rows:
        y = row_y[label] - 12
        for start, end in timers.get(timer_name, []):
            draw_bar(out, x0, plot_w, y, 24.0, start, end, t0, t1, COLORS[timer_name], 1.8)

    for task_id in task_ids:
        row_name = f"Task group {task_sort_key(task_id)}"
        group_id = task_sort_key(task_id)
        samples = sorted(by_task[task_id], key=lambda item: item[0])
        lane_h = 9.0
        lane_gap = 3.0
        y0 = row_y[row_name] - 26
        bar_extra = 'opacity="0.48" stroke-dasharray="5 3"' if group_id not in trained_groups else ""
        for sample_i, (_, spans, _) in enumerate(samples):
            y = y0 + sample_i * (lane_h + lane_gap)
            for kind, start, end in spans:
                draw_bar(out, x0, plot_w, y, lane_h, start, end, t0, t1, COLORS[kind], 1.2, bar_extra)

    span = max(seconds(t0, t1), 1.0)
    step = 20 if span <= 220 else 40
    tick = 0
    while tick <= span + 0.1:
        x = x0 + tick / span * plot_w
        out.append(f'<line x1="{x:.1f}" y1="{top_y + plot_h - 34:.1f}" x2="{x:.1f}" y2="{top_y + plot_h - 25:.1f}" stroke="#000" stroke-width="2" />')
        out.append(text(x, top_y + plot_h, str(int(tick)), "tick", "middle"))
        tick += step
    out.append(text(x0 + plot_w / 2, top_y + plot_h + 43, "Elapsed Time (seconds)", "xlabel", "middle"))
    out.append("</svg>")
    return "\n".join(out)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument(
        "--max-task-groups",
        type=int,
        help="Keep only the first N task groups by numeric task id for comparable figures.",
    )
    parser.add_argument(
        "--only-training-relevant-groups",
        action="store_true",
        help=(
            "Hide rollout groups that were merely prefetched and never consumed by training. "
            "Groups explicitly dropped by the accept-fraction logic are still shown."
        ),
    )
    parser.add_argument(
        "--heartbeat-clock",
        action="store_true",
        help="Add a configured gateway heartbeat interval clock lane above the timeline.",
    )
    parser.add_argument(
        "--heartbeat-actual",
        action="store_true",
        help="Add actual gateway heartbeat events parsed from logs/polar_gateway.log.",
    )
    parser.add_argument(
        "--heartbeat-origin-task-group",
        type=int,
        help="Align heartbeat 0s to the Nth sample dispatch of this task group.",
    )
    parser.add_argument(
        "--heartbeat-origin-sample",
        type=int,
        default=1,
        help="1-based sample dispatch index within --heartbeat-origin-task-group.",
    )
    args = parser.parse_args()

    run_dir = args.run_dir.resolve()
    output = args.output or run_dir / "exact_characterization.svg"
    sessions = filter_sessions_by_task_count(load_sessions(run_dir), args.max_task_groups)
    if args.only_training_relevant_groups:
        sessions = filter_sessions_by_training_relevance(run_dir, sessions)
    heartbeat_origin = None
    if args.heartbeat_origin_task_group is not None:
        heartbeat_origin = dispatch_time_for_task_sample(
            sessions,
            task_group=args.heartbeat_origin_task_group,
            sample_number=args.heartbeat_origin_sample,
        )
        if heartbeat_origin is None:
            raise SystemExit(
                "Could not find requested heartbeat origin: "
                f"task_group={args.heartbeat_origin_task_group}, "
                f"sample={args.heartbeat_origin_sample}"
            )
    output.write_text(
        make_svg(
            run_dir,
            sessions,
            show_heartbeat_clock=args.heartbeat_clock,
            show_actual_heartbeats=args.heartbeat_actual,
            heartbeat_origin=heartbeat_origin,
        )
    )
    print(output)


if __name__ == "__main__":
    main()
