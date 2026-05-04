"""
commentary/generator.py
========================
Rule-based template NLG that translates detected race events into
on-screen commentary strings.

Follows the Barzilay & Lapata (2005) content-planning approach used
in live sports ticker systems — event types → parameterised templates.
"""

from __future__ import annotations
import random
from analysis.race_stats import RaceEvent, EventType
from analysis.penalty_detector import PenaltyEvent, PenaltyType

DEFAULT_HOLD_FRAMES = 90   # 3 s at 30 fps

# ── Template bank ─────────────────────────────────────────────────────────────

RACE_TEMPLATES: dict[EventType, list[str]] = {
    EventType.OVERTAKE: [
        "{overtaker} makes a move on {overtaken}!",
        "Brilliant overtake! {overtaker} goes around the outside of {overtaken}.",
        "{overtaker} dives down the inside — {overtaken} has to give way.",
        "Position change! {overtaker} gets through on {overtaken}.",
        "{overtaken} loses a place — {overtaker} is through!",
        "What a move by {overtaker}! Takes the position from {overtaken}.",
        "{overtaker} has been hunting {overtaken} — and finally makes it stick!",
        "Wheel to wheel into the corner — {overtaker} comes out ahead.",
    ],
    EventType.PIT_ENTRY: [
        "{car_id} is in! Box, box, box.",
        "Pit stop for {car_id} — mechanics ready.",
        "{car_id} heads for the pit lane. Strategy play.",
        "Into the pits comes {car_id}. Let's see how long this stop takes.",
        "{car_id} pits. Could this be the move that changes the race?",
    ],
    EventType.PIT_EXIT: [
        "{car_id} rejoins on fresh rubber.",
        "{car_id} out of the pits — back in the fight.",
        "Clean stop for {car_id}, back out onto the circuit.",
        "{car_id} emerges from the pit lane — where does the strategy put them?",
    ],
    EventType.CLOSE_BATTLE: [
        "{car_a} right on the gearbox of {car_b} — only {gap_m:.0f}m between them!",
        "Intense battle between {car_a} and {car_b}. Nothing in it.",
        "{car_a} is all over {car_b}. This could happen any moment.",
        "DRS range! {car_a} closes right up to {car_b}.",
        "Side by side — {car_a} and {car_b}. Spectacular racing.",
        "{gap_m:.0f}m! {car_a} hunting {car_b} through every corner.",
    ],
    EventType.CRASH: [
        "Incident on track! Safety car may be needed.",
        "Big moment! A car is in trouble — yellow flags expected.",
        "There has been a coming together. Marshals on alert.",
        "Crash! That is a hefty impact into the barriers.",
        "Safety car likely — debris on the circuit.",
    ],
    EventType.YELLOW_FLAG: [
        "Yellow flags. Slow down — no overtaking in this sector.",
        "Caution — yellows are out. Cars must back off.",
        "Yellow flag situation. Hold your position.",
        "Marshals waving yellows. Something has happened ahead.",
    ],
    EventType.RACE_START: [
        "LIGHTS OUT AND AWAY WE GO!",
        "Five lights go out — and it's a clean start!",
        "They're racing! The field gets away.",
        "GO GO GO! The race is underway!",
    ],
}

PENALTY_TEMPLATES: dict[PenaltyType, list[str]] = {
    PenaltyType.TRACK_LIMIT_VIOLATION: [
        "{car_id} may have exceeded track limits — stewards will check.",
        "All four wheels over the white line for {car_id}. That'll be noted.",
        "Track limits at that corner for {car_id}. Could cost them lap time.",
        "Stewards looking at {car_id} — possible track limits infringement.",
    ],
    PenaltyType.PUSHING_OFF_TRACK: [
        "{car_id} squeezes {pushed_car} wide — that looks like a forcing move.",
        "Controversial! {car_id} pushes {pushed_car} beyond the limits.",
        "{car_id} under investigation — no room left for {pushed_car}.",
        "Stewards will look at that — {car_id} forces {pushed_car} off.",
    ],
    PenaltyType.UNFAIR_OVERTAKE: [
        "{car_id} made that move under yellow flags — place may be handed back.",
        "That pass by {car_id} looks suspicious. Flags were out.",
        "{car_id} overtakes off-track — stewards investigating.",
        "Was that legal? {car_id} under review.",
    ],
    PenaltyType.WEAVING: [
        "{car_id} changes direction more than once. Stewards watching.",
        "More than one move to defend from {car_id} — investigation possible.",
        "{car_id} weaving on the straight. The stewards won't like that.",
    ],
    PenaltyType.PIT_LANE_SPEEDING: [
        "{car_id} too fast in the pit lane. A penalty looks certain.",
        "Pit lane speed limit breached by {car_id}. Stewards alerted.",
        "{car_id} will be worried — that looked quick through the pit lane.",
    ],
}


class CommentaryGenerator:
    """
    Converts RaceEvent and PenaltyEvent lists to (frame_start, frame_end, text) tuples.
    """

    def __init__(self, hold_frames: int = DEFAULT_HOLD_FRAMES, seed: int = 42):
        self._rng        = random.Random(seed)
        self.hold_frames = hold_frames

    def generate(
        self,
        race_events:    list[RaceEvent],
        penalty_events: list[PenaltyEvent] | None = None,
        leaderboard:    dict | None = None,
    ) -> list[tuple[int, int, str]]:
        """
        Returns sorted list of (frame_start, frame_end, text) triples.
        """
        lines = []
        for ev in race_events:
            text = self._render_race(ev, leaderboard or {})
            if text:
                lines.append((ev.frame_idx, ev.frame_idx + self.hold_frames, text))

        for ev in (penalty_events or []):
            text = self._render_penalty(ev)
            if text:
                lines.append((ev.frame_idx, ev.frame_idx + self.hold_frames, text))

        lines.sort(key=lambda x: x[0])
        return lines

    def active_at(
        self,
        lines:     list[tuple[int, int, str]],
        frame_idx: int,
    ) -> str | None:
        """Return the most recent active commentary line at frame_idx, or None."""
        active = [text for (f0, f1, text) in lines if f0 <= frame_idx <= f1]
        return active[-1] if active else None

    # ── Render helpers ────────────────────────────────────────────────────────

    def _render_race(self, ev: RaceEvent, lb: dict) -> str | None:
        templates = RACE_TEMPLATES.get(ev.type)
        if not templates:
            return None
        tmpl = self._rng.choice(templates)
        ctx  = dict(ev.details)
        ctx["year"] = "2024"

        if ev.type == EventType.OVERTAKE:
            car_a = ev.car_ids[0] if ev.car_ids else 0
            car_b = ev.car_ids[1] if len(ev.car_ids) > 1 else 0
            ctx["overtaker"] = self._resolve(ctx.get("overtaker", car_a), lb, ev.frame_idx)
            ctx["overtaken"] = self._resolve(ctx.get("overtaken", car_b), lb, ev.frame_idx)
            # Remove position from context so templates with P{new_pos_a} are avoided
            pos = ctx.get("new_pos_a")
            ctx["new_pos_a"] = str(pos) if pos and str(pos) not in ("", "?", "???", "None") else ""

        elif ev.type in (EventType.PIT_ENTRY, EventType.PIT_EXIT, EventType.CRASH):
            ctx["car_id"] = self._resolve(ev.car_ids[0] if ev.car_ids else "?", lb, ev.frame_idx)

        elif ev.type == EventType.CLOSE_BATTLE:
            ctx["car_a"] = self._resolve(ev.car_ids[0], lb, ev.frame_idx)
            ctx["car_b"] = self._resolve(ev.car_ids[1], lb, ev.frame_idx)
            ctx.setdefault("gap_m", 0.0)

        try:
            return tmpl.format(**ctx)
        except KeyError as e:
            # Fill any missing keys with empty string instead of failing
            import re
            keys = re.findall(r'\{(\w+)[^}]*\}', tmpl)
            for k in keys:
                ctx.setdefault(k, "")
            try:
                return tmpl.format(**ctx)
            except Exception:
                return None

    def _render_penalty(self, ev: PenaltyEvent) -> str | None:
        templates = PENALTY_TEMPLATES.get(ev.type)
        if not templates:
            return None
        tmpl = self._rng.choice(templates)
        ctx  = dict(ev.details)
        ctx["car_id"] = str(ev.car_id)
        ctx.setdefault("pushed_car", "?")
        try:
            return tmpl.format(**ctx)
        except KeyError:
            return None

    @staticmethod
    def _resolve(car_id: int | str, lb: dict, frame_idx: int) -> str:
        """Return car as #ID format. Uses leaderboard position number if available."""
        try:
            if lb:
                nearest = min(lb.keys(), key=lambda fi: abs(fi - frame_idx), default=None)
                if nearest is not None:
                    entry = lb[nearest]
                    pos_to_num = entry.position_to_number() if hasattr(entry, "position_to_number") else {}
                    num = pos_to_num.get(int(car_id))
                    if num:
                        return f"#{num}"
        except (ValueError, TypeError):
            pass
        return f"#{car_id}"
