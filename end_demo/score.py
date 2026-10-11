"""The scoreboard, and the rules about how a number may be printed.

Two of these are not formatting preferences; they are the difference between a result and a
claim.

**Every number carries its N.** `pct` is never printed on its own anywhere in this file.
Forty trials on one seed is a sighting, not a measurement, and a bare percentage invites a
reader to forget which it was.

**Always split one-of-a-kind targets from identical ones.** Words score 100% on a thing that
has a name and about 33% on one of four identical bottles, so a blended figure hides the
entire argument and happens to flatter the baseline we are beating. The split is not an
appendix; it is the finding.

And one that is about honesty rather than arithmetic: **a call that never reached the model
is recorded and not scored.** The robot says it cannot - it does not fall back on a word
list - so a network failure is not a wrong answer and must not be counted as one.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from end_demo.session import Attempt


@dataclass
class Tally:
    """The scoreboard for one round."""

    round: str
    n: int = 0
    correct: int = 0
    words: int = 0
    questions: int = 0
    invalid: int = 0
    failed: int = 0
    ambiguous: int = 0
    contradictions: int = 0
    unclear: int = 0
    silences: int = 0
    misled: int = 0
    gave_up: int = 0
    out_of_steps: int = 0
    calls: int = 0
    seconds: float = 0.0
    latencies: list[float] = field(default_factory=list)

    def add(self, a: Attempt) -> None:
        self.n += 1
        self.correct += int(a.ok)
        self.words += a.words
        self.questions += a.questions
        self.invalid += int(a.invalid)
        self.failed += int(bool(a.failed))
        self.ambiguous += int(a.ambiguous)
        self.contradictions += a.contradictions
        self.unclear += a.unclear
        self.silences += a.silences
        self.misled += a.misled
        self.gave_up += int(a.gave_up)
        self.out_of_steps += int(a.out_of_steps)
        self.calls += a.model_calls
        self.seconds += a.model_s
        self.latencies += [float(c.get("latency_s", 0.0)) for c in a.calls]

    @property
    def scored(self) -> int:
        """Attempts that actually got an answer out of the model."""
        return self.n - self.failed

    @property
    def pct(self) -> float:
        return 100.0 * self.correct / self.scored if self.scored else 0.0

    @property
    def mean_words(self) -> float:
        return self.words / self.n if self.n else 0.0

    @property
    def mean_questions(self) -> float:
        return self.questions / self.n if self.n else 0.0

    @property
    def mean_call_s(self) -> float:
        return sum(self.latencies) / len(self.latencies) if self.latencies else 0.0

    @property
    def worst_call_s(self) -> float:
        return max(self.latencies) if self.latencies else 0.0

    def score(self) -> str:
        """The score with its N attached. There is no way to get it without."""
        return f"{self.correct}/{self.scored} = {self.pct:.0f}%"


def tally(attempts: list[Attempt]) -> list[Tally]:
    """Group attempts by round, keeping the order they were run in."""
    out: dict[str, Tally] = {}
    for a in attempts:
        out.setdefault(a.round, Tally(a.round)).add(a)
    return list(out.values())


def split(attempts: list[Attempt]) -> tuple[list[Attempt], list[Attempt]]:
    """One-of-a-kind targets, and targets that are one of several identical things."""
    return [a for a in attempts if a.unique], [a for a in attempts if not a.unique]


HEAD = "  {r:<{w}}  {right:>9}  {score:>6}  {words:>6}  {asks:>5}"


def _rows(attempts: list[Attempt], width: int) -> list[str]:
    out = []
    for r in tally(attempts):
        out.append(
            f"  {r.round:<{width}}  {r.correct:>4}/{r.scored:<4}  {r.pct:>5.0f}%  "
            f"{r.mean_words:>6.1f}  {r.mean_questions:>5.1f}"
        )
    return out


def scoreboard(attempts: list[Attempt]) -> list[str]:
    """The whole scoreboard, split, with N on every line."""
    rows = tally(attempts)
    if not rows:
        return ["", "SCOREBOARD", "", "  nothing ran"]
    width = max(len(r.round) for r in rows)
    head = HEAD.format(r="round", w=width, right="right", score="score", words="words", asks="asks")
    lines = ["", "SCOREBOARD", "", head]
    lines += _rows(attempts, width)
    unique, same = split(attempts)
    if unique and same:
        lines += ["", "  when the thing is one of a kind (a word for it exists):", head]
        lines += _rows(unique, width)
        lines += ["", "  when it is one of several identical things (no word exists):", head]
        lines += _rows(same, width)
    return lines


def notes(attempts: list[Attempt]) -> list[str]:
    """Everything that is not the score, and that a reader is owed anyway."""
    lines: list[str] = []
    for r in tally(attempts):
        bits = []
        if r.invalid:
            bits.append(f"{r.invalid} named something not on the shelf (scored wrong)")
        if r.failed:
            bits.append(f"{r.failed} could not reach the model (NOT scored)")
        if r.gave_up:
            bits.append(f"{r.gave_up} said it could not")
        if r.out_of_steps:
            bits.append(f"{r.out_of_steps} ran out of steps")
        if r.contradictions:
            bits.append(f"{r.contradictions} contradictions, none acted on (R5)")
        if r.unclear:
            bits.append(f"{r.unclear} points that settled nothing")
        if r.silences:
            bits.append(f"{r.silences} questions not answered")
        if r.misled:
            bits.append(f"{r.misled} yes/no answers the person gave wrongly")
        if bits:
            lines.append(f"  {r.round}: " + "; ".join(bits))
    return lines


def timing(attempts: list[Attempt]) -> list[str]:
    """Latency, mean and worst. The worst is the one that matters with a judge watching."""
    calls = [float(c.get("latency_s", 0.0)) for a in attempts for c in a.calls]
    firsts = [
        float(c.get("first_token_s", 0.0)) for a in attempts for c in a.calls if c.get("streamed")
    ]
    if not calls:
        return []
    out = [
        f"  model time    {sum(calls) / len(calls):.1f} s a decision on average, "
        f"{max(calls):.1f} s at worst (N={len(calls)} decisions)"
    ]
    if firsts:
        out.append(
            f"  first token   {sum(firsts) / len(firsts):.2f} s on average, "
            f"{max(firsts):.2f} s at worst (N={len(firsts)}) - the gate is under 1 s"
        )
    return out
