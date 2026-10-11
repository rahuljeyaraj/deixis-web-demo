"""What the robot can see, what it has done, and what it is not able to do.

Two ideas run through this module.

**The robot's eyes are the prompt.** Everything the model knows about the shelf comes from
`describe`, and it is written to be what a camera could honestly report: what each thing
looks like, whereabouts on the shelf it is, how much of the person's pointing cone it holds,
and - said plainly - that several of the things are identical and no word separates them.
Nothing in it hints at which one the person means.

**A failed action is told to the model, not fixed behind its back.** If the left arm is
asked for something it cannot reach, the answer is a sentence saying so, handed back as the
tool's result, and the model works out what to do (pass it across, or use the other arm).
That is the closed loop the robot-learning literature calls inner monologue, and it is the
difference between a demo that recovers and a demo that is quietly steered by its harness.
The temptation to silently substitute the other arm is exactly the bug that made part of the
three-rounds score the harness's rather than the model's.

**Which arm.** Passing `arm` is optional. When the model leaves it out, the robot uses an
arm that can do the job - kinematics, not judgement, and it changes no score, because
selection is about which *thing*. When the person says *you*, the model passes `arm`, and
then it means it: a request of the arm that cannot reach the person has to be passed across.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

from end_demo.point import Cone
from end_demo.scene import Arm, Thing, by_id, kind_counts, shelf
from end_demo.scene import arms as default_arms

#: Where a thing may be put. A **reference**, never coordinates: the core's standing
#: invariant is that the language model never writes a number for a place
#: (`deixis/refs.py`). "pointed" means wherever the person is pointing now.
Where = Literal["tray", "shelf", "pointed"]

ORDINALS = ("1st", "2nd", "3rd", "4th", "5th", "6th", "7th", "8th", "9th", "10th")


@dataclass(frozen=True)
class Outcome:
    """What happened, in the words the model will read."""

    ok: bool
    text: str
    #: Set when the outcome is the end of the job.
    finished: bool = False


@dataclass
class World:
    """The shelf, the arms, and everything that has happened to them."""

    things: list[Thing] = field(default_factory=shelf)
    arms: list[Arm] = field(default_factory=default_arms)
    #: arm id -> the thing it is holding.
    held: dict[str, str] = field(default_factory=dict)
    #: Things that are now in the person's hand. The job is done when the one they wanted
    #: is in here.
    with_person: list[str] = field(default_factory=list)
    in_tray: list[str] = field(default_factory=list)
    #: Things no longer on the shelf, because an arm took them off it.
    off_shelf: set[str] = field(default_factory=set)
    #: Every action tried, ok or not, in order. The transcript of the body.
    did: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- looking

    @property
    def index(self) -> dict[str, Thing]:
        return by_id(self.things)

    @property
    def arm_index(self) -> dict[str, Arm]:
        return {a.id: a for a in self.arms}

    def on_shelf(self) -> list[Thing]:
        return [t for t in self.things if t.id not in self.off_shelf]

    def arm_for(self, thing_id: str) -> Arm | None:
        """An arm that can pick this thing up, nearest first. None if neither can."""
        t = self.index.get(thing_id)
        if t is None:
            return None
        able = [a for a in self.arms if a.can_reach(t) and a.id not in self.held]
        if not able:
            return None
        return min(able, key=lambda a: abs(a.x - t.pos[0]))

    def _resolve_arm(self, arm: str | None, thing_id: str | None = None) -> tuple[Arm | None, str]:
        """The arm to use, or a sentence saying why there isn't one."""
        if arm:
            got = self.arm_index.get(arm)
            if got is None:
                return None, f"There is no arm called {arm!r}. The arms are: " + ", ".join(
                    a.id for a in self.arms
                ) + "."
            return got, ""
        if thing_id is None:
            free = [a for a in self.arms if a.id not in self.held]
            return (free[0] if free else self.arms[0]), ""
        got = self.arm_for(thing_id)
        if got is None:
            return None, self._out_of_reach(thing_id)
        return got, ""

    def _out_of_reach(self, thing_id: str) -> str:
        t = self.index.get(thing_id)
        if t is None:
            return f"There is nothing called {thing_id!r} on the shelf."
        busy = [a.id for a in self.arms if a.can_reach(t) and a.id in self.held]
        if busy:
            return (
                f"The only arm that can reach {thing_id} is {busy[0]}, and it is already "
                f"holding {self.held[busy[0]]}."
            )
        return f"Neither arm can reach {thing_id}."

    def place_of(self, t: Thing) -> str:
        row = "top shelf" if t.row == 0 else "lower shelf"
        return f"{row}, {ORDINALS[t.col]} from the left"

    def describe(self, cone: Cone | None = None, said: str = "") -> str:
        """The scene, as the robot's own camera and arms could report it.

        The cone column is the person's pointing likelihood, straight out of the resolver -
        the same numbers the page draws. A dash means the thing is outside the cone.
        """
        weights = {c.id: c.weight for c in (cone.candidates if cone else [])}
        header = "  thing       looks like                 where                            point"
        rows = ["THE SHELF - what I can see:", "", header]
        for t in self.on_shelf():
            w = weights.get(t.id)
            share = f"{w:.2f}" if w is not None else "  -  "
            rows.append(f"  {t.id:<11} {t.seen:<26} {self.place_of(t):<32} {share}")
        same = [f"{n} {k}s" for k, n in kind_counts(self.on_shelf()).items() if n > 1]
        if same:
            rows += [
                "",
                "  " + " and ".join(same) + " are identical. No word tells them apart, so "
                "when the person's",
                "  words fit more than one thing, their words are not the answer: where they "
                "pointed is.",
            ]
        rows += ["", "THE ARMS:"]
        for a in self.arms:
            cols = ", ".join(ORDINALS[c] for c in a.reach)
            reach = (
                "can put things in the person's hand"
                if a.reaches_person
                else ("CANNOT reach the person - it has to pass a thing to the other arm first")
            )
            holding = self.held.get(a.id)
            rows.append(f"  {a.id:<11} {a.seen}")
            rows.append(f"  {'':<11} picks up from: {cols} from the left; {reach}")
            if holding:
                rows.append(f"  {'':<11} holding {holding}")
        if self.with_person:
            rows += ["", "  the person is holding: " + ", ".join(self.with_person)]
        if self.in_tray:
            rows += ["", "  in the tray: " + ", ".join(self.in_tray)]
        if cone is not None:
            rows += [
                "",
                "The point column is how much of the person's pointing cone each thing holds "
                f"(their aim is off by about {cone.sigma_deg:.0f} degrees per axis, so "
                "neighbours overlap).",
            ]
        if said:
            rows += ["", f'The recogniser heard: "{said}". It mishears; the point can correct it.']
        return "\n".join(rows)

    # ---------------------------------------------------------------- doing

    def _log(self, text: str) -> None:
        self.did.append(text)

    def pick(self, thing_id: str, arm: str | None = None) -> Outcome:
        """Take a thing off the shelf."""
        t = self.index.get(thing_id)
        if t is None:
            return self._fail(f"There is nothing called {thing_id!r} on the shelf.")
        if thing_id in self.off_shelf:
            who = next((a for a, h in self.held.items() if h == thing_id), "")
            where = f"{who} is holding it" if who else "it is not on the shelf any more"
            return self._fail(f"{thing_id} is already off the shelf - {where}.")
        got, why = self._resolve_arm(arm, thing_id)
        if got is None:
            return self._fail(why)
        if got.id in self.held:
            return self._fail(f"{got.id} is already holding {self.held[got.id]}.")
        if not got.can_reach(t):
            cols = ", ".join(ORDINALS[c] for c in got.reach)
            other = next((a.id for a in self.arms if a.can_reach(t)), "")
            hint = f" {other} can reach it." if other else ""
            return self._fail(
                f"{got.id} cannot reach {thing_id} ({self.place_of(t)}). It only picks up "
                f"from: {cols} from the left.{hint}"
            )
        if not t.graspable:
            return self._fail(f"I cannot get a grip on {thing_id}.")
        self.held[got.id] = thing_id
        self.off_shelf.add(thing_id)
        self._log(f"{got.id} picked up {thing_id}")
        return self._ok(f"{got.id} is holding {thing_id}.")

    def pass_to(self, arm: str | None = None) -> Outcome:
        """Pass the held thing to the other arm. The only way a thing on the far left of
        the shelf reaches the person, because the left arm cannot reach them."""
        holders = [a for a in self.arms if a.id in self.held]
        if not holders:
            return self._fail("Neither arm is holding anything to pass.")
        taker, why = self._resolve_arm(arm)
        if taker is None:
            return self._fail(why)
        giver = next((a for a in holders if a.id != taker.id), None)
        if giver is None:
            return self._fail(
                f"{taker.id} is the arm that is holding something. Name the arm that should "
                "take it."
            )
        if taker.id in self.held:
            return self._fail(f"{taker.id} is already holding {self.held[taker.id]}.")
        thing_id = self.held.pop(giver.id)
        self.held[taker.id] = thing_id
        self._log(f"{giver.id} passed {thing_id} to {taker.id}")
        return self._ok(f"{giver.id} passed {thing_id} to {taker.id}. {taker.id} is holding it.")

    def hand_over(self, arm: str | None = None) -> Outcome:
        """Put the held thing into the person's hand. The job, most of the time."""
        holders = [a for a in self.arms if a.id in self.held]
        if not holders:
            return self._fail("Neither arm is holding anything to hand over.")
        got, why = (
            self._resolve_arm(arm)
            if arm
            else (
                next((a for a in holders if a.reaches_person), holders[0]),
                "",
            )
        )
        if got is None:
            return self._fail(why)
        if got.id not in self.held:
            return self._fail(f"{got.id} is not holding anything.")
        if not got.reaches_person:
            other = next((a.id for a in self.arms if a.reaches_person), "the other arm")
            return self._fail(
                f"{got.id} cannot reach the person. Pass {self.held[got.id]} to {other} "
                "first, then hand it over."
            )
        thing_id = self.held.pop(got.id)
        self.with_person.append(thing_id)
        self._log(f"{got.id} handed {thing_id} to the person")
        return self._ok(f"{got.id} put {thing_id} in the person's hand.", finished=True)

    def place(self, where: str, arm: str | None = None) -> Outcome:
        """Put the held thing down: in the tray, or back on the shelf."""
        holders = [a for a in self.arms if a.id in self.held]
        if not holders:
            return self._fail("Neither arm is holding anything to put down.")
        got, why = self._resolve_arm(arm) if arm else (holders[0], "")
        if got is None:
            return self._fail(why)
        if got.id not in self.held:
            return self._fail(f"{got.id} is not holding anything.")
        if where not in ("tray", "shelf"):
            return self._fail(
                f"I do not know where {where!r} is. I can put it in the tray, or back on the shelf."
            )
        thing_id = self.held.pop(got.id)
        if where == "tray":
            self.in_tray.append(thing_id)
            self._log(f"{got.id} put {thing_id} in the tray")
            return self._ok(f"{thing_id} is in the tray.")
        self.off_shelf.discard(thing_id)
        self._log(f"{got.id} put {thing_id} back on the shelf")
        return self._ok(f"{thing_id} is back on the shelf.")

    def point_at(self, thing_id: str, arm: str | None = None) -> Outcome:
        """Hold the gripper over a thing, without asking anything and without moving it.

        An arm can point at more of the shelf than it can pick up from: pointing is a lean,
        not a grasp. So this never fails for reach - which is right, because asking about a
        thing has to be cheaper than reaching for it.
        """
        if thing_id not in self.index:
            return self._fail(f"There is nothing called {thing_id!r} on the shelf.")
        got, why = self._resolve_arm(arm)
        if got is None:
            return self._fail(why)
        self._log(f"{got.id} pointed at {thing_id}")
        return self._ok(f"{got.id} is holding its gripper over {thing_id}.")

    # ---------------------------------------------------------------- bookkeeping

    def _ok(self, text: str, finished: bool = False) -> Outcome:
        return Outcome(True, text, finished)

    def _fail(self, text: str) -> Outcome:
        self._log(f"could not: {text}")
        return Outcome(False, text)

    def gave(self, thing_id: str) -> bool:
        return thing_id in self.with_person
