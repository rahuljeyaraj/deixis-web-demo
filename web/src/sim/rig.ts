/**
 * The body (`workbench-requirements.md` stage 2, FR15, FR16): one arm on a rail that really
 * picks things up, carries them, puts them down, drops them into the pot, and refuses in words
 * when it cannot.
 *
 * This is the workbench's answer to `arm.ts`, which does the same job for the cubes demo. It is
 * a second class rather than a parameter on the first because almost everything differs: seven
 * things to drive instead of six, two ways of coming at a thing instead of one, twenty-one
 * things of their own shapes instead of three identical cubes, and a carriage that has to be
 * somewhere before the arm can reach at all. What **is** shared is kept shared: the solver
 * (`ik.ts`), the grip assist's weld, and the rule that nothing moves until every pose in a plan
 * has been solved.
 *
 * ## The two ways in
 *
 * **Down** is for a thing on the bench with nothing over it: the hand comes straight down, the
 * fingers pointing at the bench. **Level** is the only way into a bay, because a bay has a
 * board over it - the hand comes in horizontally, fingers towards the shelf, and backs straight
 * out again. Stage 0 swept the reach ring separately for each (`out/stage0-bay-ring.txt`), and
 * which one a thing needs is written beside it in `workbench/scene.py`.
 *
 * A grip never **rolls** the hand: a thing gripped upright stays upright, through the carry and
 * the put-down, because the jaws close on its sides either way. Turning a fingers-down grip
 * into a fingers-level carry would lay a jar of screws on its side.
 *
 * ## Why the carriage moves first, and alone
 *
 * The arm reaches 470 mm at best and the bench is 700 wide, so where the carriage stands is
 * part of every plan. It slides to the pick, the arm works; it slides to the place with the
 * thing held up out of the way, the arm works again. Sliding and reaching never happen at once:
 * a hand swinging while the carriage runs is hard to watch and harder to reason about when it
 * knocks something over.
 */

import {
  BENCH,
  SHELF,
  type Thing,
  THINGS,
  bay as bayOf,
  bayAt,
  benchToWorld,
  onRail,
  ringFor,
  ringNear,
  slideFor,
  slideOf,
  thing,
  worldToBench,
} from './bench';
import { ArmKinematics, type IkResult, type IkTarget } from './ik';
import type { MjData, MjModel, Mujoco } from './mujoco';
import type { Vec3 } from './table';

/** The six arm controls, in the order so101.xml declares them; the slide comes before them. */
const ARM_CTRL = [
  'shoulder_pan',
  'shoulder_lift',
  'elbow_flex',
  'wrist_flex',
  'wrist_roll',
  'gripper',
] as const;

/**
 * The gripper joint, and what it means in millimetres.
 *
 * The jaws run from shut to **107 mm tip to tip** at the joint's own limit of 1.745 radians
 * (`out/stage0-grip-width.txt`), near enough in proportion over that range. The demo needs the
 * number rather than a pair of constants because the things are not all one size: a tube of
 * glue is 30 mm across and a mug is 76, and jaws driven fully shut on a 64 mm jar do not grip
 * it - they shove it out of the way and then report that it slipped.
 */
export const JAWS_WIDEST_MM = 107;
export const JAWS_LIMIT_RAD = 1.745;
/**
 * And the other end of that line: the jaws are **not shut at nought**. The cubes demo found
 * them touching a 30 mm block at 0.18 (`arm.ts`), and `out/stage2-jaws.txt` shows the same from
 * the model - at a joint of nought the two jaws still stand well apart.
 *
 * A line through those two measured points is apart = 21 + 49.3 x joint. Reading it instead as
 * a line from nothing is what made every grip in this world miss: told to close on a 64 mm jar
 * it stopped at a joint of 0.93, where the jaws are 67 mm apart, and closed on air.
 */
export const JAWS_SHUT_MM = 21;
export const GRIPPER_OPEN = 1.6; // wide, for coming at anything in this world
export const GRIPPER_SHUT = 0.0;

/** The joint value whose jaws stand `mm` apart. */
export function jawsAt(mm: number): number {
  const per = (JAWS_WIDEST_MM - JAWS_SHUT_MM) / JAWS_LIMIT_RAD;
  return Math.max(0, Math.min(JAWS_LIMIT_RAD, (mm - JAWS_SHUT_MM) / per));
}

/**
 * Where the middle of jaws standing `mm` apart is, from the site, along the way they open (m).
 *
 * **The jaws are not symmetric about the site.** One of them is fixed and the other swings, so
 * the middle of the opening slides as they open: the fixed jaw's face sits 11.9 mm to one side
 * of the site - solved from the cubes demo's figure of 3.1 mm for a 30 mm block (`arm.ts`) -
 * and the middle is half the opening past that. Open 82 mm for a jar, that is 29 mm off the
 * site, and aiming the site at the jar's own middle instead put the fixed jaw 3 mm from its
 * side. Any error at all then clipped it, and the jar was found 22 mm further into its bay than
 * it started.
 */
export function gripMiddle(mm: number): number {
  return -0.0119 + mm / 2000;
}

/**
 * How much wider than the thing the jaws open to come at it, and how much narrower than it
 * they close to hold it, millimetres.
 *
 * 18 is not generous and it is not meant to be: the two tubes of glue stand 50 mm apart and are
 * 30 mm across, so jaws opening to 48 reach within 5 mm of the tube next door. Opening to the
 * joint's limit every time would knock the neighbour over on the way in.
 */
const OPEN_CLEAR_MM = 18;
const SQUEEZE_MM = 7;

/**
 * The widest thing the jaws hold through a lift and a move, mm. Measured on 24 Oct by
 * `tests/probes/grip-width.probe.test.ts`: 96 with the weld, **76** on the jaws and friction
 * alone, and 76 is the figure used because it does not depend on the grip assist.
 */
export const GRIP_WIDEST_MM = 76;

/** Where the fingers point for a grip from above: at the bench. A level grip works it out. */
const DOWN: Vec3 = [0, 0, -1];
/** And where they point while carrying: level, straight at the shelf. */
const CARRY_POINT: Vec3 = [1, 0, 0];
/** Up, in the world. */
const UP: Vec3 = [0, 0, 1];

/**
 * How far back from the person the bench top is worth offering, mm.
 *
 * Nearer than this is the rail trough and the carriage that runs in it. The layout keeps the
 * loose things clear of it for the same reason.
 */
const BENCH_FROM_Y_MM = 75;

/**
 * How much of a thing's side the jaws need to close on inside a bay, mm.
 *
 * The pads stand about 20 mm above and below the fingers and the hand can only come in level,
 * so a thing lying flatter than this on a board has no way in at all: the lower pad would meet
 * the board before the jaws met the thing. `workbench/scene.py` reasons the same figure out
 * from the jaw geometry and calls it `BAY_TALLEST_FLAT_MM`.
 *
 * It is a gate on **taking** a thing out of a bay, not on putting one in. A screwdriver can be
 * put away on its side and then cannot be fetched back by this gripper, which is a true answer
 * about this hand rather than a reason to refuse to tidy up.
 */
const BAY_GRIPPABLE_SIDE_MM = 50;

/**
 * How far out the hand starts and ends a level approach, m: the first of these that works.
 *
 * 100 mm is the one to want - it is clear of the shelf, so the hand is out of the bay before it
 * turns or lifts. It is not always to be had. The bottle of oil is gripped 353 mm from where
 * the arm stands, and backing 100 mm straight out along that line puts the hand 253 mm out,
 * which is **inside** the dead circle the ring measured at that height (260 mm). Backing off
 * less is the honest answer there, not refusing to pick the bottle up.
 */
const BACK_OFF_M = [0.1, 0.07, 0.05];
/**
 * How far the fingers may tilt off the way they should point: at the grip, and at the far end
 * of a way in from above, degrees.
 *
 * At the grip it has to be small, or the jaws meet a thing's side at an angle and slide off it.
 * Nine centimetres above the bench it cannot be: the wrist runs out of travel and the hand ends
 * up 7 to 10 degrees off level, which is what refused five things on the bench the first time
 * this ran. A **level** way in keeps the small figure the whole way, because there is a board
 * just above the hand and a tilted hand would meet it.
 */
const TILT_AT_THE_GRIP_DEG = 8;
/**
 * And for a level grip on something standing in the open, degrees. A bay has a board 20 mm over
 * the hand and a tilt would meet it; the bottle of oil on the bench has nothing above it, and
 * at 353 mm out the hand cannot hold itself to six degrees - it was refused by exactly that.
 */
const TILT_IN_THE_OPEN_DEG = 14;
const TILT_WAY_OUT_DEG = 30;
/**
 * And at the far end of a **level** way in, degrees. The hand is clear of the shelf by then -
 * a bay's front is 35 mm in front of the row, so 100 mm back is outside it - but anywhere
 * nearer in it has a board just overhead and has to stay flat.
 */
const TILT_CLEAR_OF_THE_BAY_DEG = 14;
/** How far above a thing the hand starts and ends a downward approach, m. */
const ABOVE_M = 0.09;
/** Poses on each straight way in or out. */
const STEPS = 3;
/**
 * How far above a thing's own bottom the site sits for a grip from above, m.
 *
 * The site is about 3 mm above the fingertips, and the pads stand above the fingertips again,
 * so 7 mm puts the pads on the side of anything taller than about 15 mm and keeps the tips off
 * the bench. It is the figure the cubes demo settled on for a 30 mm block and it carries over.
 */
const GRASP_M = 0.007;
/** The fastest the largest joint moves, rad/s, and the carriage, m/s. */
const SPEED = 1.6;
const SLIDE_SPEED = 0.35;

/**
 * How high the hand holds a thing while the carriage runs, mm above the bench, and how far
 * forward of the rail.
 *
 * 240 is measured against the world rather than chosen: the tallest thing standing on the
 * bench is the bottle of oil at 148 mm, and a thing held at 240 hangs its lowest point about
 * 190 mm up, which clears it. Any lower and the carriage would sweep whatever it is holding
 * through the bench.
 */
const CARRY_Z_MM = 240;
const CARRY_Y_MM = 170;
/**
 * How far the hand may tilt while carrying, degrees, and the heights to try.
 *
 * **Everything is carried with the hand level**, whichever way it was picked up, and that is
 * measured rather than chosen (`out/stage2-carry.txt`). Asked to hold a thing with the fingers
 * still pointing at the bench, the arm cannot: at 160 mm up it is 24 degrees off, at 220 it is
 * 54, and above that there is no pose at all. It is the same finding stage 0 made about
 * reaching down - it dies above 100 mm - met from the other side.
 *
 * Level, the same sweep comes back within 4 to 6 degrees anywhere from 140 to 260 mm out at
 * 250 to 280 mm up. So a thing taken from above is turned on its side as it comes up, which is
 * what a person does with a screwdriver, and nothing picked up from above is open at the top -
 * the one thing in this world that is, the pot of screws, is never picked up at all.
 */
const CARRY_TILT_DEG = 10;
/**
 * And how far it may tilt while carrying something **upright**, degrees.
 *
 * Wider, and it has to be. Holding a thing upright asks the hinge to stand on end, which with
 * the place is all five joints: what is left over is the hand's heading, and the fingers end up
 * pointing wherever that leaves them rather than straight at the shelf. Where the fingers point
 * while a jar is being carried matters to nothing - the jar is upright either way, which is the
 * thing that matters - so the figure here is slack on purpose.
 */
const CARRY_TILT_HINGED_DEG = 45;
/**
 * And how far it may tilt going up and over with an **empty** hand, degrees.
 *
 * Wider again, because there is nothing to hold level: the pose is only "up and out of the way
 * while the carriage runs", and 10 degrees - the figure for carrying something - refused it
 * outright at 11 degrees off and lost the grip that came after it.
 */
const OVER_TILT_DEG = 25;
/**
 * The heights to try holding a thing up at, in the order they are tried.
 *
 * 300 is in the list because 270, 240 and 210 are all over the bench where the tools lie, and
 * with the world check in, a tube of glue could not be held up at any of them: the forearm was
 * 31 mm into something each time. Higher is clear, and `out/stage2-carry.txt` found the hand
 * holds itself within 4 to 6 degrees of level anywhere from 140 to 260 mm out at 250 to 280 up,
 * so the top of the band is no worse a place to hold something than the middle of it.
 */
const CARRY_HEIGHTS_MM = [CARRY_Z_MM + 30, CARRY_Z_MM + 60, CARRY_Z_MM, CARRY_Z_MM - 30];
/** And how far forward of the rail, in the order they are tried: the swept band, 140 to 260. */
const CARRY_OUT_MM = [CARRY_Y_MM, 230, 140, 260];

/**
 * How far a grip may be off where it was aimed before it counts as having slipped, m.
 *
 * Not nothing, because a thing is meant to move a little as it is gripped: the hand comes in
 * with the jaws open around it, and the moving jaw then pushes it against the fixed one. Half
 * the clearance, about 9 mm on the things here.
 *
 * **25, measured, not 18.** A thing is let through the jaws on the way in rather than shoved by
 * them, and what the hand does instead is carry it a little further along its own line: every
 * jar and the box of odds and ends came out 17 mm deeper into its bay than the jaws were aimed,
 * which 18 called a slip. What catches a grip that closed on nothing is `JAWS_ROUND_IT_M`
 * below, which asks where the jaws are **now** against where the thing is **now** and has no
 * plan in it at all; this one only asks whether the thing stayed put.
 */
const SLIP_M = 0.025;

/**
 * How far a thing's middle may be from the middle of the jaws before they are not round it, m.
 *
 * A separate question from the slip, and a harsher one: the slip asks whether the thing stayed
 * where the plan aimed, which two stale numbers can agree on between themselves. This asks
 * where the jaws are now and where the thing is now. 25 mm is about a third of the widest
 * thing here, which is enough slack for a 76 mm mug held off its middle and not enough for a
 * hand that has closed somewhere else entirely.
 */
const JAWS_ROUND_IT_M = 0.025;

/**
 * How far the arm may graze the bench or the shelf, m, as against a thing, which it may not
 * touch at all.
 *
 * Measured, not chosen: `out/stage2-bay-height.txt` walks a level hand down each bay against
 * the boards themselves and finds the band where it fits is narrower than the opening and off
 * centre, and that a hand reaching under a board is within a few millimetres of it the whole
 * way. 10 mm is what lets a grip in all three bays through while still refusing the plans that
 * put a forearm 18 to 33 mm through a board or across the bench.
 */
const GRAZE_THE_SHELF_M = 0.01;

/**
 * And how far it may come into a **thing** before that counts as knocking it over, m.
 *
 * Tighter than the shelf by half, and not nothing. These are bounding shapes meeting bounding
 * shapes in a scratch copy of the world, and a 3 mm overlap between the back of the wrist and
 * the label on a bottle is not a bottle going over - while 22 mm of forearm through an offcut
 * of wood is. 5 mm is a nudge; past it, the plan is refused before anything moves, which is
 * what FR15 asks for and what "the hand knocks out all the things on its way" asked for.
 */
const NUDGE_A_THING_M = 0.005;

/** How much of a held thing's footprint to forgive when asking whether a place is free, mm. */
const ELBOW_ROOM_MM = 6;

export type Refusal =
  | 'busy'
  | 'stopped'
  | 'there is nothing there'
  | 'that is never picked up'
  | 'that is too wide for the gripper'
  | 'that is out of reach from anywhere on the rail'
  | 'there is no way in to it'
  | 'it slipped'
  | 'it is already holding something'
  | 'nothing is being held'
  | 'there is no way to put it there'
  | 'there is something there already'
  | 'it would knock something over on the way'
  | 'it is too tall for that bay';

export interface Done {
  ok: boolean;
  /** Why not, in words the model can use (FR15). */
  why?: Refusal;
  /** True when the weld took it before both jaws were touching. */
  assisted?: boolean;
}

/** Where a thing is to go: its centre across and away, and the height its bottom rests at. */
export interface Place {
  xMm: number;
  yMm: number;
  zMm: number;
}

export interface ThingState {
  id: string;
  /** Its centre, world metres. */
  centre: Vec3;
  /** Its centre in the bench frame, mm, and how high its bottom sits. */
  atMm: Vec3;
  yaw: number;
}

interface Held {
  thing: Thing;
  body: number;
  /** Its centre in the gripper site's frame when the weld took it. */
  local: Vec3;
  /** Where the fingers pointed when it was gripped: it is carried and put down the same way. */
  point: Vec3;
  /**
   * And which way the jaws opened, in the world. A thing is put back down the same way round
   * it was picked up, which a heading cannot say and this can.
   */
  open: Vec3;
  /**
   * And which way the hinge pointed. A level grip holds a thing with its own up along the
   * hinge, so this says which end of the hinge is the thing's own top - and a put-down that
   * stood the hinge the other way up would put the thing down upside down.
   */
  hinge: Vec3;
  /** Taken level, from under a board; otherwise taken from above. */
  level: boolean;
  /** How far its lowest point is below its own centre, m. */
  below: number;
  /** How wide the part the jaws closed on is, m: its height if it is ever laid on its side. */
  heldWide: number;
  /** How the thing itself was lying on the bench when the jaws took it, radians. */
  yawAtGrip: number;
  /** And which way round the jaws were then: the two turn together from here on. */
  headingAtGrip: number;
}

interface Segment {
  to: number[]; // 7 controls: the slide, then the arm's six
  duration: number;
  start?: number;
  from?: number[];
  onEnd?: () => void;
}

interface SitePose {
  pos: Vec3;
  point: Vec3;
  open: Vec3;
}

function minJerk(s: number): number {
  const t = Math.min(1, Math.max(0, s));
  return t * t * t * (10 - 15 * t + 6 * t * t);
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

function quatMul(a: number[], b: number[]): number[] {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

/**
 * Is this pose near enough to grip from?
 *
 * The solver's own `ok` wants the jaws within 2 degrees of the heading asked for, and that is
 * stricter than a grip needs: two tins of varnish and a mug were refused for a jaw heading two
 * degrees out, on a 72 mm tin the jaws close on from both sides. The position and the way the
 * fingers point are what matter, and they are held to the solver's own figures.
 */
function close(got: { posErrMm: number; angErrDeg: number; yawErrDeg: number }, tilt: number): boolean {
  return got.posErrMm < 1 && got.angErrDeg < Math.max(2, tilt) && got.yawErrDeg < 6;
}

/**
 * And is it near enough when the hinge was asked for rather than the fingers?
 *
 * **The fingers' direction is not judged at all here, and that is not a loosening.** A vertical
 * hinge puts the fingers level by itself: the hand's three axes are square to each other, so
 * fingers pointing out of level would tip the hinge off vertical by the same amount. What the
 * hinge leaves free is the hand's **heading** - which way round the bench the fingers face -
 * and that is what `angErrDeg` measures against the direction asked for. A level hand reaching
 * into a bay can be 20 degrees off head-on and still have both jaws on a jar; it is the board
 * above and below that is unforgiving, and the hinge is what keeps the hand clear of them.
 */
function hinged(got: { posErrMm: number; yawErrDeg: number }): boolean {
  return got.posErrMm < 1 && got.yawErrDeg < 6;
}

/** A thing's footprint on the bench: where its middle is, how big it is, and its turn. */
interface Footprint {
  xMm: number;
  yMm: number;
  acrossMm: number;
  deepMm: number;
  turn: number;
}

/**
 * Where a thing's footprint lies if its own middle is put at `xMm, yMm`.
 *
 * `worldYaw` is what the simulation holds, which is the bench frame's turn plus the quarter turn
 * between the two frames (`BENCH_TO_WORLD_QUAT` in the generator), so the quarter turn comes
 * back off here - `room` and `box` are both written in the thing's own frame, which is the
 * bench's. A round thing's footprint does not turn at all: turning the square round a jar makes
 * its corners reach 41 per cent further out than the jar ever does.
 */
function footprint(t: Thing, xMm: number, yMm: number, worldYaw: number, lessMm = 0): Footprint {
  const turn = t.round ? 0 : worldYaw + Math.PI / 2;
  const [mx, my] = t.round ? [0, 0] : t.box;
  return {
    xMm: xMm + Math.cos(turn) * mx - Math.sin(turn) * my,
    yMm: yMm + Math.sin(turn) * mx + Math.cos(turn) * my,
    acrossMm: Math.max(1, t.room[0] - lessMm),
    deepMm: Math.max(1, t.room[1] - lessMm),
    turn,
  };
}

/** Do two footprints cover any of the same bench? A separating-axis test on two rectangles. */
function overlap(a: Footprint, b: Footprint): boolean {
  const axes: [number, number][] = [
    [Math.cos(a.turn), Math.sin(a.turn)],
    [-Math.sin(a.turn), Math.cos(a.turn)],
    [Math.cos(b.turn), Math.sin(b.turn)],
    [-Math.sin(b.turn), Math.cos(b.turn)],
  ];
  const dx = b.xMm - a.xMm;
  const dy = b.yMm - a.yMm;
  const reach = (f: Footprint, ax: [number, number]) =>
    Math.abs((Math.cos(f.turn) * ax[0] + Math.sin(f.turn) * ax[1]) * f.acrossMm) / 2 +
    Math.abs((-Math.sin(f.turn) * ax[0] + Math.cos(f.turn) * ax[1]) * f.deepMm) / 2;
  for (const ax of axes) {
    if (Math.abs(dx * ax[0] + dy * ax[1]) > reach(a, ax) + reach(b, ax)) return false;
  }
  return true;
}

/** How a way round was asked for, for the report that says where a plan gave out. */
function describe(o: Opening): string {
  if (o.hinge) return `hinge ${o.hinge[2] >= 0 ? 'up' : 'down'}`;
  if (o.open) return `jaws ${Math.round((Math.atan2(o.open[1], o.open[0]) * 180) / Math.PI)}deg`;
  return o.yaw === null || o.yaw === undefined
    ? 'free'
    : `${Math.round((o.yaw * 180) / Math.PI)}deg`;
}

/** Down, in the world: the other way a jaw hinge can stand on end. */
const UPSIDE: Vec3 = [0, 0, -1];


/**
 * How far the hinge may lean off the vertical and the thing still be called upright, degrees.
 *
 * Not nothing: a level hand is allowed eight degrees of tilt at the grip and the hinge leans
 * with it, and a jar that leaves its bay eight degrees off upright arrives upright, because the
 * put-down is allowed the same eight degrees back. 25 is loose enough not to refuse a grip over
 * a few degrees and far from the 90 that lays a jar on its side.
 */
const UPRIGHT_ENOUGH_DEG = 25;

/** The jaws' heading for a level grip: across the way the hand points. */
function levelHeading(point: Vec3): number {
  return Math.atan2(point[1], point[0]) + Math.PI / 2;
}

/** Which way round the jaws face, as a heading: enough to put a thing back as it was. */
function headingOf(open: Vec3): number {
  return Math.atan2(open[1], open[0]);
}

/**
 * Which way round to try opening the jaws: a compass heading, or a whole direction.
 *
 * A heading settles a grip from above, where the fingers point straight down and nothing else
 * is free. Coming in level it does not, so a level way in asks for the whole direction.
 */
interface Opening {
  yaw?: number | null;
  open?: Vec3;
  /**
   * The hinge has to come out within this many degrees of standing on end, whatever else was
   * asked for. A check, not a constraint.
   *
   * There is a third way to ask for an upright grip, and it is the one that works best: pin the
   * jaws' **heading**, which the solver finds far easier than a hinge, and then throw out any
   * answer whose hinge has fallen over. A heading alone is not enough - jaws opening straight
   * up satisfy any heading at all, and the solver took exactly that, standing the jaws above
   * and below a jar instead of either side of it - but a heading plus this is.
   */
  uprightWithin?: number;
  /**
   * Which way the jaw hinge points: the one thing that says which way up a thing is held.
   *
   * A level grip asks for this and for nothing else about the hand's turn. The hinge and the
   * place are five things, which is what this arm has; adding the fingers' direction to them
   * makes six, and the jar that came up straight could then not be carried at all.
   */
  hinge?: Vec3;
}

/** How a thing standing where it is **now** has to be come at. */
interface Approach {
  /** In under a board, hand horizontal; otherwise down from above. */
  level: boolean;
  /**
   * Where the site has to be for the jaws to be round it: a height above the bench, mm.
   *
   * A starting figure for a level grip, not a fixed one - `planTake` walks it down until the
   * hand fits under the board above.
   */
  gripZMm: number;
  /** The nearest height a reach ring was really swept at, mm. */
  ringZ: number;
  /** Which bay it stands in now, or null for the bench top. */
  bay: number | null;
  /** Where the carriage has to stand to do it, bench mm, or null for nowhere on the rail. */
  slideTo: number | null;
}

function quatYaw(q: ArrayLike<number>, i: number): number {
  const [w, x, y, z] = [q[i], q[i + 1], q[i + 2], q[i + 3]];
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

export class Rig {
  readonly ik: ArmKinematics;
  /** The resting pose, read from the scene's own keyframe: the slide, then the arm's six. */
  readonly rest: number[];

  private readonly queue: Segment[] = [];
  private readonly ctrlAdr: number[];
  private readonly slideQpos: number;
  private readonly gripperBody: number;
  private readonly gripperBodies: Set<number>;
  private readonly gripperQpos: number;
  private readonly gripEq: number;
  private readonly noEquality: number;
  private readonly bodyOf = new Map<string, number>();
  private readonly geomsOf = new Map<string, number[]>();
  private readonly qposOf = new Map<string, number>();
  private held: Held | null = null;
  private holdGrip = GRIPPER_SHUT;
  /**
   * Where the last plan that could not be made gave out, for whoever is looking into it.
   *
   * "There is no way in to it" is the right thing to say to a person and a useless thing to
   * read in a report: a straight way in is four poses, and knowing which of the four, and by
   * how much, is the difference between a morning and a minute.
   */
  private why = '';
  /** How far the arm was from the pose it was sent to when the jaws closed, for the report. */
  private lag = '';
  /**
   * What a plan being worked out may pass through, and how wide the jaws are while it does.
   *
   * The gripper is let through the thing it is reaching for on purpose - the long comment in
   * `planTake` says why - so the world check has to be told about that one thing, and only
   * that one. Everything else in the world is in the way.
   */
  private passThrough: Set<number> = new Set();
  private jawsNow = GRIPPER_OPEN;
  /** Every geom belonging to a thing: what must not be touched at all, as against grazed. */
  private readonly thingGeoms = new Set<number>();
  private onIdle: ((d: Done) => void) | null = null;
  private result: Done = { ok: true };

  constructor(
    private readonly mj: Mujoco,
    private readonly model: MjModel,
    private readonly data: MjData,
  ) {
    const O = mj.mjtObj;
    const id = (type: number, name: string) => {
      const i = mj.mj_name2id(model, type, name);
      if (i < 0) throw new Error(`no ${name} in the workbench scene`);
      return i;
    };
    this.ik = new ArmKinematics(mj, model, 'arm_');
    this.ik.ridesOn('slide');
    this.ctrlAdr = [
      id(O.mjOBJ_ACTUATOR.value, 'slide'),
      ...ARM_CTRL.map((n) => id(O.mjOBJ_ACTUATOR.value, `arm_${n}`)),
    ];
    const qa = model.jnt_qposadr as Int32Array;
    this.slideQpos = qa[id(O.mjOBJ_JOINT.value, 'slide')];
    this.gripperQpos = qa[id(O.mjOBJ_JOINT.value, 'arm_gripper')];
    this.gripperBody = id(O.mjOBJ_BODY.value, 'arm_gripper');
    this.gripperBodies = new Set([
      this.gripperBody,
      id(O.mjOBJ_BODY.value, 'arm_moving_jaw_so101_v1'),
    ]);
    this.gripEq = id(O.mjOBJ_EQUALITY.value, 'grip');
    this.noEquality = mj.mjtDisableBit.mjDSBL_EQUALITY.value;
    this.setWeld(false);
    for (const t of THINGS) {
      this.bodyOf.set(t.id, id(O.mjOBJ_BODY.value, t.id));
      this.qposOf.set(t.id, qa[id(O.mjOBJ_JOINT.value, t.id)]);
    }
    const geomBody = model.geom_bodyid as Int32Array;
    for (let g = 0; g < model.ngeom; g++) {
      const name = mj.mj_id2name(model, O.mjOBJ_BODY.value, geomBody[g]);
      if (name && this.bodyOf.has(name)) this.geomsOf.set(name, [...(this.geomsOf.get(name) ?? []), g]);
    }
    for (const gs of this.geomsOf.values()) for (const g of gs) this.thingGeoms.add(g);
    const key = model.key_qpos as Float64Array;
    this.rest = [key[0], key[1], key[2], key[3], key[4], key[5], key[6]];
  }

  get busy(): boolean {
    return this.queue.length > 0;
  }

  /** Where the last plan that could not be made gave out. Empty when the last one was made. */
  get trouble(): string {
    return [this.why, this.lag].filter(Boolean).join('; ');
  }

  get holding(): string | null {
    return this.held?.thing.id ?? null;
  }

  /** Where the carriage is now, bench mm. */
  get slideMm(): number {
    return this.data.qpos[this.slideQpos] * 1000 + 350;
  }

  /** The five arm joints now. */
  armQ(): number[] {
    return this.ik.qposAdr.map((a) => this.data.qpos[a]);
  }

  /** Put the arm in its resting pose and hold it there, without moving through anything. */
  restNow(): void {
    this.data.qpos[this.slideQpos] = this.rest[0];
    for (let j = 0; j < 5; j++) this.data.qpos[this.ik.qposAdr[j]] = this.rest[j + 1];
    this.data.qpos[this.gripperQpos] = this.rest[6];
    this.setCtrl(this.rest);
    this.mj.mj_forward(this.model, this.data);
  }

  state(id: string): ThingState {
    const body = this.bodyOf.get(id);
    if (body === undefined) throw new Error(`no thing called ${id}`);
    const p = this.data.xpos as Float64Array;
    const centre: Vec3 = [p[3 * body], p[3 * body + 1], p[3 * body + 2]];
    const [x, y] = worldToBench(centre[0], centre[1]);
    return { id, centre, atMm: [x, y, centre[2] * 1000], yaw: quatYaw(this.data.xquat as Float64Array, 4 * body) };
  }

  /** Call before every physics step: follow the current segment, run its hook at the end. */
  update(): void {
    const t = this.data.time;
    const seg = this.queue[0];
    if (!seg) return;
    if (seg.start === undefined) {
      seg.start = t;
      seg.from = this.ctrlNow();
    }
    const s = minJerk((t - seg.start) / seg.duration);
    const v = seg.from!.map((f, i) => f + (seg.to[i] - f) * s);
    // A held thing no longer touches the jaws, so they are held on its faces, not shut.
    if (this.held && seg.to[6] === GRIPPER_SHUT) v[6] = this.holdGrip;
    this.setCtrl(v);
    if (t - seg.start >= seg.duration) {
      this.queue.shift();
      seg.onEnd?.();
      if (!this.queue.length) this.finish(this.result);
    }
  }

  /** Stop where it is, at once (FR27): hold every joint where it stands and drop the plan. */
  stop(): void {
    if (!this.busy) return;
    this.queue.length = 0;
    this.setCtrl([this.data.qpos[this.slideQpos], ...this.armQ(), this.data.qpos[this.gripperQpos]]);
    this.finish({ ok: false, why: 'stopped' });
  }

  /** Back to the resting pose at the end of the rail. Anything still held is let go first. */
  park(): Promise<Done> {
    if (this.busy) return Promise.resolve({ ok: false, why: 'busy' });
    this.result = { ok: true };
    const now = this.ctrlNow();
    if (this.held) {
      const letGo = this.seg([...now.slice(0, 6), GRIPPER_OPEN], now, 0.4);
      const held = this.held;
      letGo.onEnd = () => this.release(held);
      this.queue.push(letGo);
    }
    this.queue.push(this.seg([now[0], ...this.rest.slice(1)], now, 0.8), this.seg(this.rest, now, 0.9));
    return this.whenIdle();
  }

  /** Pick a thing up. Nothing moves unless every pose of the plan has been solved. */
  take(id: string): Promise<Done> {
    if (this.busy) return Promise.resolve({ ok: false, why: 'busy' });
    const plan = this.planTake(id);
    if (typeof plan === 'string') return Promise.resolve({ ok: false, why: plan });
    this.result = { ok: true };
    this.queue.push(...plan);
    return this.whenIdle();
  }

  /** Put down whatever is held, with its bottom resting at `place`. */
  putOn(place: Place): Promise<Done> {
    if (this.busy) return Promise.resolve({ ok: false, why: 'busy' });
    const plan = this.planPut(place);
    if (typeof plan === 'string') return Promise.resolve({ ok: false, why: plan });
    this.result = { ok: true };
    this.queue.push(...plan);
    return this.whenIdle();
  }

  /**
   * Let a held thing go over the rim of something open (FR16): the pot of screws.
   *
   * Not a put-down. The hand goes to a height above the rim measured from the pot itself and
   * opens, and the thing falls the rest of the way - which is what dropping something into a
   * pot is, and is why the pot is the one thing in the world that is never picked up.
   */
  dropInto(id: string): Promise<Done> {
    const into = THINGS.find((t) => t.id === id);
    if (!into) return Promise.resolve({ ok: false, why: 'there is nothing there' });
    return this.putOn({ xMm: into.at[0], yMm: into.at[1], zMm: into.rises[1] + 30 });
  }

  /**
   * Can this thing be picked up at all, whatever stands in the way? A reason, or null.
   *
   * Asked of where it **is**, not of where it started. Everything here used to come from
   * `workbench_reach.json`, which describes the world as it was laid out: a jar carried from
   * the bottom bay to the top one was still described as a bottom-bay jar, so the arm reached
   * for where it used to live and swept the real one along with its forearm (rahul, 24 Oct).
   */
  whyNot(id: string): Refusal | null {
    const t = THINGS.find((x) => x.id === id);
    if (!t) return 'there is nothing there';
    if (!t.graspable) return 'that is never picked up';
    if (t.gripMm > GRIP_WIDEST_MM) return 'that is too wide for the gripper';
    const how = this.approachTo(t, this.state(id));
    if (!how) return 'there is no way in to it';
    if (how.slideTo === null) return 'that is out of reach from anywhere on the rail';
    return null;
  }

  /**
   * Where the held thing can be put, and where it cannot: the reason, or null for yes.
   *
   * The cheap half of `planPut` - the bay it would stand in, whether it fits there, and
   * whether the arm can work at that spot from anywhere on the rail - without solving a single
   * pose. The page sweeps it over the bench and every bay to **show** where a held thing can
   * go, which is the answer to "wherever I tried to place a thing it keeps on saying there is
   * no way to put it there" (rahul, 24 Oct): a refusal nobody can see coming reads as a bug,
   * and the same refusal drawn on the bench reads as the arm's reach.
   */
  canPut(place: Place): Refusal | null {
    const h = this.held;
    if (!h) return 'nothing is being held';
    const put = this.putAs(h, place);
    if (put.bay !== null) {
      const [floor, ceiling] = bayOf(put.bay);
      // Room to be let go of, not merely room to exist: the jaws have to open and come back
      // out, so a thing that fills its bay to the last millimetre cannot be put in it.
      if (put.tall > ceiling - floor - 4) return 'it is too tall for that bay';
      // And room in the bay's depth. A thing goes in broadside - its long way across the bay,
      // its short way into it - so the short way is the one that has to fit.
      const deep = Math.min(h.thing.room[0], h.thing.room[1]);
      if (place.yMm - deep / 2 < SHELF.from[1] || place.yMm + deep / 2 > SHELF.to[1] - 10) {
        return 'there is no way to put it there';
      }
    }
    if (put.ringZ === null || put.slideTo === null) return 'there is no way to put it there';
    if (this.somethingAt(place, h, put.tall)) return 'there is something there already';
    return null;
  }

  /**
   * Where the held thing can be put down, swept over the bench and every bay: the places, in
   * bench millimetres, and how big a patch each one stands for.
   *
   * This is `canPut` turned into something a person can see. "Wherever i tried to place thing
   * it keeps on saying there is no way to put it there" (rahul, 24 Oct) is what a reach limit
   * feels like when it is invisible: a jar held upright can only be let go where the arm can
   * work level at 55 mm, which is the back of the bench and the bays, and nowhere near the
   * person. Drawn on the bench, the same limit reads as the arm's reach rather than as a bug.
   *
   * Swept rather than reasoned: every spot is asked the same question the arm will be asked.
   */
  wherePut(stepMm = 20): { xMm: number; yMm: number; zMm: number }[] {
    if (!this.held) return [];
    const out: { xMm: number; yMm: number; zMm: number }[] = [];
    const levels: [number, number, number][] = [
      // The bench top, from clear of the rail trough to the front of the shelf unit.
      [0, BENCH_FROM_Y_MM, SHELF.from[1] - stepMm / 2],
      // And inside each bay, between the uprights and in front of the back panel.
      ...([0, 1, 2].map((n) => [bayOf(n)[0], SHELF.from[1] + 6, SHELF.to[1] - 16]) as [
        number,
        number,
        number,
      ][]),
    ];
    for (const [zMm, fromY, toY] of levels) {
      for (let yMm = fromY; yMm <= toY; yMm += stepMm) {
        for (let xMm = 10; xMm <= BENCH.to[0] - 10; xMm += stepMm) {
          const place = { xMm, yMm, zMm };
          if (this.canPut(place) === null) out.push(place);
        }
      }
    }
    return out;
  }

  /**
   * Is something already standing where a held thing would go?
   *
   * Footprint against footprint, from where everything **is**. Both rectangles are the ones the
   * things really cover: carried round by their own turn, and offset from their middles, because
   * nothing here stands in the middle of its own footprint - a hammer's head is all at one end,
   * and a footprint drawn about the middle of a hammer covers 50 mm of bare bench and misses
   * 50 mm of hammer.
   *
   * **Rectangles, not circles.** A screwdriver 193 mm long read as a circle is a 96 mm disc
   * about its middle, which overlaps everything within four inches of it: that refused to put
   * four tools back on the spots they had just been lifted from.
   *
   * Heights are compared first, so a thing in the bottom bay is no reason not to put something
   * in the one above it.
   */
  private somethingAt(place: Place, h: Held, tallMm: number): boolean {
    // Where the held thing will be lying when it lands. A put-down from above asks the jaws for
    // the heading they took when they picked it up, so the thing goes back the way round it
    // was; a level put-down turns it with the arm, and the things taken level are round.
    // The held thing's own footprint, less a few millimetres all round. The layout stands
    // things 14 mm apart and the two jars in the bottom bay 70 mm from middle to middle with
    // 66 mm of jar between them, so a test that refuses any touch at all refuses the place a
    // jar has just been lifted out of. Three millimetres a side is the difference between
    // "there is something there" and "there is something beside it".
    const mine = footprint(h.thing, place.xMm, place.yMm, h.yawAtGrip, ELBOW_ROOM_MM);
    for (const t of THINGS) {
      if (t.id === h.thing.id) continue;
      const at = this.state(t.id);
      const lowMm = at.atMm[2] - (t.at[2] - t.rises[0]);
      if (lowMm + (t.rises[1] - t.rises[0]) <= place.zMm + 2) continue;
      if (lowMm >= place.zMm + tallMm - 2) continue;
      if (overlap(mine, footprint(t, at.atMm[0], at.atMm[1], at.yaw))) {
        this.why = `${t.what} is standing there`;
        return true;
      }
    }
    return false;
  }

  // ------------------------------------------------------------------ planning

  private planTake(id: string): Segment[] | Refusal {
    // Every refusal from here on says something of its own. A `why` left over from the last
    // plan reads as this one's reason and sends whoever is looking into it somewhere else: for
    // three things in `out/stage2-body.txt` it described a level grip on a thing taken from
    // above, which cannot happen.
    this.why = '';
    this.lag = '';
    const no = this.whyNot(id);
    if (no) return no;
    if (this.held) return 'it is already holding something'; // one hand, one thing
    const t = thing(id);
    const at = this.state(id);
    // Everything about the way in comes from where the thing **is**: which bay it is in now,
    // what height to grip it at, where to slide to. `whyNot` has already said there is one.
    const how = this.approachTo(t, at);
    if (!how || how.slideTo === null) {
      this.why = how
        ? 'nowhere on the rail to stand for it'
        : 'nothing was ever swept for a grip at that height, or there is too little of its side';
      return 'there is no way in to it';
    }
    const park = onRail(how.slideTo);
    this.ik.rideTo(slideOf(park));
    const level = how.level;
    const point = level ? this.bearingTo(at.centre) : DOWN;
    const below = (t.at[2] - t.rises[0]) / 1000;
    // The jaws may close through this one thing, and nothing else (see the long note below).
    this.passThrough = new Set(this.geomsOf.get(id) ?? []);
    this.jawsNow = jawsAt(t.gripMm + OPEN_CLEAR_MM);
    // **How high up its side to take it is searched, not fixed.**
    //
    // The three bay heights are the middles of the three openings, which is where stage 0 swept
    // the reach rings - and `out/stage2-bay-height.txt` shows the hand does not fit at any of
    // them. There is a band in each bay where it does, and it is not where the middle is: the
    // forearm meets the board above at the top of a bay and the pads meet the board below at
    // the bottom, and the band between them is narrower than the opening and sits off centre.
    // Reaching in at the middle anyway is the arm leaning on the shelf for the length of every
    // grip, which is half of "the hand knocks out all the things on its way" (rahul, 24 Oct).
    //
    // So heights are tried outwards from the middle of the thing's own side - where the jaws
    // have the most to hold on to - and the first whose whole way in is clear of the world is
    // the one used. The ring is re-asked at each, because the rings are measurements and not a
    // curve: 40 mm and 55 are both answered by the one swept at 55.
    let plan: number[][] | null = null;
    const bottomMm = at.atMm[2] - below * 1000;
    const topMm = bottomMm + (t.rises[1] - t.rises[0]);
    const band: [number, number] =
      how.bay === null
        ? [bottomMm + 22, topMm]
        : [bayOf(how.bay)[0] + 22, bayOf(how.bay)[1] - 24];
    const middleOfIt = Math.min(Math.max((bottomMm + topMm) / 2, band[0]), band[1]);
    const heights = level
      ? [0, 1, -1, 2, -2, 3, -3]
          .map((k) => middleOfIt + k * 12)
          .filter((z) => z >= band[0] && z <= band[1] && ringNear('level', z) !== null)
      : [how.gripZMm];
    for (const zMm of heights) {
      plan = this.wayIn(
        [at.centre[0], at.centre[1], zMm / 1000],
        point,
        // **Level, the jaws' heading is asked for and the hinge is checked.** A heading alone
        // leaves the hand free to roll - jaws opening straight up have whatever heading you
        // like, and the solver took exactly that, standing the jaws above and below a jar
        // instead of either side of it and putting the forearm through the board. A heading the
        // solver finds easy, plus a check that the hinge still stands on end, is both. The
        // hinge asked for outright is the fallback, for the few it cannot do the first way.
        //
        // From above, the heading is chosen by the thing's own long way.
        level
          ? [
              { yaw: levelHeading(point), uprightWithin: UPRIGHT_ENOUGH_DEG },
              { hinge: UP },
              { hinge: UPSIDE },
            ]
          : this.openings(t, at.yaw),
        level,
        how.bay !== null,
        gripMiddle(t.gripMm + OPEN_CLEAR_MM),
      );
      if (plan) {
        how.gripZMm = zMm;
        break;
      }
    }
    if (!plan) return 'there is no way in to it';
    const q = plan;
    const slide = slideOf(park);
    // Open wide enough to come at it, then close to a little narrower than it is, so the jaws
    // press on it rather than driving through where it stands.
    const wide = jawsAt(t.gripMm + OPEN_CLEAR_MM);
    const hold = jawsAt(t.gripMm - SQUEEZE_MM);
    const open = [...q[0], wide];
    // **The thing stops touching the gripper for the length of the way in.**
    //
    // This gripper has one fixed jaw and one that swings, so the middle of its grip slides as
    // it opens: 29 mm from the site open, 17 mm closed. A 64 mm jar cannot be both clear of the
    // fixed jaw on the way in and in the middle when the jaws meet - it is 7 mm too fat for
    // that, and `out/stage2-jaws.txt` shows what happens, which is that the jar is shoved
    // between 11 and 58 mm whichever way it is aimed.
    //
    // The grip assist is the plan's own answer to this (`workbench-requirements.md` 4.2a): the
    // jaws close and a weld carries the thing. Letting them close **through** where it stands
    // rather than against it keeps the thing exactly where the person pointed, and the weld
    // then holds it from where it really is. What this gives up is pretending the friction is
    // real, and the report says so for every grip: `assisted` is true whenever the weld took it
    // before both jaws were on it.
    // A moment at the grip before the jaws move.
    //
    // The arm is held by position motors, not teleported: reaching level into a bay it is
    // 373 mm out on a long lever, and when the last step of the way in ended it was still
    // 21 mm short of where it had been sent. The jaws then closed on the air in front of the
    // jar. Waiting half a second at the pose costs nothing to watch and lets the motors
    // arrive, and the grip check below is what says whether they did.
    const arrive = this.seg([slide, ...q[q.length - 1], wide], q[q.length - 1], 0.5);
    const close = this.seg([slide, ...q[q.length - 1], hold], q[q.length - 1], 0.6);
    const body = this.bodyOf.get(id)!;
    const expected = this.inSite(this.ik.fk(q[q.length - 1]), at.centre);
    // Up and over first, then down (or in). Sent straight from the resting pose to a pose 90 mm
    // above a screwdriver, the arm takes the shortest way through joint space, which runs
    // through the bench: it arrived 43 degrees and 242 mm from where it had been sent, because
    // it was leaning on the bench the whole way. Going by way of the pose it carries things in
    // keeps the hand above everything while it crosses.
    //
    // With an **empty** hand, so nothing is asked about which way up it is: there is nothing in
    // the jaws to be the wrong way up yet.
    const over = this.toCarry(slide, this.rest.slice(1, 6), wide, { yaw: null }, OVER_TILT_DEG);
    if (!over) return 'there is no way in to it';
    const reach = this.seg([slide, ...open], over[0].to, 0.8);
    // **The thing stops touching the gripper before the hand sets off towards it, not when it
    // arrives.** It used to be let through at the end of this move, which left the whole swing
    // from the carry pose to the mouth of the bay with the thing still solid - and that swing
    // is a straight line through joint space, not a line through the room, so it goes wherever
    // it likes. Every jar, every tube of glue and the box of odds and ends came out of their
    // grips pushed 17 to 48 mm **along the fingers**, which is the hand shoving them further
    // into the bay as it came. The window matters and nothing else is gained by keeping it:
    // this is the one thing the jaws are allowed through anyway.
    over[over.length - 1].onEnd = () => this.touchGripper(id, false);
    close.onEnd = () => {
      const want = q[q.length - 1];
      const now = this.armQ();
      const stray = Math.max(...want.map((v, j) => Math.abs(v - now[j])));
      const strayMm = Math.hypot(
        ...[0, 1, 2].map((k) => this.ik.fk(want).pos[k] - this.livePose().pos[k]),
      );
      this.lag = `the arm was ${((stray * 180) / Math.PI).toFixed(1)} degrees and ${(strayMm * 1000).toFixed(0)} mm from the pose it was sent to`;
      if (!this.grip(t, body, expected, point, level, below)) {
        this.touchGripper(id, true); // it is not held, so it is solid again
        this.result = { ok: false, why: 'it slipped' };
        this.queue.length = 1; // let the way out run, with nothing held
      }
    };
    const out = this.path(slide, [...q].reverse(), hold, 0.7);
    // Carried the way it was gripped: whichever end of the hinge the thing's own top went to,
    // that end stays up, or the jar would be carried upside down.
    const up = this.toCarry(
      slide,
      q[0],
      hold,
      level ? { hinge: this.uprightHinge() } : { open: UP },
    );
    if (!up) return 'there is no way in to it';
    return [
      this.seg([slide, ...this.rest.slice(1, 6), wide], this.ctrlNow(), 0.5),
      ...over,
      reach,
      ...this.path(slide, q, wide, 0.8),
      arrive,
      close,
      ...out,
      ...up,
    ];
  }

  private planPut(place: Place): Segment[] | Refusal {
    this.why = '';
    const h = this.held;
    if (!h) return 'nothing is being held';
    const no = this.canPut(place);
    if (no) return no;
    // **What the place asks for, not what the pick-up asked for.** This used to come down to
    // `h.level` - how the thing was taken - so a screwdriver taken off the bench tried to come
    // straight down into a bay that has a board over it, and a jar taken out of a bay tried to
    // come in level at the height the bay was swept at wherever it was going. Between them that
    // was "items from table i am not able to put in shelf and other way around" (rahul,
    // 24 Oct).
    const put = this.putAs(h, place);
    if (put.slideTo === null) return 'there is no way to put it there';
    const park = onRail(put.slideTo);
    this.ik.rideTo(slideOf(park));
    const slide = slideOf(park);
    // Where the thing's **centre** should end up, and so where the hand has to be: worked out
    // from the real grip rather than from where the thing was taken, because the two differ.
    //
    // A thing's centre is not the middle of it. The bottle of oil is 148 mm tall and its body
    // is 96, so its centre sits 48 mm up, not 74: putting it down by half its height would
    // bury a quarter of it in the bench.
    const [X, Y] = benchToWorld(place.xMm, place.yMm);
    const want: Vec3 = [X, Y, place.zMm / 1000 + put.below + 0.001];
    const point = put.level ? this.bearingTo(want) : DOWN;
    // And the way up it goes down is the way up it is being held (FR15, rahul 24 Oct: "it is
    // placed sideways when dropped"). A level hand holds a thing upright while its hinge stands
    // on end; a hand coming down puts a thing back the way round it picked it up, which
    // `yaw: null` never did - a screwdriver landed at whatever angle the solver liked.
    const how: Opening = put.level
      ? h.level
        ? { yaw: levelHeading(point), uprightWithin: UPRIGHT_ENOUGH_DEG }
        : { open: UP }
      : { yaw: headingOf(h.open) };
    const tilt = put.bay !== null ? TILT_CLEAR_OF_THE_BAY_DEG : TILT_IN_THE_OPEN_DEG;
    // The thing in the jaws is in the jaws; it is the rest of the world the hand must miss.
    this.passThrough = new Set(this.geomsOf.get(h.thing.id) ?? []);
    this.jawsNow = this.holdGrip;
    // Where to hold the hand, and the whole way in to it, worked out together until the two
    // agree about where the thing will land.
    //
    // `handFor` answers for one pose and `wayIn` is free to find another - and it does, because
    // what is pinned leaves the hand some freedom. Seeding the way in with the pose `handFor`
    // settled on keeps them on the same branch, and the landing is then checked against the way
    // in's **own** last pose rather than trusted: a thing held 40 mm off the site lands 40 mm
    // out for every ten degrees the two disagree by.
    let aimAt: Vec3 = [...want];
    let plan: number[][] | null = null;
    let landedOff = Infinity;
    // Each round starts from the pose the **plan** came out with, not from nothing. Without
    // that the two walk off to different branches each time round and the loop never settles:
    // the hand is worked out for one way round, the way in is solved for another, and the
    // landing comes out 40 to 55 mm away however many rounds it is given.
    let branch: number[] | undefined;
    for (let k = 0; k < 4 && landedOff > 0.003; k++) {
      const aim = this.handFor(aimAt, h, point, how, tilt, branch);
      if (!aim) return 'there is no way to put it there';
      plan = this.wayIn(
        aim.hand,
        point,
        put.level
          ? [how, ...(h.level ? [{ hinge: this.uprightHinge() }] : [])]
          : [how, { yaw: null }],
        put.level,
        put.bay !== null,
        0,
        aim.q,
      );
      if (!plan) return 'there is no way to put it there';
      branch = plan[plan.length - 1];
      const pose = this.ik.fk(branch);
      const lands = this.fromSite(pose, h.local).map((v, i) => pose.pos[i] + v) as Vec3;
      const slip: Vec3 = [want[0] - lands[0], want[1] - lands[1], want[2] - lands[2]];
      landedOff = Math.hypot(...slip);
      aimAt = [aimAt[0] + slip[0], aimAt[1] + slip[1], aimAt[2] + slip[2]];
    }
    if (!plan || landedOff > 0.008) {
      this.why = `the way in would land it ${(landedOff * 1000).toFixed(0)} mm from the place`;
      return 'there is no way to put it there';
    }
    const q = plan;
    const at = q[q.length - 1];
    // The same wait before letting go, for the same reason: a thing dropped from 20 mm up
    // because the arm had not arrived is a thing put down somewhere else.
    const arrive = this.seg([slide, ...at, GRIPPER_SHUT], at, 0.5);
    const settle = this.seg([slide, ...at, GRIPPER_SHUT], at, 0.2);
    settle.onEnd = () => this.release(h);
    const wide = jawsAt(h.thing.gripMm + OPEN_CLEAR_MM);
    const letGo = this.seg([slide, ...at, wide], at, 0.45);
    const out = this.path(slide, [...q].reverse(), wide, 0.7);
    if (out.length) out[out.length - 1].onEnd = () => this.touchGripper(h.thing.id, true);
    return [
      this.seg([slide, ...this.armQ(), GRIPPER_SHUT], this.ctrlNow(), 0.9),
      ...this.path(slide, q, GRIPPER_SHUT, 0.9),
      arrive,
      settle,
      letGo,
      ...out,
    ];
  }

  /**
   * How a thing standing where it is **now** has to be come at - or null for no way at all.
   *
   * Three things come out of the live place and nothing comes out of the layout: which bay the
   * thing is in, how high up its own side the jaws have to close, and where the carriage has to
   * stand. Those were all read from `workbench_reach.json` before, which says where every thing
   * **started**; a jar moved from the bottom bay to the top one was still a bottom-bay jar, so
   * the arm went to the bottom bay for it (rahul, 24 Oct).
   */
  private approachTo(t: Thing, at: ThingState): Approach | null {
    const below = (t.at[2] - t.rises[0]) / 1000;
    const bottomMm = at.atMm[2] - below * 1000;
    const bay = bayAt(at.atMm[0], at.atMm[1], bottomMm);
    const tallMm = t.rises[1] - t.rises[0];
    // Under a board there is one way in and it is level, whichever way the thing was first
    // picked up. In the open it is whatever its own shape asks for: a bottle of oil by its
    // neck, because the hand cannot hold itself over 148 mm of bottle; a screwdriver from
    // above, because it is 37 mm tall and has no side worth closing on.
    const level = bay !== null || t.gripHow === 'level';
    if (bay !== null && tallMm < BAY_GRIPPABLE_SIDE_MM) return null;
    const gripZMm = level ? bottomMm + (t.gripAbove ?? tallMm / 2) : bottomMm + GRASP_M * 1000;
    const ringZ = ringNear(level ? 'level' : 'down', gripZMm);
    if (ringZ === null) return null;
    return {
      level,
      gripZMm,
      ringZ,
      bay,
      // Searched along the rail against the measured ring, not straight in front of the place.
      slideTo: slideFor(at.atMm[0], at.atMm[1], level ? 'level' : 'down', ringZ),
    };
  }

  /**
   * How a held thing would be put down at a place: the way in, the way up, and where to stand.
   *
   * The cheap arithmetic behind both `canPut` and `planPut`, so that what the page draws and
   * what the arm does cannot disagree.
   */
  private putAs(
    h: Held,
    place: Place,
  ): {
    bay: number | null;
    level: boolean;
    below: number;
    tall: number;
    ringZ: number | null;
    slideTo: number | null;
  } {
    const bay = bayAt(place.xMm, place.yMm, place.zMm);
    // Into a bay the hand can only come in level - there is a board over it. On the bench it
    // goes down the way it came up, so the thing lands the way it was standing.
    const level = bay !== null || h.level;
    const sameWayUp = level === h.level;
    // A thing taken from above and put down level goes in **on its side**: the hand pitches a
    // quarter turn to carry it (nothing can be held with the fingers down at height -
    // `out/stage2-carry.txt`), which stands the jaws' own axis up. What was the width the jaws
    // closed on is then its height, and the middle of the jaws is the middle of that.
    const below = sameWayUp ? h.below : h.heldWide / 2;
    const tall = sameWayUp ? h.thing.rises[1] - h.thing.rises[0] : h.heldWide * 1000;
    const siteZMm = !level
      ? place.zMm + GRASP_M * 1000
      : place.zMm + (sameWayUp ? (h.thing.gripAbove ?? below * 1000) : (h.heldWide * 1000) / 2);
    const ringZ = ringNear(level ? 'level' : 'down', siteZMm);
    return {
      bay,
      level,
      below,
      tall,
      ringZ,
      // The spot a thing is handed over at is 130 mm from the rail and the arm cannot work
      // closer in than 135, so square in front of a place is sometimes the one place on the
      // rail it cannot be done from. Searched, like every other stand.
      slideTo:
        ringZ === null ? null : slideFor(place.xMm, place.yMm, level ? 'level' : 'down', ringZ),
    };
  }

  /**
   * Solve a pose that is **also clear of the world**, tried from more than one starting guess.
   *
   * A five-joint arm reaching into a bay has a choice of elbows, and the solver answers with
   * whichever one it walks downhill into from the pose it was given. One of them puts the
   * forearm through the board over the bay and another does not, and nothing used to ask: the
   * answer came back, the arm was driven to it, and it spent the grip leaning on the shelf
   * ("the hand knocks out all the things on its way", rahul 24 Oct).
   *
   * So the question is asked of several: the pose before this one on the way in, the solver's
   * own guess, and that guess with the shoulder and elbow swung either way. The first answer
   * that is both near enough and touching nothing is the one used. What comes back when none
   * of them is clear is the deepest offender, so the refusal can name it.
   */
  /** Does the hinge still stand on end in this pose, if the opening asked that it should? */
  private standsUp(q: number[], opening: Opening): boolean {
    if (!opening.uprightWithin) return true;
    const pose = this.ik.fk(q);
    const hinge = cross(pose.open, pose.point);
    const over = (Math.acos(Math.min(1, Math.abs(hinge[2]))) * 180) / Math.PI;
    return over <= opening.uprightWithin;
  }

  /**
   * Is any joint of this pose jammed against its own stop?
   *
   * A pose can be perfect on paper and unholdable in the world. The solver answers with
   * kinematics - where the joints would put the hand - and says nothing about whether position
   * motors can hold them there; a joint sitting on its limit has no travel left on one side, so
   * gravity takes the arm off the pose and the motors cannot bring it back. That is measured,
   * not supposed: the poses rejected here are the ones that left the arm 94 to 100 mm and
   * 18 to 20 degrees from where it had been sent, with the jaws closing on air beside the jar.
   *
   * Two degrees of room is enough to tell a pose that happens to end near a stop from one that
   * is being held up by it.
   */
  private atTheStops(q: number[]): boolean {
    const room = (2 * Math.PI) / 180;
    return q.some((v, j) => v < this.ik.lo[j] + room || v > this.ik.hi[j] - room);
  }

  /** What this pose would run into: a thing at a touch, the furniture at a graze. */
  private runsInto(q: number[]): { name: string; depthM: number } | null {
    return this.ik.worldHit(
      this.data,
      q,
      this.jawsNow,
      this.passThrough,
      GRAZE_THE_SHELF_M,
      this.thingGeoms,
      NUDGE_A_THING_M,
    );
  }

  private solveClear(
    target: IkTarget,
    from: number[] | undefined,
    near: (got: IkResult) => boolean,
  ): { got: IkResult; into: { name: string; depthM: number } | null } {
    const seeds: (number[] | undefined)[] = [from, undefined];
    const guess = this.ik.guess(target);
    for (const [lift, elbow] of [
      [-0.5, 0.5],
      [0.5, -0.5],
      [0, 0.7],
      [0, -0.7],
    ]) {
      seeds.push([guess[0], guess[1] + lift, guess[2] + elbow, guess[3], guess[4]]);
    }
    let fallback: { got: IkResult; into: { name: string; depthM: number } | null } | null = null;
    for (const seed of seeds) {
      const got = this.ik.solve(target, seed);
      if (!near(got)) {
        fallback ??= { got, into: null };
        continue;
      }
      const into = this.runsInto(got.q);
      if (!into) return { got, into: null };
      if (!fallback?.into || into.depthM < fallback.into.depthM) fallback = { got, into };
    }
    return fallback ?? { got: this.ik.solve(target, from), into: null };
  }

  /**
   * The straight way in to a grip, and the joint poses along it (outermost first).
   *
   * Level, the hand backs off towards the person and comes straight in; down, it starts above
   * and comes straight down. Either way every pose on the line is solved before any of it is
   * run, and the fingers keep pointing the same way the whole time - a hand that rolls on the
   * way in turns a jar on its side.
   */
  private wayIn(
    grip: Vec3,
    point: Vec3,
    openings: Opening[],
    level: boolean,
    inBay: boolean,
    middle: number,
    seed?: number[],
  ): number[][] | null {
    this.why = '';
    for (const off of level ? BACK_OFF_M : [ABOVE_M]) {
      const q = this.straightIn(grip, point, openings, level, off, inBay, middle, seed);
      if (q) {
        this.why = '';
        return q;
      }
    }
    return null;
  }

  /** One try at a straight way in, backing off by `off` metres. */
  private straightIn(
    grip: Vec3,
    point: Vec3,
    openings: Opening[],
    level: boolean,
    off: number,
    inBay: boolean,
    middle: number,
    seed?: number[],
  ): number[][] | null {
    const atTheGrip = inBay ? TILT_AT_THE_GRIP_DEG : TILT_IN_THE_OPEN_DEG;
    // Level, the hand backs straight out the way it came in; from above, straight up.
    const step = (k: number): Vec3 =>
      level
        ? [grip[0] - (off * k * point[0]) / STEPS, grip[1] - (off * k * point[1]) / STEPS, grip[2]]
        : [grip[0], grip[1], grip[2] + (off * k) / STEPS];
    // **From the grip outwards**, not from the far end in. The grip is the most constrained
    // pose of the four - it is the one with the fingers level inside a bay - so solving it
    // first and feeding it to the next as a starting guess keeps the whole way in near one
    // solution. Solved the other way round, the far end picks whatever pose it likes and the
    // ones after it are dragged out of reach from there.
    const way: { pos: Vec3; tilt: number }[] = [];
    for (let k = 0; k <= STEPS; k++) {
      const f = k / STEPS;
      const tilt = level
        ? k === STEPS
          ? Math.max(TILT_CLEAR_OF_THE_BAY_DEG, atTheGrip)
          : atTheGrip
        : atTheGrip + (TILT_WAY_OUT_DEG - atTheGrip) * f;
      way.push({ pos: step(k), tilt });
    }
    for (const opening of openings) {
      const q: number[][] = [];
      // **Started from the pose the caller already has, when it has one.** A hinged pose leaves
      // the hand's heading to the arm, so the same place can be solved two ways round; a
      // put-down worked out where to hold the hand from one of them and would then be driven to
      // the other, which swung the thing it was holding 75 mm across the bench.
      let from: number[] | undefined = seed;
      // Nothing is shifted here any more. The middle of the jaws is what gets aimed at the
      // thing (`alongOpenM` in `ik.ts`), so which way round they come out no longer changes
      // where the hand has to stand - and there is no side to guess at before solving.
      for (const { pos, tilt } of way) {
        const target: IkTarget = {
          pos,
          alongOpenM: middle,
          point,
          yaw: opening.yaw ?? null,
          open: opening.open,
          hinge: opening.hinge,
          // A pair of jaws opens both ways at once, so either way round is the same grip.
          yawTurns: 2,
          maxTiltDeg: tilt,
        };
        // **And what it would run into.** Checked at every pose on the way in, against the
        // world as it stands, before any of it is driven, and tried from several elbows.
        // **Joint stops are only asked about at the grip itself.** That is the pose the arm has
        // to hold still while the jaws close, and a joint with no travel left on one side
        // cannot: gravity takes the arm off it and the motors cannot bring it back, which left
        // the hand 94 mm from where it had been sent with the jaws closing on air beside the
        // jar. The far end of a way in is nothing of the kind - it is passed through on the way
        // out, and nine centimetres above the bench the wrist is **expected** to be at its
        // limit, which is written down where the tilt allowances are. Asking there refused
        // seven put-downs for a fault that was not one.
        const atTheGripPose = q.length === 0;
        const judge = (r: IkResult) =>
          (opening.hinge ? hinged(r) : close(r, tilt)) &&
          this.standsUp(r.q, opening) &&
          (!atTheGripPose || !this.atTheStops(r.q)) &&
          !this.ik.selfHits(r.q, jawsAt(90));
        const { got, into } = this.solveClear(target, from, judge);
        const near = opening.hinge ? hinged(got) : close(got, tilt);
        const folded = near && this.ik.selfHits(got.q, jawsAt(90));
        const rolled = near && !this.standsUp(got.q, opening);
        const jammed = near && !folded && !rolled && atTheGripPose && this.atTheStops(got.q);
        if (!near || folded || jammed || into) {
          this.why =
            `${off * 1000}mm back, jaws ${describe(opening)}:` +
            ` stopped at pose ${q.length + 1} of ${way.length}` +
            (folded
              ? ' - the arm folds into itself'
              : rolled
                ? ' - the jaws would be rolled over, one above the thing and one below'
                : jammed
                  ? ' - a joint would be hard against its own stop, which it cannot hold'
                  : into
                    ? ` - it would be ${(into.depthM * 1000).toFixed(0)} mm into ${into.name}`
                    : ` - ${got.posErrMm.toFixed(1)}mm out, ${got.angErrDeg.toFixed(0)} degrees` +
                      ` ${opening.hinge ? 'round from head-on' : `off (${tilt.toFixed(0)} allowed)`},` +
                      ` jaws ${got.yawErrDeg.toFixed(0)} degrees over`);
          break;
        }
        from = got.q;
        q.push(got.q);
      }
      if (q.length === way.length) return q.reverse(); // outermost first, as a way in is run
    }
    return null;
  }

  /**
   * Which ways round to try opening the jaws, best first.
   *
   * **Across the long way of the thing, not along it.** A screwdriver is 193 mm end to end and
   * 37 mm across; jaws that open along it have nothing to close on, and the gripper opens 107.
   * Which way is long comes from the counted footprint, turned by however the thing is lying.
   * Coming in level the choice is made already - the jaws open across the bench, because up and
   * down there is a board.
   */
  private openings(t: Thing, yaw: number): Opening[] {
    // The bench frame's x is the world's -Y, so a thing's own long axis, turned by its yaw,
    // reads as this heading in the world.
    const long = -yaw + (t.room[0] >= t.room[1] ? 0 : Math.PI / 2);
    return [{ yaw: long + Math.PI / 2 }, { yaw: long }, { yaw: null }];
  }

  /**
   * Where the hand has to be for a held thing's centre to land on `want`.
   *
   * Solved round, not computed: the thing sits somewhere in the gripper's frame and the hand
   * has to be put wherever that makes the thing land in the right place, so the pose is guessed
   * and corrected three times over.
   */
  private handFor(
    want: Vec3,
    h: Held,
    point: Vec3,
    how: Opening,
    tilt: number,
    seed?: number[],
  ): { hand: Vec3; q: number[] } | null {
    let hand: Vec3 = [...want];
    let q: number[] | undefined = seed;
    for (let k = 0; k < 3; k++) {
      const got = this.ik.solve(
        {
          pos: hand,
          point,
          yaw: how.yaw ?? null,
          open: how.open,
          hinge: how.hinge,
          yawTurns: 2,
          maxTiltDeg: tilt,
        },
        q,
      );
      if (!(how.hinge ? hinged(got) : close(got, tilt))) return null;
      q = got.q;
      const pose = this.ik.fk(q);
      const is = this.fromSite(pose, h.local);
      hand = [hand[0] + (want[0] - (pose.pos[0] + is[0])), hand[1] + (want[1] - (pose.pos[1] + is[1])), hand[2] + (want[2] - (pose.pos[2] + is[2]))];
    }
    return q ? { hand, q } : null;
  }

  /**
   * Which end of the jaw hinge the held thing's own top went to, as a direction to ask for.
   *
   * A pair of jaws opens both ways at once, so a level grip may come out with the hinge
   * standing either way up and both are the same grip. Which one it came out as decides which
   * way up the thing is in the hand, so every pose after the grip has to ask for the same one.
   */
  private uprightHinge(): Vec3 {
    return this.held && dot(this.held.hinge, UP) < 0 ? UPSIDE : UP;
  }

  /** Up out of the way, so the carriage can run without sweeping what is held along the bench. */
  private toCarry(
    slide: number,
    from: number[],
    grip: number,
    how: Opening,
    slackDeg = CARRY_TILT_DEG,
  ): Segment[] | null {
    // **How far forward to hold it is tried as well as how high.** Straight out at 170 mm is
    // the place to want, and it is not always to be had: the bottle of oil stands 86 mm from
    // the rail and 148 mm tall, so for a carriage anywhere near it the arm's own forearm is
    // 28 to 31 mm through the bottle at every height there is, and three things in the shelf
    // could not be held up at all. `out/stage2-carry.txt` swept 140 to 260 mm out and found the
    // hand holds itself within 4 to 6 degrees of level across all of it, so reaching out
    // further or drawing in nearer costs nothing but the asking.
    for (const [y, z] of CARRY_HEIGHTS_MM.flatMap((z) => CARRY_OUT_MM.map((y) => [y, z]))) {
      const pos: Vec3 = [y / 1000, this.ik.base[1], z / 1000];
      // **The roll is asked for, not left free.** With the jaws free to lie whichever way round
      // was easiest, the solver rolled the wrist and laid a jar on its side on the way up.
      const got = this.ik.solve(
        {
          pos,
          point: CARRY_POINT,
          yaw: how.yaw ?? null,
          open: how.open,
          hinge: how.hinge,
          yawTurns: 2,
          maxTiltDeg: how.hinge ? CARRY_TILT_HINGED_DEG : slackDeg,
        },
        from,
      );
      const slack = how.hinge ? CARRY_TILT_HINGED_DEG : slackDeg;
      const near = how.hinge ? hinged(got) : close(got, slack);
      const into = near ? this.runsInto(got.q) : null;
      if (near && !into && !this.ik.selfHits(got.q, GRIPPER_SHUT)) {
        return [this.seg([slide, ...got.q, grip], from, 0.8)];
      }
      if (into) {
        this.why =
          `holding it up ${y} mm out at ${z} mm would be` +
          ` ${(into.depthM * 1000).toFixed(0)} mm into ${into.name}`;
        continue;
      }
      this.why =
        `no way to hold it up ${y} mm out at ${z} mm: ${got.posErrMm.toFixed(1)}mm out,` +
        ` ${got.angErrDeg.toFixed(0)} degrees off (${slack} allowed),` +
        ` hinge ${got.yawErrDeg.toFixed(0)} degrees over`;
    }
    return null;
  }

  /**
   * Which way the hand points to come at something level: along the line from where the arm
   * stands to the thing, flat.
   *
   * Not simply "at the shelf". The carriage can always slide square in front of a bay, so for
   * anything on a shelf this comes out as straight at the shelf anyway - but the bottle of oil
   * stands 86 mm from the rail, far closer in than the arm can work, and the layout solved it
   * by standing the carriage 343 mm to one side. The hand has to come at it from there, 76
   * degrees round, and a way in that always pointed at the shelf refused it.
   */
  private bearingTo(at: Vec3): Vec3 {
    const dx = at[0] - this.ik.base[0];
    const dy = at[1] - this.ik.base[1];
    const r = Math.hypot(dx, dy) || 1;
    return [dx / r, dy / r, 0];
  }

  // ------------------------------------------------------------------ the grip

  private path(slide: number, qs: number[][], grip: number, duration: number): Segment[] {
    const out: Segment[] = [];
    for (let k = 1; k < qs.length; k++) {
      out.push(this.seg([slide, ...qs[k], grip], qs[k - 1], duration / (qs.length - 1)));
    }
    return out;
  }

  private seg(to: number[], from: number[], minDuration = 0.6): Segment {
    const arm = from.length === 7 ? from.slice(1, 6) : from.slice(0, 5);
    const turn = Math.max(...to.slice(1, 6).map((v, j) => Math.abs(v - arm[j])));
    const run = from.length === 7 ? Math.abs(to[0] - from[0]) : 0;
    return { to, duration: Math.max(minDuration, turn / SPEED, run / SLIDE_SPEED) };
  }

  private whenIdle(): Promise<Done> {
    return new Promise((resolve) => {
      this.onIdle = resolve;
    });
  }

  private finish(d: Done): void {
    const done = this.onIdle;
    this.onIdle = null;
    done?.(d);
  }

  private ctrlNow(): number[] {
    return this.ctrlAdr.map((a) => this.data.ctrl[a]);
  }

  private setCtrl(v: number[]): void {
    for (let i = 0; i < 7; i++) this.data.ctrl[this.ctrlAdr[i]] = v[i];
  }

  private livePose(): SitePose {
    const s = this.ik.siteId;
    const p = this.data.site_xpos as Float64Array;
    const R = this.data.site_xmat as Float64Array;
    return {
      pos: [p[3 * s], p[3 * s + 1], p[3 * s + 2]],
      point: [R[9 * s], R[9 * s + 3], R[9 * s + 6]],
      open: [R[9 * s + 2], R[9 * s + 5], R[9 * s + 8]],
    };
  }

  private inSite(pose: SitePose, p: Vec3): Vec3 {
    const y = cross(pose.open, pose.point);
    const d: Vec3 = [p[0] - pose.pos[0], p[1] - pose.pos[1], p[2] - pose.pos[2]];
    return [dot(d, pose.point), dot(d, y), dot(d, pose.open)];
  }

  private fromSite(pose: SitePose, l: Vec3): Vec3 {
    const y = cross(pose.open, pose.point);
    return [0, 1, 2].map((i) => l[0] * pose.point[i] + l[1] * y[i] + l[2] * pose.open[i]) as Vec3;
  }

  /** Both jaws touching it? */
  private jawsTouch(id: string): boolean {
    const mine = new Set(this.geomsOf.get(id) ?? []);
    const touched = new Set<number>();
    const bodyOf = this.model.geom_bodyid as Int32Array;
    const contacts = this.data.contact;
    try {
      for (let i = 0; i < this.data.ncon; i++) {
        const c = contacts.get(i);
        if (!c) continue;
        const other = mine.has(c.geom1) ? c.geom2 : mine.has(c.geom2) ? c.geom1 : -1;
        if (other >= 0 && this.gripperBodies.has(bodyOf[other])) touched.add(bodyOf[other]);
        c.delete();
      }
    } finally {
      contacts.delete();
    }
    return touched.size === 2;
  }

  /**
   * Hold a thing with the weld, where it is now relative to the gripper.
   *
   * The jaws really closed on it first; from here the weld carries it and it stops touching
   * the gripper, so no squeeze can push it while it is carried. If it is not where the jaws
   * should have it - more than 12 mm off - nothing is held and the move reports that it
   * slipped rather than carrying thin air.
   */
  private grip(
    t: Thing,
    body: number,
    expected: Vec3,
    point: Vec3,
    level: boolean,
    below: number,
  ): boolean {
    const pose = this.livePose();
    const centre = this.state(t.id).centre;
    const local = this.inSite(pose, centre);
    const off = Math.hypot(local[0] - expected[0], local[1] - expected[1], local[2] - expected[2]);
    if (off > SLIP_M) {
      this.why =
        `the thing ended ${(off * 1000).toFixed(0)} mm from where the jaws were aimed` +
        ` (${local.map((v, i) => ((v - expected[i]) * 1000).toFixed(0)).join(', ')} mm along the` +
        ` fingers, across them and along the opening)`;
      return false;
    }
    // **And a second check, against the jaws themselves.** The one above asks whether the thing
    // moved from where the plan aimed; this one asks whether the jaws are round it at all. They
    // are not the same question, and the difference is what let a jar be welded to a hand that
    // was nowhere near it: the plan said where the thing used to be, the thing was there, and
    // both agreed with each other and with nothing in the world (rahul, 24 Oct). This one has
    // no plan in it - it is where the site is **now** against where the thing is **now**.
    // Measured against the middle of the **open** jaws, which is where the plan aimed the thing
    // and where it is meant to still be: the jaws are let close through it rather than against
    // it, so it does not move to the middle of a closed grip and should not be asked to.
    const jaws = gripMiddle(t.gripMm + OPEN_CLEAR_MM);
    const sideways = Math.hypot(local[0], local[1]);
    const alongOpen = Math.abs(local[2] - jaws);
    if (sideways > JAWS_ROUND_IT_M || alongOpen > JAWS_ROUND_IT_M) {
      this.why =
        `the jaws are not round it: its middle is ${(sideways * 1000).toFixed(0)} mm off the` +
        ` jaws' line and ${(alongOpen * 1000).toFixed(0)} mm from the middle of the opening`;
      return false;
    }
    this.why = '';
    const touching = this.jawsTouch(t.id);
    const xp = this.data.xpos as Float64Array;
    const xq = this.data.xquat as Float64Array;
    const R = this.data.xmat as Float64Array;
    const g = this.gripperBody;
    const q1 = [xq[4 * g], xq[4 * g + 1], xq[4 * g + 2], xq[4 * g + 3]];
    const q2 = [xq[4 * body], xq[4 * body + 1], xq[4 * body + 2], xq[4 * body + 3]];
    const d = [0, 1, 2].map((k) => xp[3 * body + k] - xp[3 * g + k]);
    const rel = [0, 1, 2].map((c) => R[9 * g + c] * d[0] + R[9 * g + 3 + c] * d[1] + R[9 * g + 6 + c] * d[2]);
    const relq = quatMul([q1[0], -q1[1], -q1[2], -q1[3]], q2);
    const n = 11; // mjNEQDATA: anchor (3), relpose (7), torquescale (1)
    const e = this.model.eq_data as Float64Array;
    e.set([0, 0, 0, ...rel, ...relq, e[this.gripEq * n + 10]], this.gripEq * n);
    (this.model.eq_obj2id as Int32Array)[this.gripEq] = body;
    this.setWeld(true);
    this.touchGripper(t.id, false);
    this.holdGrip = this.data.qpos[this.gripperQpos];
    this.data.ctrl[this.ctrlAdr[6]] = this.holdGrip;
    this.held = {
      thing: t,
      body,
      local,
      point,
      open: pose.open,
      hinge: cross(pose.open, pose.point),
      level,
      below,
      heldWide: t.gripMm / 1000,
      yawAtGrip: this.state(t.id).yaw,
      headingAtGrip: Math.atan2(pose.open[1], pose.open[0]),
    };
    this.result = { ok: true, assisted: !touching };
    return true;
  }

  private release(h: Held): void {
    this.setWeld(false);
    if (this.held === h) this.held = null;
  }

  /** Contact groups: a thing is 3 (it touches the gripper's 1), a held thing 2 (it does not). */
  private touchGripper(id: string, on: boolean): void {
    const contype = this.model.geom_contype as Int32Array;
    const conaffinity = this.model.geom_conaffinity as Int32Array;
    for (const g of this.geomsOf.get(id) ?? []) {
      contype[g] = on ? 3 : 2;
      conaffinity[g] = on ? 3 : 2;
    }
  }

  private setWeld(on: boolean): void {
    const opt = this.model.opt;
    opt.disableflags = on ? opt.disableflags & ~this.noEquality : opt.disableflags | this.noEquality;
  }
}

/** Every ring stage 0 swept, so a caller can say why something is out of reach. */
export { ringFor };
