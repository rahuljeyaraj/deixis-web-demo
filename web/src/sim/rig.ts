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
  type Thing,
  THINGS,
  bay as bayOf,
  benchToWorld,
  onRail,
  ringFor,
  slideFor,
  slideOf,
  thing,
  worldToBench,
} from './bench';
import { ArmKinematics, type IkTarget } from './ik';
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
const CARRY_HEIGHTS_MM = [CARRY_Z_MM + 30, CARRY_Z_MM, CARRY_Z_MM - 30];

/**
 * How far a grip may be off where it was aimed before it counts as having slipped, m.
 *
 * Not nothing, because a thing is meant to move a little as it is gripped: the hand comes in
 * with the jaws open around it, and the moving jaw then pushes it against the fixed one. Half
 * the clearance, about 9 mm on the things here. 18 leaves room for that and still catches a
 * grip that closed on nothing.
 */
const SLIP_M = 0.018;

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
  /** Taken level, from under a board; otherwise taken from above. */
  level: boolean;
  /** The height of its bottom above its own centre when it was gripped, m. */
  below: number;
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

/** The jaws' heading for a level grip: across the way the hand is pointing. */
function acrossOf(point: Vec3): number {
  return Math.atan2(point[1], point[0]) + Math.PI / 2;
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

  /** Can this thing be picked up at all, whatever stands in the way? A reason, or null. */
  whyNot(id: string): Refusal | null {
    const t = THINGS.find((x) => x.id === id);
    if (!t) return 'there is nothing there';
    if (!t.graspable) return 'that is never picked up';
    if (t.gripMm > GRIP_WIDEST_MM) return 'that is too wide for the gripper';
    if (t.slideTo === null) return 'that is out of reach from anywhere on the rail';
    return null;
  }

  // ------------------------------------------------------------------ planning

  private planTake(id: string): Segment[] | Refusal {
    const no = this.whyNot(id);
    if (no) return no;
    if (this.held) return 'it is already holding something'; // one hand, one thing
    const t = thing(id);
    const at = this.state(id);
    const park = onRail(t.slideTo ?? this.slideMm);
    this.ik.rideTo(slideOf(park));
    const level = t.gripHow === 'level';
    const point = level ? this.bearingTo(at.centre) : DOWN;
    // Where the hand has to be for the jaws to be round the thing.
    //
    // Level, that is the height the ring for it was swept at, which is a height above the
    // bench and not above the board it stands on - 55, 177 and 299 are the middles of the
    // three bays' openings. Down, it is a little above the thing's own bottom, because the
    // pads stand above the fingertips and the fingertips come in near the bench.
    const below = (t.at[2] - t.rises[0]) / 1000;
    const zM = level ? t.gripZ / 1000 : at.centre[2] - below + GRASP_M;
    const grip: Vec3 = [at.centre[0], at.centre[1], zM];
    const plan = this.wayIn(
      grip,
      point,
      this.openings(t, at.yaw, level, point),
      level,
      t.bay !== null,
      gripMiddle(t.gripMm + OPEN_CLEAR_MM),
    );
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
    const over = this.toCarry(slide, this.rest.slice(1, 6), wide);
    if (!over) return 'there is no way in to it';
    const reach = this.seg([slide, ...open], over[0].to, 0.8);
    reach.onEnd = () => this.touchGripper(id, false);
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
    const up = this.toCarry(slide, q[0], hold);
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
    const h = this.held;
    if (!h) return 'nothing is being held';
    const t = h.thing;
    const level = h.level;
    const bay = this.bayAt(place);
    if (bay !== null && t.rises[1] - t.rises[0] > bayOf(bay)[1] - bayOf(bay)[0] - 4) {
      return 'it is too tall for that bay';
    }
    // Where to stand to put it there: searched along the rail against the measured ring, not
    // straight in front of the place. The spot a thing is handed over at is 130 mm from the
    // rail and the arm cannot work closer in than 135, so square in front of it is the one
    // place the arm certainly cannot do it from.
    const how = level ? 'level' : 'down';
    const z = level ? t.gripZ : 35;
    const found = slideFor(place.xMm, place.yMm, how, z);
    if (found === null) return 'there is no way to put it there';
    const park = onRail(found);
    this.ik.rideTo(slideOf(park));
    const slide = slideOf(park);
    // Where the thing's **centre** should end up, and so where the hand has to be: worked out
    // from the real grip rather than from where the thing was taken, because the two differ.
    //
    // A thing's centre is not the middle of it. The bottle of oil is 148 mm tall and its body
    // is 96, so its centre sits 48 mm up, not 74: putting it down by half its height would
    // bury a quarter of it in the bench.
    const [X, Y] = benchToWorld(place.xMm, place.yMm);
    const want: Vec3 = [X, Y, place.zMm / 1000 + h.below + 0.001];
    const point = level ? this.bearingTo(want) : DOWN;
    const aim = this.handFor(want, h, point, level);
    if (!aim) return 'there is no way to put it there';
    // Nothing to shift: a held thing's place in the jaws is already known, and `handFor` has
    // put the hand wherever makes it land in the right place.
    const plan = this.wayIn(aim, point, level ? [acrossOf(point)] : [null], level, bay !== null, 0);
    if (!plan) return 'there is no way to put it there';
    const q = plan;
    const at = q[q.length - 1];
    // The same wait before letting go, for the same reason: a thing dropped from 20 mm up
    // because the arm had not arrived is a thing put down somewhere else.
    const arrive = this.seg([slide, ...at, GRIPPER_SHUT], at, 0.5);
    const settle = this.seg([slide, ...at, GRIPPER_SHUT], at, 0.2);
    settle.onEnd = () => this.release(h);
    const wide = jawsAt(t.gripMm + OPEN_CLEAR_MM);
    const letGo = this.seg([slide, ...at, wide], at, 0.45);
    const out = this.path(slide, [...q].reverse(), wide, 0.7);
    if (out.length) out[out.length - 1].onEnd = () => this.touchGripper(t.id, true);
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
    openings: (number | null)[],
    level: boolean,
    inBay: boolean,
    middle: number,
  ): number[][] | null {
    this.why = '';
    for (const off of level ? BACK_OFF_M : [ABOVE_M]) {
      const q = this.straightIn(grip, point, openings, level, off, inBay, middle);
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
    openings: (number | null)[],
    level: boolean,
    off: number,
    inBay: boolean,
    middle: number,
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
      let from: number[] | undefined;
      // Which way the jaws will be opening is only known once a pose is solved, and where the
      // hand has to be depends on it - so the grip pose is solved at the thing's own middle
      // first, and the whole way in is then shifted along the direction that comes back.
      let shift: Vec3 = [0, 0, 0];
      if (middle) {
        const look = this.ik.solve(
          { pos: way[0].pos, point, yaw: opening, yawTurns: 2, maxTiltDeg: atTheGrip },
          from,
        );
        if (!close(look, atTheGrip)) continue;
        const opens = this.ik.fk(look.q).open;
        shift = [-middle * opens[0], -middle * opens[1], -middle * opens[2]];
        from = look.q;
      }
      for (const { pos: raw, tilt } of way) {
        const pos: Vec3 = [raw[0] + shift[0], raw[1] + shift[1], raw[2] + shift[2]];
        const target: IkTarget = {
          pos,
          point,
          yaw: opening,
          // A pair of jaws opens both ways at once, so either way round is the same grip.
          yawTurns: 2,
          maxTiltDeg: tilt,
        };
        const got = this.ik.solve(target, from);
        const folded = close(got, tilt) && this.ik.selfHits(got.q, jawsAt(90));
        if (!close(got, tilt) || folded) {
          this.why =
            `${off * 1000}mm back, jaws ${opening === null ? 'free' : `${Math.round((opening * 180) / Math.PI)}deg`}:` +
            ` stopped at pose ${q.length + 1} of ${way.length}` +
            (folded
              ? ' - the arm folds into itself'
              : ` - ${got.posErrMm.toFixed(1)}mm out, ${got.angErrDeg.toFixed(0)} degrees off` +
                ` (${tilt.toFixed(0)} allowed), jaws ${got.yawErrDeg.toFixed(0)} degrees round`);
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
  private openings(t: Thing, yaw: number, level: boolean, point: Vec3): (number | null)[] {
    if (level) return [acrossOf(point)];
    // The bench frame's x is the world's -Y, so a thing's own long axis, turned by its yaw,
    // reads as this heading in the world.
    const long = -yaw + (t.room[0] >= t.room[1] ? 0 : Math.PI / 2);
    return [long + Math.PI / 2, long, null];
  }

  /**
   * Where the hand has to be for a held thing's centre to land on `want`.
   *
   * Solved round, not computed: the thing sits somewhere in the gripper's frame and the hand
   * has to be put wherever that makes the thing land in the right place, so the pose is guessed
   * and corrected three times over.
   */
  private handFor(want: Vec3, h: Held, point: Vec3, level: boolean): Vec3 | null {
    let hand: Vec3 = [...want];
    let q: number[] | undefined;
    for (let k = 0; k < 3; k++) {
      const got = this.ik.solve(
        {
          pos: hand,
          point,
          yaw: level ? acrossOf(point) : null,
          yawTurns: 2,
          maxTiltDeg: TILT_AT_THE_GRIP_DEG,
        },
        q,
      );
      if (!close(got, TILT_AT_THE_GRIP_DEG)) return null;
      q = got.q;
      const pose = this.ik.fk(q);
      const is = this.fromSite(pose, h.local);
      hand = [hand[0] + (want[0] - (pose.pos[0] + is[0])), hand[1] + (want[1] - (pose.pos[1] + is[1])), hand[2] + (want[2] - (pose.pos[2] + is[2]))];
    }
    return hand;
  }

  /** Up out of the way, so the carriage can run without sweeping what is held along the bench. */
  private toCarry(slide: number, from: number[], grip: number): Segment[] | null {
    for (const z of CARRY_HEIGHTS_MM) {
      const pos: Vec3 = [CARRY_Y_MM / 1000, this.ik.base[1], z / 1000];
      const got = this.ik.solve(
        { pos, point: CARRY_POINT, yaw: null, maxTiltDeg: CARRY_TILT_DEG },
        from,
      );
      if (close(got, CARRY_TILT_DEG) && !this.ik.selfHits(got.q, GRIPPER_SHUT)) {
        return [this.seg([slide, ...got.q, grip], from, 0.8)];
      }
      this.why =
        `no way to hold it up at ${z} mm: ${got.posErrMm.toFixed(1)}mm out,` +
        ` ${got.angErrDeg.toFixed(0)} degrees off (${CARRY_TILT_DEG} allowed)`;
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

  /** Which bay a place is in, or null for the bench top. */
  private bayAt(place: Place): number | null {
    if (place.yMm < 350) return null;
    for (let n = 0; n < 3; n++) if (Math.abs(place.zMm - bayOf(n)[0]) < 1) return n;
    return null;
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
    const centre = this.state(t.id).centre;
    const local = this.inSite(this.livePose(), centre);
    const off = Math.hypot(local[0] - expected[0], local[1] - expected[1], local[2] - expected[2]);
    if (off > SLIP_M) {
      this.why =
        `the thing ended ${(off * 1000).toFixed(0)} mm from where the jaws were aimed` +
        ` (${local.map((v, i) => ((v - expected[i]) * 1000).toFixed(0)).join(', ')} mm along the` +
        ` fingers, across them and along the opening)`;
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
    this.held = { thing: t, body, local, point, level, below };
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
