/**
 * The arm's moves (plan 6.2): scripted pick and place with smooth timing, the position
 * actuators following, and the grip assist.
 *
 * - Straight down and straight up: the way to a block and the way to a place are chains of
 *   inverse-kinematics poses above the spot, so the fingers don't swing into a block.
 * - Grip assist: the jaws really close on the block; then a weld holds it and it stops
 *   touching the gripper, so no squeeze can push it while it's carried or let go
 *   (table_scene.xml explains the weld switch).
 * - The place is solved again from the real grip, so the block lands where it was meant to.
 *
 * Heights are of the `gripperframe` site, about 3 mm above the fingertips. The jaws open along
 * the site's z axis; the fixed jaw is 20 mm to its -z side, so a block is gripped with its
 * centre about 3 mm to the +z side of the site (tests/probes/grip.probe.test.ts).
 */

import { ArmKinematics, type IkTarget } from './ik';
import type { MjData, MjModel, Mujoco } from './mujoco';
import { BLOCK_MM, type Vec3, tableToWorld, worldToTable } from './table';

export const BLOCK_IDS = ['red', 'yellow', 'green', 'blue'] as const;
export type BlockId = (typeof BLOCK_IDS)[number];

export const GRIPPER_OPEN = 0.6; // about 60 mm between the jaws
export const GRIPPER_SHUT = 0.0; // pushes on a 30 mm block (the jaws touch it near 0.18)
/** Arm up, the gripper high and facing the person, off the work area: 5 joints and gripper. */
export const HOME = [0, -1.45, 0.75, 1.45, 0, 0.2];

const Z_GRASP = 0.007; // fingertips about 4 mm above the table
const LIFTS = [0.07, 0.058, 0.048]; // the highest that works: 0.07 clears a 30 mm block by 3 cm
const STEPS = 4; // poses on each straight way up or down
const GRIP_OFFSET = 0.0031; // block centre from the site along the opening axis, m
const SPEED = 1.6; // rad/s for the largest joint in a move
const HALF = BLOCK_MM / 2000;

export type Reason = 'cant_reach' | 'busy' | 'no_block' | 'slipped' | 'stopped';

export interface Block {
  id: BlockId;
  body: number;
  geom: number;
  qposAdr: number;
  dofAdr: number;
  /** When it last moved (seconds on the page clock), for the core's "it". */
  movedAt: number;
}

export interface BlockState {
  id: BlockId;
  center: Vec3; // world, m
  yaw: number; // radians
  tableMm: [number, number]; // its centre in the table frame
  heightMm: number; // its top above the table
}

export interface Place {
  xMm: number;
  yMm: number;
  /** The height the block's bottom rests at: 0 on the table, 30 on another block. */
  zMm: number;
}

interface Segment {
  to: number[]; // 6 ctrl targets
  duration: number; // s of sim time
  start?: number;
  from?: number[];
  onEnd?: () => void;
}

export interface MoveResult {
  ok: boolean;
  reason?: Reason;
  /** True if the jaws didn't touch both sides when the weld took the block. */
  assisted?: boolean;
}

interface SitePose {
  pos: Vec3;
  point: Vec3; // the fingers' direction
  open: Vec3; // the jaws' opening direction
}

function minJerk(s: number): number {
  const t = Math.min(1, Math.max(0, s));
  return t * t * t * (10 - 15 * t + 6 * t * t);
}

function quatYaw(q: ArrayLike<number>, i: number): number {
  const [w, x, y, z] = [q[i], q[i + 1], q[i + 2], q[i + 3]];
  return Math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z));
}

function quatMul(a: number[], b: number[]): number[] {
  return [
    a[0] * b[0] - a[1] * b[1] - a[2] * b[2] - a[3] * b[3],
    a[0] * b[1] + a[1] * b[0] + a[2] * b[3] - a[3] * b[2],
    a[0] * b[2] - a[1] * b[3] + a[2] * b[0] + a[3] * b[1],
    a[0] * b[3] + a[1] * b[2] - a[2] * b[1] + a[3] * b[0],
  ];
}

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}

function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}

export class Arm {
  readonly ik: ArmKinematics;
  readonly blocks = new Map<BlockId, Block>();
  private readonly queue: Segment[] = [];
  private readonly ctrlAdr: number[]; // actuators in HOME order
  private readonly gripperBodies: Set<number>;
  private readonly gripperBody: number;
  private readonly gripperQpos: number;
  private readonly gripEq: number;
  private readonly noEquality: number;
  private held: Block | null = null;
  private onIdle: ((r: MoveResult) => void) | null = null;
  private result: MoveResult = { ok: true };

  constructor(
    private readonly mj: Mujoco,
    private readonly model: MjModel,
    private readonly data: MjData,
  ) {
    this.ik = new ArmKinematics(mj, model);
    const O = mj.mjtObj;
    const id = (type: number, name: string) => {
      const i = mj.mj_name2id(model, type, name);
      if (i < 0) throw new Error(`no ${name} in the model`);
      return i;
    };
    const names = ['shoulder_pan', 'shoulder_lift', 'elbow_flex', 'wrist_flex', 'wrist_roll', 'gripper'];
    this.ctrlAdr = names.map((n) => id(O.mjOBJ_ACTUATOR.value, n));
    this.gripperBody = id(O.mjOBJ_BODY.value, 'gripper');
    this.gripperBodies = new Set([this.gripperBody, id(O.mjOBJ_BODY.value, 'moving_jaw_so101_v1')]);
    this.gripperQpos = (model.jnt_qposadr as Int32Array)[id(O.mjOBJ_JOINT.value, 'gripper')];
    this.gripEq = id(O.mjOBJ_EQUALITY.value, 'grip');
    this.noEquality = mj.mjtDisableBit.mjDSBL_EQUALITY.value;
    this.setWeld(false);
    for (const b of BLOCK_IDS) {
      const joint = id(O.mjOBJ_JOINT.value, b);
      this.blocks.set(b, {
        id: b,
        body: id(O.mjOBJ_BODY.value, b),
        geom: id(O.mjOBJ_GEOM.value, b),
        qposAdr: (model.jnt_qposadr as Int32Array)[joint],
        dofAdr: (model.jnt_dofadr as Int32Array)[joint],
        movedAt: 0,
      });
    }
  }

  get busy(): boolean {
    return this.queue.length > 0;
  }

  get holding(): BlockId | null {
    return this.held?.id ?? null;
  }

  /** The 5 arm joints now. */
  armQ(): number[] {
    return this.ik.qposAdr.map((a) => this.data.qpos[a]);
  }

  /** Put the arm at home and hold it there (no motion). */
  goHomeNow(): void {
    for (let j = 0; j < 5; j++) this.data.qpos[this.ik.qposAdr[j]] = HOME[j];
    this.data.qpos[this.gripperQpos] = HOME[5];
    this.setCtrl(HOME);
    this.mj.mj_forward(this.model, this.data);
  }

  /** Place a block flat at a table-frame spot (resets its velocity). */
  putBlock(id: BlockId, xMm: number, yMm: number, yaw = 0, zBottomMm = 0): void {
    const b = this.blocks.get(id)!;
    const [X, Y] = tableToWorld(xMm, yMm);
    const q = this.data.qpos;
    q[b.qposAdr] = X;
    q[b.qposAdr + 1] = Y;
    q[b.qposAdr + 2] = zBottomMm / 1000 + HALF + 0.0005;
    q[b.qposAdr + 3] = Math.cos(yaw / 2);
    q[b.qposAdr + 4] = 0;
    q[b.qposAdr + 5] = 0;
    q[b.qposAdr + 6] = Math.sin(yaw / 2);
    for (let k = 0; k < 6; k++) this.data.qvel[b.dofAdr + k] = 0;
    this.mj.mj_forward(this.model, this.data);
  }

  block(id: BlockId): BlockState {
    const b = this.blocks.get(id)!;
    const p = this.data.xpos as Float64Array;
    const center: Vec3 = [p[3 * b.body], p[3 * b.body + 1], p[3 * b.body + 2]];
    const [x, y] = worldToTable(center[0], center[1]);
    return {
      id,
      center,
      yaw: quatYaw(this.data.xquat as Float64Array, 4 * b.body),
      tableMm: [x, y],
      heightMm: (center[2] + HALF) * 1000,
    };
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
    this.setCtrl(seg.from!.map((f, i) => f + (seg.to[i] - f) * s));
    if (t - seg.start >= seg.duration) {
      this.queue.shift();
      seg.onEnd?.();
      if (!this.queue.length) this.finish(this.result);
    }
  }

  /** Stop where it is: hold the joints' current positions and drop the plan. */
  stop(): void {
    if (!this.busy) return;
    this.queue.length = 0;
    this.setCtrl([...this.armQ(), this.data.qpos[this.gripperQpos]]);
    this.finish({ ok: false, reason: 'stopped' });
  }

  /** Go home. A block still held after a stop is let go first, straight down where it is. */
  home(): Promise<MoveResult> {
    if (this.busy) return Promise.resolve({ ok: false, reason: 'busy' });
    this.result = { ok: true };
    const q = this.armQ();
    const b = this.held;
    if (b) {
      const letGo = this.seg([...q, GRIPPER_OPEN], q, 0.4);
      letGo.onEnd = () => {
        this.release(b);
        this.touchGripper(b, true);
      };
      this.queue.push(letGo);
    }
    this.queue.push(this.seg(HOME, q, 0.6));
    return this.whenIdle();
  }

  /** Run a pick and place. Resolves when the arm is home again. */
  move(id: BlockId, place: Place): Promise<MoveResult> {
    if (this.busy) return Promise.resolve({ ok: false, reason: 'busy' });
    const plan = this.canMove(id, place);
    if (typeof plan === 'string') return Promise.resolve({ ok: false, reason: plan });
    this.result = { ok: true };
    this.queue.push(...plan);
    return this.whenIdle();
  }

  /**
   * Plan a pick and place; nothing moves unless every pose is reachable. `place` is where the
   * block's centre should go in the table frame, and the height its bottom should rest at.
   */
  canMove(id: BlockId, place: Place): Segment[] | Reason {
    const b = this.blocks.get(id);
    if (!b) return 'no_block';
    const st = this.block(id);
    const q0 = this.armQ();

    // Down at the block with the jaws turned to its faces and the fixed jaw just off one face.
    // A block on another block is gripped that much higher.
    const zGrasp = Z_GRASP + Math.max(0, st.center[2] - HALF);
    const target = (pos: Vec3): IkTarget => ({ pos, yaw: st.yaw, maxTiltDeg: 4 });
    const grasp0 = this.ik.solve(target([st.center[0], st.center[1], zGrasp]), q0);
    if (!grasp0.ok) return 'cant_reach';
    const open = this.ik.fk(grasp0.q).open;
    const sx = st.center[0] + GRIP_OFFSET * open[0];
    const sy = st.center[1] + GRIP_OFFSET * open[1];
    const grasp = this.ik.solve(target([sx, sy, zGrasp]), grasp0.q);
    if (!grasp.ok) return 'cant_reach';
    const down = this.vertical([sx, sy], zGrasp, st.yaw, grasp.q); // top first
    if (!down) return 'cant_reach';

    // Where the block should sit in the site's frame once gripped, and the way down at the
    // place. Both are worked out again from the real grip, once the jaws have closed.
    const expected = this.toSite(this.ik.fk(grasp.q), st.center);
    const [px, py] = tableToWorld(place.xMm, place.yMm);
    const want: Vec3 = [px, py, place.zMm / 1000 + HALF + 0.001];
    const planned = this.placePath(expected, want, down[0]);
    if (!planned) return 'cant_reach';

    const op = GRIPPER_OPEN;
    const shut = GRIPPER_SHUT;
    const close = this.seg([...grasp.q, shut], grasp.q, 0.6);
    const lift = this.path([...down].reverse(), shut, 0.7);
    const rest = this.carryAndPlace(b, planned, down[0]);
    close.onEnd = () => {
      const local = this.grip(b, expected);
      const real = local ? this.placePath(local, want, down[0]) : null;
      const next: Segment[] = [];
      if (!local) {
        // The block isn't where the jaws should hold it: let go and go home.
        this.release(b);
        this.touchGripper(b, true);
        this.result = { ok: false, reason: 'slipped' };
        next.push(this.seg([...down[0], op], grasp.q), this.seg(HOME, down[0]));
      } else {
        next.push(...this.carryAndPlace(b, real ?? planned, down[0]));
      }
      // Replace what follows the lift (the lift itself is already queued).
      const i = this.queue.indexOf(lift[lift.length - 1]);
      this.queue.splice(i + 1, this.queue.length - i - 1, ...next);
    };
    return [this.seg([...down[0], op], q0), ...this.path(down, op, 0.8), close, ...lift, ...rest];
  }

  /**
   * Carry from above the block to above the place, down, let go, up, home. `way` is the place
   * path from placePath: top first, then the place, then a pose 5 mm above it.
   */
  private carryAndPlace(b: Block, way: number[][], from: number[]): Segment[] {
    const op = GRIPPER_OPEN;
    const down = way.slice(0, -1);
    const placeQ = down[down.length - 1];
    const eased = way[way.length - 1];
    // The weld lets go (the block drops the last millimetre; it no longer touches the
    // gripper), then the jaw opens while the arm eases up 5 mm, so the jaw can't scrape the
    // table, and only once the arm is up does the block touch the gripper again.
    const settle = this.seg([...placeQ, GRIPPER_SHUT], placeQ, 0.15);
    settle.onEnd = () => this.release(b);
    const letGo = this.seg([...eased, op], placeQ, 0.4);
    const rise = this.path([eased, ...[...down].reverse().slice(1)], op, 0.6);
    rise[rise.length - 1].onEnd = () => this.touchGripper(b, true);
    return [
      this.seg([...down[0], GRIPPER_SHUT], from),
      ...this.path(down, GRIPPER_SHUT, 0.8),
      settle,
      letGo,
      ...rise,
      this.seg(HOME, down[0]),
    ];
  }

  /**
   * Joint poses straight above a spot, from the highest lift that works down to `qBottom` (top
   * first). The fingers may tilt more the higher they are, hardly at all near the bottom.
   */
  private vertical(
    xy: [number, number],
    zBottom: number,
    yaw: number | null,
    qBottom: number[],
  ): number[][] | null {
    for (const lift of LIFTS) {
      const zTop = Math.max(lift, zBottom + (lift - Z_GRASP));
      const qs = [qBottom];
      let q = qBottom;
      for (let k = 1; k <= STEPS; k++) {
        const f = k / STEPS;
        const z = zBottom + f * (zTop - zBottom);
        const r = this.ik.solve({ pos: [xy[0], xy[1], z], yaw, maxTiltDeg: 4 + 31 * f }, q);
        if (!r.ok) break;
        q = r.q;
        qs.push(q);
      }
      if (qs.length === STEPS + 1) return qs.reverse();
    }
    return null;
  }

  /**
   * The way down that puts a gripped block's centre at `want`, flat (top first), then a pose
   * 5 mm above the place for letting go. `local` is the block's centre in the site's frame.
   */
  private placePath(local: Vec3, want: Vec3, q0: number[]): number[][] | null {
    let q = q0;
    for (let k = 0; k < 3; k++) {
      const off = this.fromSite(this.ik.fk(q), local);
      const r = this.ik.solve(
        { pos: [want[0] - off[0], want[1] - off[1], want[2] - off[2]], yaw: null, maxTiltDeg: 4 },
        q,
      );
      if (!r.ok) return null;
      q = r.q;
    }
    const site = this.ik.fk(q).pos;
    const way = this.vertical([site[0], site[1]], site[2], null, q);
    const eased = this.ik.solve({ pos: [site[0], site[1], site[2] + 0.005], yaw: null, maxTiltDeg: 4 }, q);
    return way && eased.ok ? [...way, eased.q] : null;
  }

  /** Segments through joint poses, `duration` seconds at least in all. */
  private path(qs: number[][], grip: number, duration: number): Segment[] {
    const out: Segment[] = [];
    for (let k = 1; k < qs.length; k++) {
      out.push(this.seg([...qs[k], grip], qs[k - 1], duration / (qs.length - 1)));
    }
    return out;
  }

  private seg(to: number[], from5: number[], minDuration = 0.6): Segment {
    const d = Math.max(...to.slice(0, 5).map((v, j) => Math.abs(v - from5[j])));
    return { to, duration: Math.max(minDuration, d / SPEED) };
  }

  private whenIdle(): Promise<MoveResult> {
    return new Promise((resolve) => {
      this.onIdle = resolve;
    });
  }

  private finish(r: MoveResult): void {
    const done = this.onIdle;
    this.onIdle = null;
    done?.(r);
  }

  private ctrlNow(): number[] {
    return this.ctrlAdr.map((a) => this.data.ctrl[a]);
  }

  private setCtrl(v: number[]): void {
    for (let i = 0; i < 6; i++) this.data.ctrl[this.ctrlAdr[i]] = v[i];
  }

  /** The gripper site's pose in the live simulation. */
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

  private toSite(pose: SitePose, p: Vec3): Vec3 {
    const y = cross(pose.open, pose.point);
    const d: Vec3 = [p[0] - pose.pos[0], p[1] - pose.pos[1], p[2] - pose.pos[2]];
    return [dot(d, pose.point), dot(d, y), dot(d, pose.open)];
  }

  private fromSite(pose: SitePose, l: Vec3): Vec3 {
    const y = cross(pose.open, pose.point);
    return [0, 1, 2].map((i) => l[0] * pose.point[i] + l[1] * y[i] + l[2] * pose.open[i]) as Vec3;
  }

  /** Both jaws touching the block? */
  private jawsTouch(b: Block): boolean {
    const touched = new Set<number>();
    const contacts = this.data.contact;
    const bodyOf = this.model.geom_bodyid as Int32Array;
    try {
      for (let i = 0; i < this.data.ncon; i++) {
        const c = contacts.get(i);
        if (!c) continue;
        const other = c.geom1 === b.geom ? c.geom2 : c.geom2 === b.geom ? c.geom1 : -1;
        if (other >= 0 && this.gripperBodies.has(bodyOf[other])) touched.add(bodyOf[other]);
        c.delete();
      }
    } finally {
      contacts.delete();
    }
    return touched.size === 2;
  }

  /**
   * Hold the block with the weld where it is now relative to the gripper. Returns the block's
   * centre in the site's frame, or null (and holds nothing) if it isn't between the jaws as
   * expected: more than 8 mm off, or tipped over.
   */
  private grip(b: Block, expected: Vec3): Vec3 | null {
    const local = this.toSite(this.livePose(), this.block(b.id).center);
    const off = Math.hypot(local[0] - expected[0], local[1] - expected[1], local[2] - expected[2]);
    const xq = this.data.xquat as Float64Array;
    const [, qx, qy] = [xq[4 * b.body], xq[4 * b.body + 1], xq[4 * b.body + 2]];
    const tipped = 1 - 2 * (qx * qx + qy * qy) < Math.cos((15 * Math.PI) / 180);
    if (off > 0.008 || tipped) return null;

    const touching = this.jawsTouch(b);
    const xp = this.data.xpos as Float64Array;
    const R = this.data.xmat as Float64Array;
    const g = this.gripperBody;
    const q1 = [xq[4 * g], xq[4 * g + 1], xq[4 * g + 2], xq[4 * g + 3]];
    const q2 = [xq[4 * b.body], xq[4 * b.body + 1], xq[4 * b.body + 2], xq[4 * b.body + 3]];
    const d = [0, 1, 2].map((k) => xp[3 * b.body + k] - xp[3 * g + k]);
    // relpose: body2's origin in body1's frame (R^T d), and its orientation q1^-1 q2.
    const rel = [0, 1, 2].map((c) => R[9 * g + c] * d[0] + R[9 * g + 3 + c] * d[1] + R[9 * g + 6 + c] * d[2]);
    const relq = quatMul([q1[0], -q1[1], -q1[2], -q1[3]], q2);
    const n = 11; // mjNEQDATA: anchor (3), relpose (7), torquescale (1)
    const e = this.model.eq_data as Float64Array;
    e.set([0, 0, 0, ...rel, ...relq, e[this.gripEq * n + 10]], this.gripEq * n);
    (this.model.eq_obj2id as Int32Array)[this.gripEq] = b.body;
    this.setWeld(true);
    this.touchGripper(b, false);
    this.held = b;
    this.result = { ok: true, assisted: !touching };
    return local;
  }

  private release(b: Block): void {
    this.setWeld(false);
    b.movedAt = performance.now() / 1000;
    this.held = null;
  }

  /** Contact groups: a block is 3 (touches the gripper's 1), a held block 2 (doesn't). */
  private touchGripper(b: Block, on: boolean): void {
    (this.model.geom_contype as Int32Array)[b.geom] = on ? 3 : 2;
    (this.model.geom_conaffinity as Int32Array)[b.geom] = on ? 3 : 2;
  }

  private setWeld(on: boolean): void {
    const opt = this.model.opt;
    opt.disableflags = on ? opt.disableflags & ~this.noEquality : opt.disableflags | this.noEquality;
  }
}
