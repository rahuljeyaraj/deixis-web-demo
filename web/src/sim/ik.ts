/**
 * Inverse kinematics for the SO-101 (plan 6.2): damped least squares on MuJoCo's site
 * Jacobian (`mj_jacSite` on `gripperframe`), with the fingers pointing a chosen way - straight
 * down unless asked otherwise - and the jaws turned to a chosen yaw. Solved in a scratch
 * `MjData`, so the live simulation is never disturbed.
 *
 * The `gripperframe` site: its x axis is the way the fingers point, its z axis is the way the
 * jaws open (the moving jaw swings along it), its y axis is the jaw hinge.
 */

import type { MjData, MjModel, Mujoco } from './mujoco';
import type { Vec3 } from './table';

export const ARM_JOINTS = [
  'shoulder_pan',
  'shoulder_lift',
  'elbow_flex',
  'wrist_flex',
  'wrist_roll',
] as const;

export interface IkTarget {
  /** Where the gripperframe site should be, world metres. */
  pos: Vec3;
  /**
   * Aim a point this far from the site **along the way the jaws open** at `pos`, rather than
   * the site itself, metres.
   *
   * This is how to aim the middle of the grip at a thing. The jaws are not symmetric about the
   * site - one is fixed and the other swings along the site's z - so the middle of the opening
   * sits about 30 mm to one side of it, and which side depends on which way round the jaws come
   * out. Standing the hand to one side by hand cannot work: the side has to be chosen before
   * the pose is solved and is only known after, and trying it both ways found that the arm
   * opens the jaws away from wherever the hand is put, so neither side ever lands the thing in
   * the middle (measured, 24 Oct: 50 and 58 mm off, whichever side was tried).
   *
   * Asked for this way there is no side to choose. The solver aims the middle of the opening
   * itself, and the hand ends up wherever that puts it - the jaws may come out either way round
   * and the thing is between them either way.
   */
  alongOpenM?: number;
  /**
   * The jaws' opening direction (yaw about z, radians), modulo a quarter turn: a cube looks the
   * same every 90 degrees, so the solver picks the nearest one. Null: any yaw.
   */
  yaw?: number | null;
  /** The jaws open exactly along `yaw`, not the nearest of its four quarter turns. */
  yawExact?: boolean;
  /**
   * How many ways round count as the same grip. 4 (the default) is a cube, which looks the
   * same every quarter turn. **2 is a pair of jaws**: they open both ways at once, so opening
   * along +y and along -y are the same grip and the solver should be free to take whichever it
   * can reach. Demanding one of the two cost six things their grip - three jars, two tubes of
   * glue and a bottle - all of them refused for a jaw heading 168 degrees out, which is to say
   * for being 12 degrees from the other way round.
   */
  yawTurns?: number;
  /**
   * How far the fingers may tilt from `point`, degrees (0: exactly along it). Close to the
   * base the wrist can't point straight down at height, so carry poses allow some tilt.
   */
  maxTiltDeg?: number;
  /**
   * Which way the **jaw hinge** should point, world metres, any length: the site's y axis, the
   * pin the moving jaw swings on.
   *
   * This is what says which way up a held thing is. A thing held by a level hand stands with
   * its own up along the hinge, so a vertical hinge is an upright jar and a hinge laid over is
   * a jar on its side - which is what the solver chose when it was left free (rahul, 24 Oct).
   *
   * **Ask for this rather than for `point` and `open` together.** Those two pin the hand's
   * whole orientation, which with the position is six things asked of a five-joint arm: the
   * jar came up straight and then there was no pose at all to carry it in. The hinge and the
   * position are five, which this arm has, and what is left free is the hand's heading - so
   * `point` becomes a preference rather than a demand, pulled at faintly and reported as
   * `angErrDeg` for the caller to judge against `maxTiltDeg`.
   *
   * Exact: there is no "either way round" here. A pair of jaws opens both ways, but a hinge
   * turned end for end holds a thing upside down.
   */
  hinge?: Vec3;
  /**
   * Which way the jaws should open, in full, world metres, any length - not merely their
   * compass heading, which is what `yaw` says.
   *
   * `yaw` can only ask for a heading, and a heading says nothing about the hand's roll: the
   * jaws opening along +x are the same heading whether they stand level or on end. That is
   * enough for a grip, where the hand's pointing settles the roll between them, and it is not
   * enough for a **carry**. Told to hold a jar level with the yaw free, the solver rolled the
   * wrist to whatever was easiest and laid the jar on its side (rahul, 24 Oct). The jar stays
   * upright only while the jaws stay level, which is a direction in three dimensions.
   *
   * Given, it replaces `yaw`. `yawTurns` still applies: a pair of jaws opens both ways at
   * once, so `open` and its opposite are the same grip.
   */
  open?: Vec3;
  /**
   * Which way the fingers should point, world metres, any length. Default: straight down,
   * which is how a thing lying on a table or standing on an open shelf is picked up.
   *
   * A thing inside a shelf bay cannot be: there is a board over it, so the hand has to come
   * in from the front, level. That is what the workbench's bays need, and it is what
   * `tests/probes/bay-ring.probe.test.ts` measures the reach at bay height with.
   */
  point?: Vec3;
}

export interface IkResult {
  q: number[]; // the 5 arm joints, radians
  posErrMm: number;
  angErrDeg: number; // between the fingers and `point` (straight down unless it was given)
  yawErrDeg: number;
  ok: boolean;
  iterations: number;
}

export interface Pose {
  pos: Vec3;
  point: Vec3; // the fingers' direction (site x axis)
  open: Vec3; // the jaws' opening direction (site z axis)
}

const DOWN: Vec3 = [0, 0, -1];
const POS_TOL_M = 0.0005;
const ANG_TOL = (1 * Math.PI) / 180;

function cross(a: Vec3, b: Vec3): Vec3 {
  return [a[1] * b[2] - a[2] * b[1], a[2] * b[0] - a[0] * b[2], a[0] * b[1] - a[1] * b[0]];
}
function dot(a: Vec3, b: Vec3): number {
  return a[0] * b[0] + a[1] * b[1] + a[2] * b[2];
}
/** A direction, made unit length. */
function unit(v: Vec3): Vec3 {
  const n = Math.hypot(...v) || 1;
  return [v[0] / n, v[1] / n, v[2] / n];
}
/** The horizontal part of a direction, made unit length. */
function flat(v: Vec3): Vec3 {
  const n = Math.hypot(v[0], v[1]) || 1;
  return [v[0] / n, v[1] / n, 0];
}
function angle(a: Vec3, b: Vec3): number {
  return Math.atan2(Math.hypot(...cross(a, b)), dot(a, b));
}

/** Solve A x = b for a small dense square system (Gaussian elimination, partial pivoting). */
export function solveLinear(A: number[][], b: number[]): number[] {
  const n = b.length;
  const M = A.map((row, i) => [...row, b[i]]);
  for (let c = 0; c < n; c++) {
    let p = c;
    for (let r = c + 1; r < n; r++) if (Math.abs(M[r][c]) > Math.abs(M[p][c])) p = r;
    [M[c], M[p]] = [M[p], M[c]];
    const d = M[c][c] || 1e-12;
    for (let r = c + 1; r < n; r++) {
      const f = M[r][c] / d;
      for (let k = c; k <= n; k++) M[r][k] -= f * M[c][k];
    }
  }
  const x = new Array<number>(n).fill(0);
  for (let r = n - 1; r >= 0; r--) {
    let s = M[r][n];
    for (let k = r + 1; k < n; k++) s -= M[r][k] * x[k];
    x[r] = s / (M[r][r] || 1e-12);
  }
  return x;
}

export class ArmKinematics {
  readonly qposAdr: number[];
  readonly dofAdr: number[];
  readonly lo: number[];
  readonly hi: number[];
  readonly siteId: number;
  private readonly gripperAdr: number;
  /** The arm's own bodies (base to jaws), for the check that it doesn't hit itself. */
  private readonly armBodies: Set<number>;
  private readonly scratch: MjData;
  /** The body the arm hangs from: its own base, or the carriage when it rides a rail. */
  private readonly rootBody: number;
  /** The carriage's slide joint, qpos address, or -1 when the arm stands still. */
  private railAdr = -1;
  private readonly jacp: InstanceType<Mujoco['DoubleBuffer']>;
  private readonly jacr: InstanceType<Mujoco['DoubleBuffer']>;

  /**
   * Where this arm stands, world metres. The one-arm scene puts its arm at the origin; the
   * end demo's shelf scene has two, each inside a `<body name="left_stand"/>` of its own, so
   * nothing may assume the base is at 0.
   *
   * **Not fixed when the arm rides a rail**: the workbench's arm stands on a carriage, and
   * `rideTo` moves this with it. Without that, every solve would be answered for a carriage
   * parked in the middle of the rail wherever the real one stood, which is the whole point of
   * having a rail.
   */
  base: Vec3;
  /**
   * Which way it faces, radians about z, 0 being +x. The shoulder turns 110 degrees either
   * side of this and no further, which is what decides whether a thing can be reached at all
   * - the end demo's arms are turned so that one fan covers both the shelf and the person.
   */
  readonly baseYaw: number;

  /**
   * @param prefix What `<attach model="so101" prefix="left_"/>` put in front of every name.
   *   Empty for the one-arm scene, whose names are bare.
   */
  constructor(
    private readonly mj: Mujoco,
    private readonly model: MjModel,
    private readonly prefix = '',
  ) {
    const J = mj.mjtObj.mjOBJ_JOINT.value;
    const ids = ARM_JOINTS.map((n) => {
      const id = mj.mj_name2id(model, J, `${prefix}${n}`);
      if (id < 0) throw new Error(`no joint ${prefix}${n}`);
      return id;
    });
    const qa = model.jnt_qposadr as Int32Array;
    const da = model.jnt_dofadr as Int32Array;
    const range = model.jnt_range as Float64Array;
    this.qposAdr = ids.map((i) => qa[i]);
    this.dofAdr = ids.map((i) => da[i]);
    this.lo = ids.map((i) => range[2 * i]);
    this.hi = ids.map((i) => range[2 * i + 1]);
    this.siteId = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE.value, `${prefix}gripperframe`);
    if (this.siteId < 0) throw new Error(`no ${prefix}gripperframe site`);
    this.gripperAdr = qa[mj.mj_name2id(model, J, `${prefix}gripper`)];
    const root = model.body_rootid as Int32Array;
    const armRoot = root[(model.jnt_bodyid as Int32Array)[ids[0]]];
    // Each arm's own bodies. With two arms this is what keeps the self-collision check from
    // calling one arm touching the other a fold in the first arm: they are separate trees,
    // rooted at their own stand.
    this.armBodies = new Set([...root.keys()].filter((b) => b > 0 && root[b] === armRoot));
    this.rootBody = armRoot;
    const bp = model.body_pos as Float64Array;
    this.base = [bp[armRoot * 3], bp[armRoot * 3 + 1], bp[armRoot * 3 + 2]];
    // The stand's yaw out of its quaternion (w, x, y, z). Only the turn about z matters: a
    // stand that leaned would break more than the first guess.
    const q = model.body_quat as Float64Array;
    const [w, qx, qy, qz] = [q[armRoot * 4], q[armRoot * 4 + 1], q[armRoot * 4 + 2], q[armRoot * 4 + 3]];
    this.baseYaw = Math.atan2(2 * (w * qz + qx * qy), 1 - 2 * (qy * qy + qz * qz));
    this.scratch = new mj.MjData(model);
    this.jacp = new mj.DoubleBuffer(3 * model.nv);
    this.jacr = new mj.DoubleBuffer(3 * model.nv);
  }

  /**
   * Tell the solver the arm rides a rail, and name the joint that moves it.
   *
   * Everything here is solved in a scratch `MjData` of its own, so that planning a move never
   * disturbs the scene being drawn. That scratch starts every joint at nought - including the
   * carriage's - so until it is told otherwise the solver answers for an arm parked in the
   * middle of its rail, whatever the real one is doing.
   */
  ridesOn(jointName: string): void {
    const id = this.mj.mj_name2id(this.model, this.mj.mjtObj.mjOBJ_JOINT.value, jointName);
    if (id < 0) throw new Error(`no joint ${jointName} to ride on`);
    this.railAdr = (this.model.jnt_qposadr as Int32Array)[id];
  }

  /** Where on the rail the carriage is, for everything solved from here on (metres). */
  rideTo(where: number): void {
    if (this.railAdr < 0) throw new Error('this arm does not ride on a rail');
    this.scratch.qpos[this.railAdr] = where;
    this.mj.mj_kinematics(this.model, this.scratch);
    const p = this.scratch.xpos as Float64Array;
    this.base = [p[3 * this.rootBody], p[3 * this.rootBody + 1], p[3 * this.rootBody + 2]];
  }

  /** Where on the rail the solver currently thinks the carriage is, metres. */
  get ridingAt(): number {
    return this.railAdr < 0 ? 0 : this.scratch.qpos[this.railAdr];
  }

  /** The gripper's pose for the 5 arm joints (the rest of the scene is ignored). */
  fk(q: number[]): Pose {
    const d = this.scratch;
    for (let j = 0; j < 5; j++) d.qpos[this.qposAdr[j]] = q[j];
    this.mj.mj_kinematics(this.model, d);
    return this.pose();
  }

  /**
   * How deep the arm sinks into itself in this pose, m (0: it doesn't touch), with the gripper
   * at `gripper`. Near the base a hand turned far hits the upper arm with its camera mount.
   */
  selfDepth(q: number[], gripper: number): number {
    const d = this.scratch;
    for (let j = 0; j < 5; j++) d.qpos[this.qposAdr[j]] = q[j];
    d.qpos[this.gripperAdr] = gripper;
    this.mj.mj_fwdPosition(this.model, d);
    const bodyOf = this.model.geom_bodyid as Int32Array;
    const contacts = d.contact;
    let depth = 0;
    for (let i = 0; i < d.ncon; i++) {
      const c = contacts.get(i);
      if (!c) continue;
      if (this.armBodies.has(bodyOf[c.geom1]) && this.armBodies.has(bodyOf[c.geom2])) {
        depth = Math.max(depth, -c.dist);
      }
      c.delete();
    }
    contacts.delete();
    return depth;
  }

  /**
   * The arm sinks more than `toleranceM` into itself in this pose (see `selfDepth`). Any real
   * touch counts: even 6 mm stopped the hand 4 cm above a block once (6 Oct).
   */
  selfHits(q: number[], gripper: number, toleranceM = 0.001): boolean {
    return this.selfDepth(q, gripper) > toleranceM;
  }

  /**
   * What the arm would run into in this pose, and how deep - or null for nothing.
   *
   * **The other half of `selfDepth`, and the one that was missing.** That one asks whether the
   * arm folds into itself; this one asks whether it folds into the world. Nothing asked that
   * before, so every pose in a plan was solved and then driven blind: "the hand knocks out all
   * the things on its way" (rahul, 24 Oct), and two of the failures in `out/stage2-body.txt`
   * were the arm leaning on the bench, 97 mm from the pose it had been sent to, with the
   * motors pushing against a board for the rest of the move.
   *
   * The world is read from `live`, not from this solver's own scratch: a scratch that has never
   * been told where anything is has every thing at the place the scene file put it, which is
   * the same mistake as planning from the layout. Everything but the arm's own joints is copied
   * across, so the check is against the world as it stands at this moment.
   *
   * What does not count: the arm's own bodies (that is `selfDepth`), the body the arm stands on
   * - a carriage sitting in its trough touches the rail by design - and whatever geoms the
   * caller is willing to pass through, which is how a grip is allowed to close through a jar.
   */
  worldHit(
    live: MjData,
    q: number[],
    gripper: number,
    ignore: ReadonlySet<number>,
    toleranceM = 0.002,
    /**
     * Geoms held to a **stricter** figure than the rest, and what that figure is.
     *
     * The two are not the same question. A thing that is touched falls over, and that is the
     * fault being hunted - "the hand knocks out all the things on its way" (rahul, 24 Oct) - so
     * the slightest touch of one counts. The bench and the shelf do not fall over, and a hand
     * reaching under a board grazes it: refusing every plan that grazes the shelf refused every
     * grip in all three bays, which is a worse answer than a graze.
     */
    strict?: ReadonlySet<number>,
    strictM = 0.002,
  ): { name: string; depthM: number } | null {
    const d = this.scratch;
    const rail = this.railAdr >= 0 ? d.qpos[this.railAdr] : 0;
    (d.qpos as Float64Array).set(live.qpos as Float64Array);
    if (this.railAdr >= 0) d.qpos[this.railAdr] = rail;
    for (let j = 0; j < 5; j++) d.qpos[this.qposAdr[j]] = q[j];
    d.qpos[this.gripperAdr] = gripper;
    this.mj.mj_fwdPosition(this.model, d);
    const bodyOf = this.model.geom_bodyid as Int32Array;
    const contacts = d.contact;
    let worst: { name: string; depthM: number } | null = null;
    for (let i = 0; i < d.ncon; i++) {
      const c = contacts.get(i);
      if (!c) continue;
      const pair: [number, number][] = [
        [c.geom1, c.geom2],
        [c.geom2, c.geom1],
      ];
      for (const [arm, other] of pair) {
        const mine = this.armBodies.has(bodyOf[arm]) && bodyOf[arm] !== this.rootBody;
        if (!mine) continue;
        if (this.armBodies.has(bodyOf[other]) || ignore.has(other)) continue;
        const depthM = -c.dist;
        const allowed = strict?.has(other) ? strictM : toleranceM;
        if (depthM > allowed && (!worst || depthM > worst.depthM)) {
          // The geom's own name first: every bench slab, shelf board and upright has one, and
          // all of them hang off the world body, so the body's name would say only "world".
          // Both ends of it are named: what was hit, and what hit it. Which part of the arm
          // is leaning on the shelf is the difference between lowering the hand and choosing
          // another elbow.
          const naming = (g: number) =>
            this.mj.mj_id2name(this.model, this.mj.mjtObj.mjOBJ_GEOM.value, g) ??
            this.mj.mj_id2name(this.model, this.mj.mjtObj.mjOBJ_BODY.value, bodyOf[g]) ??
            'something';
          worst = { name: `${naming(other)} (with its ${naming(arm)})`, depthM };
        }
      }
      c.delete();
    }
    contacts.delete();
    return worst;
  }

  private pose(): Pose {
    const p = this.scratch.site_xpos as Float64Array;
    const R = this.scratch.site_xmat as Float64Array;
    const s = this.siteId;
    return {
      pos: [p[3 * s], p[3 * s + 1], p[3 * s + 2]],
      point: [R[9 * s], R[9 * s + 3], R[9 * s + 6]],
      open: [R[9 * s + 2], R[9 * s + 5], R[9 * s + 8]],
    };
  }

  /** A first guess that faces the target. Positive pan turns the arm toward -y. */
  guess(target: IkTarget): number[] {
    const [x, y] = target.pos;
    // From where this arm stands and which way it faces, not from the world origin.
    const bearing = Math.atan2(y - this.base[1], x - this.base[0]) - this.baseYaw;
    return [-Math.atan2(Math.sin(bearing), Math.cos(bearing)), -0.3, 0.6, 1.2, 0];
  }

  /** Which arm this is, for an error message: `left_` or `right_`, empty in the one-arm scene. */
  get name(): string {
    return this.prefix;
  }

  /**
   * Solve with a steady pull toward `point` (straight down by default: the straightest grip);
   * if that fails and some tilt is allowed, solve again with the tilt free inside the cone.
   */
  solve(target: IkTarget, q0?: number[], maxIter = 300): IkResult {
    const steady = this.solveIn(target, q0, maxIter, false);
    if (steady.ok || !target.maxTiltDeg) return steady;
    const cone = this.solveIn(target, q0, maxIter, true);
    return cone.ok || cone.posErrMm < steady.posErrMm ? cone : steady;
  }

  private solveIn(target: IkTarget, q0: number[] | undefined, maxIter: number, cone: boolean): IkResult {
    const best = this.solveFrom(target, q0 ?? this.guess(target), maxIter, cone);
    if (best.ok || q0 === undefined) return best;
    const again = this.solveFrom(target, this.guess(target), maxIter, cone);
    return again.posErrMm < best.posErrMm ? again : best;
  }

  private solveFrom(target: IkTarget, q0: number[], maxIter: number, cone: boolean): IkResult {
    const d = this.scratch;
    const m = this.model;
    const q = q0.map((v, j) => Math.min(this.hi[j], Math.max(this.lo[j], v)));
    const want = unit(target.point ?? DOWN);
    const along = target.alongOpenM ?? 0;
    const nv = m.nv;
    const maxTilt = ((target.maxTiltDeg ?? 0) * Math.PI) / 180;
    // 1 rad of tilt weighs like 10 cm of position; steady: much less when some tilt is
    // allowed. Cone: inside the allowed tilt only a faint pull toward straight down, so the
    // position wins; past it, the tilt is pulled back to the edge. (The steady pull fights the
    // position wherever the wrist sits at its limit: the hand stopped 13 mm short, and stacks
    // near the base were "too high", rahul 6 Oct.)
    const W_ANG = 0.1;
    const W_ANG_STEADY = 0.02;
    const W_ANG_INSIDE = 0.003;
    const wYaw = 0.05;
    const lambda2 = 1e-4;
    let wantOpen: Vec3 | null = null;
    const wantHinge = target.hinge ? unit(target.hinge) : null;
    let pose = this.fk(q);
    let it = 0;
    for (; it < maxIter; it++) {
      if (target.open && wantOpen === null) {
        // The nearer of the two ways round, for the same reason `yaw` takes the nearest turn.
        const c = unit(target.open);
        const turns = target.yawTurns ?? 4;
        wantOpen = turns === 2 && dot(c, pose.open) < 0 ? [-c[0], -c[1], -c[2]] : c;
      }
      if (!target.open && target.yaw !== null && target.yaw !== undefined && wantOpen === null) {
        // Of the four equivalent yaws, take the one closest to where the jaws face now.
        let bestDot = -Infinity;
        const turns = target.yawExact ? 1 : (target.yawTurns ?? 4);
        for (let k = 0; k < turns; k++) {
          const a = target.yaw + (k * 2 * Math.PI) / turns;
          const c: Vec3 = [Math.cos(a), Math.sin(a), 0];
          if (dot(c, pose.open) > bestDot) {
            bestDot = dot(c, pose.open);
            wantOpen = c;
          }
        }
      }
      // What is really being aimed: the site, or a point out along the jaws' opening.
      const aimed: Vec3 = [
        pose.pos[0] + along * pose.open[0],
        pose.pos[1] + along * pose.open[1],
        pose.pos[2] + along * pose.open[2],
      ];
      const ep: Vec3 = [
        target.pos[0] - aimed[0],
        target.pos[1] - aimed[1],
        target.pos[2] - aimed[2],
      ];
      const ea = cross(pose.point, want);
      // The yaw is the jaws' heading only (their direction seen from above), so the hand may
      // still tip forward with the jaws facing the same way.
      // Asked for a whole direction, the jaws are held to all three turns of it; asked for a
      // heading, only to the turn about up, so the hand may still tip forward.
      const jaws = target.open ? pose.open : flat(pose.open);
      const ec = wantOpen ? cross(jaws, wantOpen) : ([0, 0, 0] as Vec3);
      const hinge = cross(pose.open, pose.point);
      const eh = wantHinge ? cross(hinge, wantHinge) : ([0, 0, 0] as Vec3);
      const posErr = Math.hypot(...ep);
      const angErr = angle(pose.point, want);
      const yawErr = wantHinge
        ? angle(hinge, wantHinge)
        : wantOpen
          ? angle(jaws, wantOpen)
          : 0;
      if (posErr < POS_TOL_M && angErr < Math.max(ANG_TOL, maxTilt) && yawErr < ANG_TOL) break;

      this.mj.mj_comPos(m, d);
      this.mj.mj_jacSite(m, d, this.jacp, this.jacr, this.siteId);
      const Jp = this.jacp.GetView() as Float64Array;
      const Jr = this.jacr.GetView() as Float64Array;
      // Rows: 3 position, 3 tilt (weighted), 3 yaw (weighted); columns: the 5 arm dofs.
      const rows: number[][] = [];
      const err: number[] = [];
      // A point carried out along the opening moves with the hand's turn as well as with it,
      // so its own Jacobian is the site's plus how fast that offset is swung about.
      const spun = along
        ? this.dofAdr.map((c) =>
            cross([Jr[0 * nv + c], Jr[1 * nv + c], Jr[2 * nv + c]], pose.open),
          )
        : null;
      for (let r = 0; r < 3; r++) {
        rows.push(
          this.dofAdr.map((c, i) => Jp[r * nv + c] + (spun ? along * spun[i][r] : 0)),
        );
        err.push(ep[r]);
      }
      const over = cone && maxTilt > 0 ? Math.max(0, angErr - maxTilt) : angErr;
      // Asked for a hinge, the fingers are only a preference: the hinge and the position are
      // already all five joints, and pulling hard at the fingers as well is what left the arm
      // with no pose to carry a jar in.
      const wAng = wantHinge
        ? W_ANG_STEADY
        : maxTilt === 0
          ? W_ANG
          : !cone
            ? W_ANG_STEADY
            : over === 0
              ? W_ANG_INSIDE
              : W_ANG;
      const share = cone && over > 0 ? over / angErr : 1; // only the part past the edge
      for (let r = 0; r < 3; r++) {
        rows.push(this.dofAdr.map((c) => wAng * Jr[r * nv + c]));
        err.push(wAng * share * ea[r]);
      }
      if (wantHinge) {
        for (let r = 0; r < 3; r++) {
          rows.push(this.dofAdr.map((c) => W_ANG * Jr[r * nv + c]));
          err.push(W_ANG * eh[r]);
        }
      }
      if (wantHinge) {
        // The hinge has had its say; the jaws' own heading is then settled by it.
      } else if (wantOpen && target.open) {
        for (let r = 0; r < 3; r++) {
          rows.push(this.dofAdr.map((c) => wYaw * Jr[r * nv + c]));
          err.push(wYaw * ec[r]);
        }
      } else if (wantOpen) {
        rows.push(this.dofAdr.map((c) => wYaw * Jr[2 * nv + c])); // turning about up only
        err.push(wYaw * ec[2]);
      }
      // (J^T J + lambda^2 I) dq = J^T e. A joint at its limit that the step would push past it
      // is left out and the step solved again, so the other joints do its share.
      const stepFor = (free: boolean[]) => {
        const A = Array.from({ length: 5 }, (_, i) =>
          Array.from({ length: 5 }, (_, k) => {
            let s = i === k ? lambda2 : 0;
            if (free[i] && free[k]) for (const row of rows) s += row[i] * row[k];
            return s;
          }),
        );
        const g = Array.from({ length: 5 }, (_, i) =>
          free[i] ? rows.reduce((s, row, r) => s + row[i] * err[r], 0) : 0,
        );
        return solveLinear(A, g);
      };
      let dq = stepFor([true, true, true, true, true]);
      const stuck = dq.map((v, j) => (q[j] <= this.lo[j] + 1e-6 && v < 0) || (q[j] >= this.hi[j] - 1e-6 && v > 0));
      if (stuck.some(Boolean)) dq = stepFor(stuck.map((s) => !s));
      const step = Math.max(...dq.map(Math.abs));
      const scale = step > 0.2 ? 0.2 / step : 1;
      for (let j = 0; j < 5; j++) {
        q[j] = Math.min(this.hi[j], Math.max(this.lo[j], q[j] + scale * dq[j]));
      }
      pose = this.fk(q);
    }
    const posErrMm =
      1000 *
      Math.hypot(
        target.pos[0] - (pose.pos[0] + along * pose.open[0]),
        target.pos[1] - (pose.pos[1] + along * pose.open[1]),
        target.pos[2] - (pose.pos[2] + along * pose.open[2]),
      );
    const angErrDeg = (angle(pose.point, want) * 180) / Math.PI;
    const yawErrDeg = wantHinge
      ? (angle(cross(pose.open, pose.point), wantHinge) * 180) / Math.PI
      : wantOpen
        ? (angle(target.open ? pose.open : flat(pose.open), wantOpen) * 180) / Math.PI
        : 0;
    return {
      q,
      posErrMm,
      angErrDeg,
      yawErrDeg,
      ok: posErrMm < 1 && angErrDeg < Math.max(2, target.maxTiltDeg ?? 0) && yawErrDeg < 2,
      iterations: it,
    };
  }

  delete(): void {
    this.jacp.delete();
    this.jacr.delete();
    this.scratch.delete();
  }
}
