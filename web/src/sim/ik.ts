/**
 * Inverse kinematics for the SO-101 (plan 6.2): damped least squares on MuJoCo's site
 * Jacobian (`mj_jacSite` on `gripperframe`), with the fingers pointing straight down and the
 * jaws turned to a chosen yaw. Solved in a scratch `MjData`, so the live simulation is never
 * disturbed.
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
   * The jaws' opening direction (yaw about z, radians), modulo a quarter turn: a cube looks the
   * same every 90 degrees, so the solver picks the nearest one. Null: any yaw.
   */
  yaw?: number | null;
  /**
   * How far the fingers may tilt from straight down, degrees (0: straight down). Close to the
   * base the wrist can't point straight down at height, so carry poses allow some tilt.
   */
  maxTiltDeg?: number;
}

export interface IkResult {
  q: number[]; // the 5 arm joints, radians
  posErrMm: number;
  angErrDeg: number; // between the fingers and straight down
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
  private readonly scratch: MjData;
  private readonly jacp: InstanceType<Mujoco['DoubleBuffer']>;
  private readonly jacr: InstanceType<Mujoco['DoubleBuffer']>;

  constructor(
    private readonly mj: Mujoco,
    private readonly model: MjModel,
  ) {
    const J = mj.mjtObj.mjOBJ_JOINT.value;
    const ids = ARM_JOINTS.map((n) => {
      const id = mj.mj_name2id(model, J, n);
      if (id < 0) throw new Error(`no joint ${n}`);
      return id;
    });
    const qa = model.jnt_qposadr as Int32Array;
    const da = model.jnt_dofadr as Int32Array;
    const range = model.jnt_range as Float64Array;
    this.qposAdr = ids.map((i) => qa[i]);
    this.dofAdr = ids.map((i) => da[i]);
    this.lo = ids.map((i) => range[2 * i]);
    this.hi = ids.map((i) => range[2 * i + 1]);
    this.siteId = mj.mj_name2id(model, mj.mjtObj.mjOBJ_SITE.value, 'gripperframe');
    if (this.siteId < 0) throw new Error('no gripperframe site');
    this.scratch = new mj.MjData(model);
    this.jacp = new mj.DoubleBuffer(3 * model.nv);
    this.jacr = new mj.DoubleBuffer(3 * model.nv);
  }

  /** The gripper's pose for the 5 arm joints (the rest of the scene is ignored). */
  fk(q: number[]): Pose {
    const d = this.scratch;
    for (let j = 0; j < 5; j++) d.qpos[this.qposAdr[j]] = q[j];
    this.mj.mj_kinematics(this.model, d);
    return this.pose();
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
    return [-Math.atan2(y, x), -0.3, 0.6, 1.2, 0];
  }

  solve(target: IkTarget, q0?: number[], maxIter = 300): IkResult {
    const best = this.solveFrom(target, q0 ?? this.guess(target), maxIter);
    if (best.ok || q0 === undefined) return best;
    const again = this.solveFrom(target, this.guess(target), maxIter);
    return again.posErrMm < best.posErrMm ? again : best;
  }

  private solveFrom(target: IkTarget, q0: number[], maxIter: number): IkResult {
    const d = this.scratch;
    const m = this.model;
    const q = q0.map((v, j) => Math.min(this.hi[j], Math.max(this.lo[j], v)));
    const nv = m.nv;
    const maxTilt = ((target.maxTiltDeg ?? 0) * Math.PI) / 180;
    // 1 rad of tilt weighs like 10 cm of position; much less when some tilt is allowed.
    const wAng = maxTilt > 0 ? 0.02 : 0.1;
    const wYaw = 0.05;
    const lambda2 = 1e-4;
    let wantOpen: Vec3 | null = null;
    let pose = this.fk(q);
    let it = 0;
    for (; it < maxIter; it++) {
      if (target.yaw !== null && target.yaw !== undefined && wantOpen === null) {
        // Of the four equivalent yaws, take the one closest to where the jaws face now.
        let bestDot = -Infinity;
        for (let k = 0; k < 4; k++) {
          const a = target.yaw + (k * Math.PI) / 2;
          const c: Vec3 = [Math.cos(a), Math.sin(a), 0];
          if (dot(c, pose.open) > bestDot) {
            bestDot = dot(c, pose.open);
            wantOpen = c;
          }
        }
      }
      const ep: Vec3 = [
        target.pos[0] - pose.pos[0],
        target.pos[1] - pose.pos[1],
        target.pos[2] - pose.pos[2],
      ];
      const ea = cross(pose.point, DOWN);
      const ec = wantOpen ? cross(pose.open, wantOpen) : ([0, 0, 0] as Vec3);
      const posErr = Math.hypot(...ep);
      const angErr = angle(pose.point, DOWN);
      const yawErr = wantOpen ? angle(pose.open, wantOpen) : 0;
      if (posErr < POS_TOL_M && angErr < Math.max(ANG_TOL, maxTilt) && yawErr < ANG_TOL) break;

      this.mj.mj_comPos(m, d);
      this.mj.mj_jacSite(m, d, this.jacp, this.jacr, this.siteId);
      const Jp = this.jacp.GetView() as Float64Array;
      const Jr = this.jacr.GetView() as Float64Array;
      // Rows: 3 position, 3 tilt (weighted), 3 yaw (weighted); columns: the 5 arm dofs.
      const rows: number[][] = [];
      const err: number[] = [];
      for (let r = 0; r < 3; r++) {
        rows.push(this.dofAdr.map((c) => Jp[r * nv + c]));
        err.push(ep[r]);
      }
      for (let r = 0; r < 3; r++) {
        rows.push(this.dofAdr.map((c) => wAng * Jr[r * nv + c]));
        err.push(wAng * ea[r]);
      }
      if (wantOpen) {
        for (let r = 0; r < 3; r++) {
          rows.push(this.dofAdr.map((c) => wYaw * Jr[r * nv + c]));
          err.push(wYaw * ec[r]);
        }
      }
      // (J^T J + lambda^2 I) dq = J^T e
      const A = Array.from({ length: 5 }, (_, i) =>
        Array.from({ length: 5 }, (_, k) => {
          let s = i === k ? lambda2 : 0;
          for (const row of rows) s += row[i] * row[k];
          return s;
        }),
      );
      const g = Array.from({ length: 5 }, (_, i) => rows.reduce((s, row, r) => s + row[i] * err[r], 0));
      const dq = solveLinear(A, g);
      const step = Math.max(...dq.map(Math.abs));
      const scale = step > 0.2 ? 0.2 / step : 1;
      for (let j = 0; j < 5; j++) {
        q[j] = Math.min(this.hi[j], Math.max(this.lo[j], q[j] + scale * dq[j]));
      }
      pose = this.fk(q);
    }
    const posErrMm = 1000 * Math.hypot(
      target.pos[0] - pose.pos[0],
      target.pos[1] - pose.pos[1],
      target.pos[2] - pose.pos[2],
    );
    const angErrDeg = (angle(pose.point, DOWN) * 180) / Math.PI;
    const yawErrDeg = wantOpen ? (angle(pose.open, wantOpen) * 180) / Math.PI : 0;
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
