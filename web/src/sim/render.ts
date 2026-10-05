/**
 * Draws the MuJoCo scene with three.js (plan 6.1): one mesh per visible geom, moved from
 * `geom_xpos` / `geom_xmat` every frame, seen from where a seated person's eyes would be.
 */

import * as THREE from 'three';
import type { MjData, MjModel, Mujoco } from './mujoco';
import { AREA_MM, BASE_MM, REACH_MM, tableToWorld } from './table';

const VISIBLE_GROUPS = new Set([0, 1, 2]); // 3 and 4 are the arm's collision shapes

export interface ViewOptions {
  /** Draw over a transparent background (the webcam shows through). */
  transparent?: boolean;
}

export class TableView {
  /** Null without a canvas (Node tests: the scene and the camera for raycasts only). */
  readonly renderer: THREE.WebGLRenderer | null;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  /** Meshes the raycast may hit, with the geom id each one draws. */
  readonly pickable: THREE.Object3D[] = [];
  private readonly geoms: { id: number; obj: THREE.Object3D }[] = [];
  private readonly meshOfGeom = new Map<number, THREE.Mesh>();
  private readonly tableGeom: number;
  private readonly ghost: THREE.Mesh;

  constructor(
    readonly canvas: HTMLCanvasElement | null,
    mj: Mujoco,
    readonly model: MjModel,
    opts: ViewOptions = {},
  ) {
    this.renderer = canvas ? TableView.makeRenderer(canvas, !!opts.transparent) : null;
    if (!opts.transparent) this.scene.background = new THREE.Color(0xdfe5ee);

    this.camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.01, 10);
    this.camera.up.set(0, 0, 1);
    this.seatedView();

    this.tableGeom = mj.mj_name2id(model, mj.mjtObj.mjOBJ_GEOM.value, 'table');
    this.addLights();
    this.addGeoms(mj);
    this.addMat();
    this.ghost = this.makeGhost();
  }

  private static makeRenderer(canvas: HTMLCanvasElement, transparent: boolean): THREE.WebGLRenderer {
    const r = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: transparent });
    r.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    r.shadowMap.enabled = true;
    r.shadowMap.type = THREE.PCFShadowMap;
    r.outputColorSpace = THREE.SRGBColorSpace;
    r.localClippingEnabled = true;
    return r;
  }

  isTable(geom: number): boolean {
    return geom === this.tableGeom;
  }

  /** Make a geom glow (the block "that" means) or stop glowing. */
  glow(geom: number, on: boolean): void {
    const mesh = this.meshOfGeom.get(geom);
    if (!mesh) return;
    // Lit from inside in its own colour, so a chosen block reads as brighter, not paler; and a
    // pulsing pink ring on the table around it (the fingertip dot's colour).
    const m = mesh.material as THREE.MeshStandardMaterial;
    m.emissive.copy(on ? m.color : new THREE.Color(0x000000));
    m.emissiveIntensity = on ? 0.45 : 0;
    const halo = this.haloOf(mesh);
    halo.visible = on;
    if (on) this.halos.add(halo);
    else this.halos.delete(halo);
  }

  private readonly halos = new Set<THREE.Mesh>();

  /** A flat square ring under the block (it moves and turns with the block). */
  private haloOf(mesh: THREE.Mesh): THREE.Mesh {
    let halo = mesh.userData.halo as THREE.Mesh | undefined;
    if (!halo) {
      const box = (mesh.geometry as THREE.BoxGeometry).parameters;
      const half = box.width / 2;
      const outer = half + 0.009;
      const inner = half + 0.0015;
      const square = (s: number) => [
        new THREE.Vector2(-s, -s), new THREE.Vector2(s, -s), new THREE.Vector2(s, s), new THREE.Vector2(-s, s),
      ]; // prettier-ignore
      const shape = new THREE.Shape(square(outer));
      shape.holes.push(new THREE.Path(square(inner).reverse()));
      halo = new THREE.Mesh(
        new THREE.ShapeGeometry(shape),
        new THREE.MeshBasicMaterial({ color: 0xff4081, transparent: true, depthWrite: false }),
      );
      halo.position.z = -box.depth / 2 + 0.0012; // just above what the block stands on
      halo.raycast = () => {}; // pointing goes to the block, not its ring
      halo.renderOrder = 1;
      halo.visible = false;
      mesh.add(halo);
      mesh.userData.halo = halo;
    }
    return halo;
  }

  /**
   * Show the ghost block (where the block will go) in a block's colour, centred at a world
   * point, or hide it.
   */
  showGhost(at: [number, number, number] | null, colorOfGeom?: number): void {
    this.ghost.visible = at !== null;
    if (!at) return;
    this.ghost.position.set(at[0], at[1], at[2]);
    const src = colorOfGeom === undefined ? null : this.meshOfGeom.get(colorOfGeom);
    const color = src ? (src.material as THREE.MeshStandardMaterial).color : new THREE.Color(0xffffff);
    const mats = this.ghost.material as THREE.MeshBasicMaterial;
    mats.color.copy(color);
    (this.ghost.children[0] as THREE.LineSegments<THREE.EdgesGeometry, THREE.LineBasicMaterial>).material.color.copy(color);
  }

  private makeGhost(): THREE.Mesh {
    const size = 0.03;
    const geo = new THREE.BoxGeometry(size, size, size);
    const ghost = new THREE.Mesh(
      geo,
      new THREE.MeshBasicMaterial({ transparent: true, opacity: 0.35, depthWrite: false }),
    );
    const edges = new THREE.LineSegments(
      new THREE.EdgesGeometry(geo),
      new THREE.LineBasicMaterial({ transparent: true, opacity: 0.9 }),
    );
    ghost.add(edges);
    ghost.visible = false;
    ghost.renderOrder = 2;
    this.scene.add(ghost);
    return ghost;
  }

  /** The camera where a seated person's eyes would be, looking at the work area. */
  seatedView(): void {
    const [ex, ey] = tableToWorld(BASE_MM.x, -360);
    const [lx, ly] = tableToWorld(BASE_MM.x, 150);
    this.camera.position.set(ex, ey, 0.44);
    this.camera.lookAt(lx, ly, 0.085);
  }

  resize(width: number, height: number): void {
    this.renderer?.setSize(width, height, false);
    this.camera.aspect = width / height;
    this.camera.updateProjectionMatrix();
  }

  /** Copy every geom's pose from the simulation. */
  sync(data: MjData): void {
    const pos = data.geom_xpos as Float64Array;
    const mat = data.geom_xmat as Float64Array;
    for (const { id, obj } of this.geoms) {
      const p = id * 3;
      const r = id * 9;
      obj.matrix.set(
        mat[r], mat[r + 1], mat[r + 2], pos[p],
        mat[r + 3], mat[r + 4], mat[r + 5], pos[p + 1],
        mat[r + 6], mat[r + 7], mat[r + 8], pos[p + 2],
        0, 0, 0, 1,
      ); // prettier-ignore
      obj.matrixWorldNeedsUpdate = true;
    }
  }

  render(): void {
    const k = 0.5 + 0.5 * Math.sin(performance.now() / 160);
    for (const halo of this.halos) {
      halo.scale.set(1 + 0.15 * k, 1 + 0.15 * k, 1);
      (halo.material as THREE.MeshBasicMaterial).opacity = 0.55 + 0.45 * k;
    }
    this.renderer?.render(this.scene, this.camera);
  }

  private addLights(): void {
    this.scene.add(new THREE.HemisphereLight(0xffffff, 0x8a7560, 1.4));
    const sun = new THREE.DirectionalLight(0xffffff, 2.2);
    sun.position.set(0.45, -0.3, 0.9);
    sun.target.position.set(0.1, 0, 0);
    sun.castShadow = true;
    sun.shadow.mapSize.set(2048, 2048);
    const s = sun.shadow.camera;
    s.left = -0.45;
    s.right = 0.45;
    s.top = 0.45;
    s.bottom = -0.45;
    s.near = 0.1;
    s.far = 2;
    sun.shadow.bias = -0.0005;
    this.scene.add(sun, sun.target);
  }

  private addGeoms(mj: Mujoco): void {
    const m = this.model;
    const T = mj.mjtGeom;
    const types = m.geom_type as Int32Array;
    const groups = m.geom_group as Int32Array;
    const sizes = m.geom_size as Float64Array;
    const dataIds = m.geom_dataid as Int32Array;
    const matIds = m.geom_matid as Int32Array;
    const geomRgba = m.geom_rgba as Float32Array;
    const matRgba = m.mat_rgba as Float32Array;
    const meshCache = new Map<number, THREE.BufferGeometry>();

    for (let i = 0; i < m.ngeom; i++) {
      if (!VISIBLE_GROUPS.has(groups[i])) continue;
      const t = types[i];
      const sz = [sizes[i * 3], sizes[i * 3 + 1], sizes[i * 3 + 2]];
      let geo: THREE.BufferGeometry | null = null;
      if (t === T.mjGEOM_MESH.value) {
        const id = dataIds[i];
        geo = meshCache.get(id) ?? this.meshGeometry(id);
        meshCache.set(id, geo);
      } else if (t === T.mjGEOM_BOX.value) {
        geo = new THREE.BoxGeometry(2 * sz[0], 2 * sz[1], 2 * sz[2]);
      } else if (t === T.mjGEOM_SPHERE.value) {
        geo = new THREE.SphereGeometry(sz[0], 16, 12);
      } else if (t === T.mjGEOM_CAPSULE.value) {
        geo = new THREE.CapsuleGeometry(sz[0], 2 * sz[1], 6, 12).rotateX(Math.PI / 2);
      } else if (t === T.mjGEOM_CYLINDER.value) {
        geo = new THREE.CylinderGeometry(sz[0], sz[0], 2 * sz[1], 20).rotateX(Math.PI / 2);
      }
      if (!geo) continue;

      const mi = matIds[i];
      const rgba = mi >= 0 ? matRgba.subarray(mi * 4, mi * 4 + 4) : geomRgba.subarray(i * 4, i * 4 + 4);
      const color = new THREE.Color().setRGB(rgba[0], rgba[1], rgba[2], THREE.SRGBColorSpace);
      const isMesh = t === T.mjGEOM_MESH.value;
      const material = new THREE.MeshStandardMaterial({
        color,
        roughness: isMesh ? 0.55 : 0.7,
        metalness: 0.0,
        flatShading: isMesh,
        transparent: rgba[3] < 1,
        opacity: rgba[3],
      });
      const mesh = new THREE.Mesh(geo, material);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.matrixAutoUpdate = false;
      mesh.userData.geom = i;
      this.meshOfGeom.set(i, mesh);
      this.scene.add(mesh);
      this.geoms.push({ id: i, obj: mesh });
      this.pickable.push(mesh);
    }
  }

  private meshGeometry(meshId: number): THREE.BufferGeometry {
    const m = this.model;
    const vAdr = (m.mesh_vertadr as Int32Array)[meshId];
    const vNum = (m.mesh_vertnum as Int32Array)[meshId];
    const fAdr = (m.mesh_faceadr as Int32Array)[meshId];
    const fNum = (m.mesh_facenum as Int32Array)[meshId];
    const verts = (m.mesh_vert as Float32Array).slice(vAdr * 3, (vAdr + vNum) * 3);
    const faces = (m.mesh_face as Int32Array).slice(fAdr * 3, (fAdr + fNum) * 3);
    const geo = new THREE.BufferGeometry();
    geo.setAttribute('position', new THREE.BufferAttribute(verts, 3));
    geo.setIndex(new THREE.BufferAttribute(new Uint32Array(faces), 1));
    geo.computeVertexNormals();
    return geo;
  }

  /**
   * The lighter half ring that shows where the arm can reach (plan 4.4), cut at the area. It
   * is drawn only: it floats 0.4 mm above the table, so a raycast hits the table under it.
   */
  private addMat(): void {
    const [nearX] = tableToWorld(BASE_MM.x, AREA_MM.y); // the area's near edge, world x
    const clip = [new THREE.Plane(new THREE.Vector3(-1, 0, 0), nearX)];
    const ring = new THREE.RingGeometry(
      REACH_MM.min / 1000,
      REACH_MM.max / 1000,
      96,
      1,
      -Math.PI / 2,
      Math.PI,
    );
    const mat = new THREE.Mesh(
      ring,
      new THREE.MeshStandardMaterial({ color: 0xead6b8, roughness: 0.9, clippingPlanes: clip }),
    );
    mat.position.set(0, 0, 0.0004);
    mat.receiveShadow = true;
    this.scene.add(mat);
  }
}
