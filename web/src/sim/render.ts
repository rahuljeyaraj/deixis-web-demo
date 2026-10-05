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
  readonly renderer: THREE.WebGLRenderer;
  readonly scene = new THREE.Scene();
  readonly camera: THREE.PerspectiveCamera;
  /** Meshes the raycast may hit, with the geom id each one draws. */
  readonly pickable: THREE.Object3D[] = [];
  private readonly geoms: { id: number; obj: THREE.Object3D }[] = [];

  constructor(
    readonly canvas: HTMLCanvasElement,
    mj: Mujoco,
    readonly model: MjModel,
    opts: ViewOptions = {},
  ) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, alpha: !!opts.transparent });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.localClippingEnabled = true;
    if (!opts.transparent) this.scene.background = new THREE.Color(0xdfe5ee);

    this.camera = new THREE.PerspectiveCamera(38, 16 / 9, 0.01, 10);
    this.camera.up.set(0, 0, 1);
    this.seatedView();

    this.addLights();
    this.addGeoms(mj);
    this.addMat();
  }

  /** The camera where a seated person's eyes would be, looking at the work area. */
  seatedView(): void {
    const [ex, ey] = tableToWorld(BASE_MM.x, -260);
    const [lx, ly] = tableToWorld(BASE_MM.x, 125);
    this.camera.position.set(ex, ey, 0.36);
    this.camera.lookAt(lx, ly, 0.0);
  }

  resize(width: number, height: number): void {
    this.renderer.setSize(width, height, false);
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
    this.renderer.render(this.scene, this.camera);
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

  /** The lighter half ring that shows where the arm can reach (plan 4.4), cut at the area. */
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
    this.pickable.push(mat);
  }
}
