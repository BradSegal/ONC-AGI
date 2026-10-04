/**
 * The cohort as a field of cubes: one per patient (depth) per measurement (width), height and
 * shade from the measurement's standardised value. Evidence towers stand in front of the field,
 * one per measurement, and a threshold plane crosses them; a tower turns red at the moment it
 * rises through the plane, and a square collar marks the crossing.
 *
 * Positions given to this class are front-relative: x runs along the measurements, y is up and z
 * runs from the field's front edge (0) back through the patients. The camera stands in front.
 *
 * Every animated quantity is a spring towards a target, so any state can follow any other with
 * natural acceleration and settle.
 *
 * This engine renders offline: build/render-media drives it frame by frame at a fixed time step on a
 * GPU and encodes the frames as video, so every visitor sees the same full-quality film whatever
 * their hardware. The page itself never runs WebGL.
 */
import {
  BoxGeometry,
  NeutralToneMapping,
  AmbientLight,
  Color,
  DirectionalLight,
  DoubleSide,
  Fog,
  HemisphereLight,
  InstancedBufferAttribute,
  InstancedMesh,
  Matrix4,
  Mesh,
  MeshBasicMaterial,
  MeshPhysicalMaterial,
  PCFSoftShadowMap,
  PerspectiveCamera,
  PlaneGeometry,
  PMREMGenerator,
  Quaternion,
  RingGeometry,
  Scene,
  ShadowMaterial,
  SRGBColorSpace,
  Vector3,
  WebGLRenderer,
  type BufferGeometry,
} from "three";
import { HalfFloatType, WebGLRenderTarget } from "three";
import { RoundedBoxGeometry } from "three/examples/jsm/geometries/RoundedBoxGeometry.js";
import { EffectComposer } from "three/examples/jsm/postprocessing/EffectComposer.js";
import { RenderPass } from "three/examples/jsm/postprocessing/RenderPass.js";
import { GTAOPass } from "three/examples/jsm/postprocessing/GTAOPass.js";
import { BokehPass } from "three/examples/jsm/postprocessing/BokehPass.js";
import { OutputPass } from "three/examples/jsm/postprocessing/OutputPass.js";
import { RoomEnvironment } from "three/examples/jsm/environments/RoomEnvironment.js";

export type Col = { id: string; group: string; post: boolean };
export type Label = { key: string; text: string; id?: string; cls?: string; at: () => [number, number, number] };
/** A label as the page draws it: position as a fraction of the frame, for one moment of a film. */
export type PlacedLabel = { key: string; text: string; id?: string; cls?: string; x: number; y: number };
export type CameraPose = { pos: [number, number, number]; look: [number, number, number]; fov?: number };

const PAL = {
  ground: new Color("#0f1a2b"),
  raised: new Color("#1a2d48"),
  deep: new Color("#3e5f82"),
  pale: new Color("#c9dae6"),
  blue: new Color("#7fa7c9"),
  ink: new Color("#e9eef2"),
  signal: new Color("#d7263d"),
};
/** A value-to-lightness ramp through three colours, for standardised values in [-2.5, 2.5]. */
const ramp3 = (lo: Color, mid: Color, hi: Color) => (v: number, out = new Color()) => {
  const t = Math.max(0, Math.min(1, (v + 2.5) / 5));
  return t < 0.5 ? out.copy(lo).lerp(mid, t * 2) : out.copy(mid).lerp(hi, (t - 0.5) * 2);
};
const ramp = ramp3(PAL.raised, PAL.deep, PAL.pale);
/** Tint ramps keep each value's lightness, so a tinted column still shows what it measures. */
export const TINT = {
  red: ramp3(new Color("#5a1220"), PAL.signal, new Color("#f2a0aa")),
  ink: ramp3(new Color("#2b3d51"), new Color("#a9bacb"), new Color("#f4f7fb")),
};
export type Tint = (v: number, out?: Color) => Color;
/** Deterministic jitter in [0, 1) from two integers: the same scene every load. */
const hash = (a: number, b: number) => {
  const s = Math.sin(a * 127.1 + b * 311.7) * 43758.5453;
  return s - Math.floor(s);
};
/**
 * The colour to put into the scene so that, after Khronos PBR Neutral tone mapping at `exposure`,
 * it comes out exactly as the page's ground. In the dark range the curve only subtracts an offset
 * set by the darkest channel (x - 6.25x² for x < 0.08), so it inverts in closed form.
 */
function preToneMapped(target: Color, exposure: number): Color {
  const t = [target.r, target.g, target.b];
  const m = Math.sqrt(Math.min(...t) / 6.25);
  const offset = m - 6.25 * m * m;
  return new Color(...(t.map((v) => (v + offset) / exposure) as [number, number, number]));
}
const EXPOSURE = 0.92;
const TOWER_Z = -5; // the tower row stands in front of the field, clear of its front edge

/**
 * Damped springs, one per value. Critically damped for camera and colour; slightly under-damped
 * for heights, so cubes and towers overshoot a touch and settle like physical objects.
 */
class Springs {
  cur: Float32Array;
  vel: Float32Array;
  tgt: Float32Array;
  delay: Float32Array;
  constructor(
    n: number,
    v: number,
    private k: number,
    private c: number,
  ) {
    this.cur = new Float32Array(n).fill(v);
    this.vel = new Float32Array(n);
    this.tgt = new Float32Array(n).fill(v);
    this.delay = new Float32Array(n);
  }
  static of(n: number, v: number, stiffness: number, damping: number) {
    return new Springs(n, v, stiffness, 2 * damping * Math.sqrt(stiffness));
  }
  step(t: number, dt: number): boolean {
    let moving = false;
    const h = dt / 2; // two half-steps keep stiff springs stable at low frame rates
    for (let i = 0; i < this.cur.length; i++) {
      const d = this.tgt[i] - this.cur[i];
      // settled once within 0.002 (scene units or colour channels): far below anything visible
      if (Math.abs(d) < 2e-3 && Math.abs(this.vel[i]) < 2e-3) {
        this.cur[i] = this.tgt[i];
        this.vel[i] = 0;
        continue;
      }
      moving = true;
      if (t < this.delay[i]) continue;
      for (let s = 0; s < 2; s++) {
        this.vel[i] += (this.k * (this.tgt[i] - this.cur[i]) - this.c * this.vel[i]) * h;
        this.cur[i] += this.vel[i] * h;
      }
    }
    return moving;
  }
  snap() {
    this.cur.set(this.tgt);
    this.vel.fill(0);
  }
}

/** Standard material with two additions: hollow cubes keep only their edges, and every cube
 * darkens towards its base, a cheap contact shadow that grounds the field. */
function cubeMaterial(): MeshPhysicalMaterial {
  const m = new MeshPhysicalMaterial({ roughness: 0.46, metalness: 0.04, clearcoat: 0.25, clearcoatRoughness: 0.35, envMapIntensity: 0.3 });
  m.onBeforeCompile = (shader) => {
    shader.vertexShader = shader.vertexShader
      .replace("#include <common>", "#include <common>\nattribute float aHollow;\nattribute float aOcc;\nvarying float vHollow;\nvarying vec2 vCubeUv;\nvarying float vLocalY;\nvarying float vOcc;")
      .replace("#include <begin_vertex>", "#include <begin_vertex>\nvHollow = aHollow;\nvCubeUv = uv;\nvLocalY = position.y;\nvOcc = aOcc;");
    shader.fragmentShader = shader.fragmentShader
      .replace("#include <common>", "#include <common>\nvarying float vHollow;\nvarying vec2 vCubeUv;\nvarying float vLocalY;\nvarying float vOcc;")
      .replace(
        "#include <clipping_planes_fragment>",
        // hollow cubes keep an antialiased frame of their edges
        "#include <clipping_planes_fragment>\nfloat cubeEdge = min(min(vCubeUv.x, 1.0 - vCubeUv.x), min(vCubeUv.y, 1.0 - vCubeUv.y));\nfloat frame = 1.0 - smoothstep(0.0, fwidth(cubeEdge) * 1.5, cubeEdge - 0.06);\nif (vHollow > 0.5 && frame < 0.02) discard;",
      )
      // contact shading: darker towards the base, more so where taller neighbours crowd the cube
      .replace("#include <color_fragment>", "#include <color_fragment>\ndiffuseColor.rgb *= mix(0.42 + 0.38 * (1.0 - vOcc), 1.0, smoothstep(0.0, 0.6 + 0.3 * vOcc, vLocalY));\nif (vHollow > 0.5) diffuseColor.a *= frame;");
  };
  return m;
}

export class CohortScene {
  readonly cols: Col[];
  readonly rows: number;
  readonly colX: number[];
  private renderer: WebGLRenderer;
  private scene = new Scene();
  private camera = new PerspectiveCamera(30, 1, 0.5, 400);
  private field: InstancedMesh;
  private strip: InstancedMesh;
  private towers: InstancedMesh;
  private collars: InstancedMesh;
  private plane: Mesh;
  private planeEdge: Mesh;
  private planeBack: Mesh;
  private pending: { due: number; apply: () => void }[] = [];
  private fieldDim = Springs.of(1, 1, 20, 1); // field brightness, lowered while towers carry the story
  private fieldScale = Springs.of(1, 1, 20, 1); // field height multiplier
  private key: DirectionalLight;
  private fog: Fog;
  private m = new Matrix4();
  private q = new Quaternion();
  private v = new Vector3();
  private sc = new Vector3();
  private c = new Color();
  time = 0;
  private drift = 0;
  private dirty = true;
  private labels: Label[] = [];

  private onFrame?: (t: number) => void;

  private hollow: Float32Array;
  private baseRGB: Float32Array; // value shade per cube before tinting and ghosting
  private towerBase: Float32Array;
  private tints = new Map<number, Tint>();
  private zValues: number[][] = [];
  // springs
  private fieldH: Springs;
  private fieldY: Springs; // vertical offset: cubes drop in, bought rows lift and settle
  private fieldZ: Springs;
  private fieldX: Springs;
  private ghost: Springs;
  private fieldRGB: Springs;
  private stripH: Springs;
  private towerH: Springs;
  private towerRGB: Springs;
  private collar: Springs;
  private planeY = Springs.of(1, 0, 40, 1);
  private planeOn = Springs.of(1, 0, 30, 1);
  private fogFar = Springs.of(1, 40, 11, 1);
  private camPos = Springs.of(3, 0, 7, 1);
  private camLook = Springs.of(3, 0, 7, 1);
  private camFov = Springs.of(1, 30, 7, 1);
  private fogRange: [number, number];
  private groundIn: Color;
  private composer!: EffectComposer;
  private width: number;
  private height: number;
  private bokeh!: BokehPass;
  private shadowDirty = true;
  private driftPeriod = 40;
  private driftStart = 0;
  private focus = Springs.of(3, 0, 6, 1); // the rack-focus subject, front-relative
  private crest: { start: number; span: number } | null = null;
  towerScale = 0.9;
  line: number | null = null;

  constructor(
    readonly canvas: HTMLCanvasElement,
    cols: Col[],
    rows: number,
    opts: { width: number; height: number; fog: [number, number]; shadowDepth?: number },
  ) {
    this.cols = cols;
    this.rows = rows;
    this.width = opts.width;
    this.height = opts.height;
    this.fogRange = opts.fog;
    // measurement columns, with a gap between kinds like bands on a chromosome
    let x = 0;
    this.colX = cols.map((c, i) => {
      if (i > 0 && c.group !== cols[i - 1].group) x += 0.7;
      const at = x;
      x += 1;
      return at;
    });
    const width = x;
    const C = cols.length;
    const n = rows * C;

    // offline: the drawing buffer is kept so each finished frame can be handed to the video encoder
    const r = (this.renderer = new WebGLRenderer({ canvas, antialias: true, alpha: false, preserveDrawingBuffer: true, powerPreference: "high-performance" }));
    r.setPixelRatio(1);
    r.setSize(this.width, this.height, false);
    this.camera.aspect = this.width / this.height;
    this.camera.updateProjectionMatrix();
    r.setClearColor(0x000000, 0);
    r.toneMapping = NeutralToneMapping;
    r.toneMappingExposure = EXPOSURE;
    r.outputColorSpace = SRGBColorSpace;
    r.shadowMap.enabled = true;
    r.shadowMap.type = PCFSoftShadowMap;
    r.shadowMap.autoUpdate = false; // shadows re-render only when the geometry moves
    const pmrem = new PMREMGenerator(r);
    this.scene.environment = pmrem.fromScene(new RoomEnvironment(), 0.04).texture;
    // fog and background are pre-compensated, so after tone mapping they match the page exactly
    this.groundIn = preToneMapped(PAL.ground, EXPOSURE);
    this.fog = new Fog(this.groundIn, opts.fog[0], opts.fog[1]);
    this.scene.fog = this.fog;

    // light: a cool key from front-left above, a blue rim from behind, a dim sky fill
    this.key = new DirectionalLight(0xf4f7fb, 2.4);
    this.key.castShadow = true;
    const depth = Math.min(rows, opts.shadowDepth ?? 70);
    Object.assign(this.key.shadow.camera, { left: -depth * 0.7, right: depth * 0.7, top: depth * 0.75, bottom: -depth * 0.75, near: 1, far: 220 });
    this.key.shadow.mapSize.set(4096, 4096);
    this.key.shadow.bias = -0.0004;
    this.key.shadow.normalBias = 0.02;
    this.key.shadow.radius = 4;
    this.key.position.set(-22, 18, this.wz(-24)); // low enough for long, readable cast shadows
    this.key.target.position.set(width / 2, 0, this.wz(depth / 2));
    const rim = new DirectionalLight(0x7fa7c9, 1.2);
    rim.position.set(width + 10, 12, this.wz(rows + 30));
    this.scene.add(this.key, this.key.target, rim, new HemisphereLight(0x9fb6cc, 0x0a1220, 0.08), new AmbientLight(0x0f1a2b, 0.1));

    // shadows land on the page's own ground: an invisible plane that only darkens
    const floor = new Mesh(new PlaneGeometry(width + 60, rows + 80).rotateX(-Math.PI / 2), new ShadowMaterial({ opacity: 0.55, color: 0x050a12 }));
    floor.position.set(width / 2, 0, this.wz(rows / 2));
    floor.receiveShadow = true;
    this.scene.add(floor);

    const cube = new RoundedBoxGeometry(1, 1, 1, 3, 0.12).translate(0, 0.5, 0);
    const instanced = (geometry: BufferGeometry, count: number) => {
      const g = geometry.clone();
      g.setAttribute("aHollow", new InstancedBufferAttribute(new Float32Array(count), 1));
      g.setAttribute("aOcc", new InstancedBufferAttribute(new Float32Array(count), 1));
      const material = cubeMaterial();
      material.alphaToCoverage = true; // hollow edges antialias under MSAA without sorting
      const mesh = new InstancedMesh(g, material, count);
      mesh.instanceColor = new InstancedBufferAttribute(new Float32Array(count * 3), 3);
      mesh.castShadow = true;
      mesh.receiveShadow = true;
      mesh.frustumCulled = false;
      this.scene.add(mesh);
      return mesh;
    };
    this.field = instanced(cube, n);
    this.strip = instanced(cube, rows);
    // towers are tall and thin: a sharp box keeps them square, where a scaled bevel would read as a capsule
    this.towers = instanced(new BoxGeometry(1, 1, 1).translate(0, 0.5, 0), C);
    this.hollow = new Float32Array(C);
    this.baseRGB = new Float32Array(n * 3);
    this.towerBase = new Float32Array(C * 3);

    // collars: a flat square frame where a tower pierces the threshold
    const ring = new RingGeometry(0.72, 0.86, 4, 1).rotateZ(Math.PI / 4).rotateX(-Math.PI / 2);
    this.collars = new InstancedMesh(ring, new MeshBasicMaterial({ color: PAL.signal, side: DoubleSide, transparent: true, opacity: 0.95, depthWrite: false }), C);
    this.collars.frustumCulled = false;
    this.scene.add(this.collars);

    this.fieldH = Springs.of(n, 0, 70, 0.7);
    this.fieldY = Springs.of(n, 0, 60, 0.62);
    this.fieldZ = Springs.of(rows, 0, 38, 0.9);
    this.fieldX = Springs.of(C, 0, 38, 0.9);
    this.ghost = Springs.of(rows, 0, 30, 1);
    this.fieldRGB = Springs.of(n * 3, 0, 40, 1);
    this.stripH = Springs.of(rows, 0, 60, 0.65);
    this.towerH = Springs.of(C, 0, 55, 0.62);
    this.towerRGB = Springs.of(C * 3, 0, 90, 1);
    this.collar = Springs.of(C, 0, 80, 0.55);
    for (let rr = 0; rr < rows; rr++) this.fieldZ.cur[rr] = this.fieldZ.tgt[rr] = rr;
    this.colX.forEach((cx, ci) => (this.fieldX.cur[ci] = this.fieldX.tgt[ci] = cx));
    this.fogFar.cur[0] = this.fogFar.tgt[0] = opts.fog[1];

    // the threshold: a translucent plane across the tower row, with a hairline front edge
    const pw = width + 1.6;
    this.plane = new Mesh(new PlaneGeometry(pw, 1.3).rotateX(-Math.PI / 2), new MeshBasicMaterial({ color: PAL.signal, transparent: true, opacity: 0.18, side: DoubleSide, depthWrite: false }));
    const hairlines = new PlaneGeometry(pw, 0.045).rotateX(-Math.PI / 2);
    this.planeEdge = new Mesh(hairlines, new MeshBasicMaterial({ color: PAL.signal, transparent: true, depthWrite: false }));
    this.planeBack = new Mesh(hairlines, this.planeEdge.material);
    this.plane.position.set(width / 2, 0, this.wz(TOWER_Z));
    this.planeEdge.position.set(width / 2, 0, this.wz(TOWER_Z - 0.65));
    this.planeBack.position.set(width / 2, 0, this.wz(TOWER_Z + 0.65));
    this.scene.add(this.plane, this.planeEdge, this.planeBack);

    this.buildComposer();
    this.renderer.compile(this.scene, this.camera);
  }

  /**
   * The film finish: ambient occlusion in the crevices between cubes, then a shallow depth of
   * field racked to each beat's subject, then tone mapping. The frame is opaque here, so the
   * background carries the pre-compensated ground and leaves the pass chain as the page's colour.
   */
  private buildComposer() {
    const w = this.width;
    const h = this.height;
    const target = new WebGLRenderTarget(w, h, { type: HalfFloatType, samples: 4 });
    const composer = new EffectComposer(this.renderer, target);
    this.scene.background = this.groundIn;
    composer.addPass(new RenderPass(this.scene, this.camera));
    const ao = new GTAOPass(this.scene, this.camera, w, h);
    ao.updateGtaoMaterial({ radius: 0.7, distanceExponent: 1.4, thickness: 1.2, scale: 1.1, samples: 16 });
    ao.blendIntensity = 0.85;
    composer.addPass(ao);
    this.bokeh = new BokehPass(this.scene, this.camera, { focus: 30, aperture: 0.0011, maxblur: 0.005 });
    composer.addPass(this.bokeh);
    composer.addPass(new OutputPass());
    this.composer = composer;
  }

  /** Rack focus to a front-relative point. */
  focusOn(point: [number, number, number]) {
    this.focus.tgt.set(point);
  }

  /* ---------------------------------------------------------------- state setters */

  /** Field heights and shades from a patients × measurements matrix of standardised values.
   * `wave` delays each row by its depth, so a change sweeps away from the viewer. */
  setValues(z: number[][], wave = 0) {
    this.zValues = z;
    const C = this.cols.length;
    for (let r = 0; r < this.rows; r++)
      for (let c = 0; c < C; c++) {
        const i = r * C + c;
        const v = z[r]?.[c] ?? 0;
        this.fieldH.tgt[i] = 0.14 + 0.82 * Math.max(0, Math.min(1, (v + 2.5) / 5));
        ramp(v, this.c);
        this.baseRGB.set([this.c.r, this.c.g, this.c.b], i * 3);
        this.fieldH.delay[i] = this.time + wave * (this.fieldZ.tgt[r] / this.rows) + 0.004 * c;
      }
    this.recolour(wave);
  }

  /** One world gives way to another as a wave: each row collapses flat, then regrows to the new
   * values, front row first. `onCrest` fires as the wave passes the middle of the field. */
  swapValues(z: number[][], span = 1.6, onCrest?: () => void) {
    const C = this.cols.length;
    for (let r = 0; r < this.rows; r++) {
      const d = this.time + span * (this.fieldZ.tgt[r] / this.rows);
      for (let c = 0; c < C; c++) {
        const i = r * C + c;
        this.fieldH.tgt[i] = 0.1;
        this.fieldH.delay[i] = d;
      }
    }
    // each row holds flat for a beat before the new world rises in it
    this.pending.push({ due: this.time + 0.75, apply: () => this.setValues(z, span) });
    this.crest = { start: this.time, span };
    if (onCrest) this.pending.push({ due: this.time + span * 0.5 + 0.5, apply: onCrest });
  }

  /** Field emphasis: dim and flatten the cohort while the towers carry the story. */
  setFieldEmphasis(brightness: number, height: number) {
    this.fieldDim.tgt[0] = brightness;
    this.fieldScale.tgt[0] = height;
  }

  /** Cubes fall into place row by row, front first: the cohort being assembled. */
  assemble(span = 1.4) {
    const C = this.cols.length;
    for (let r = 0; r < this.rows; r++)
      for (let c = 0; c < C; c++) {
        const i = r * C + c;
        this.fieldY.cur[i] = 6 + 1.5 * hash(r, c);
        this.fieldY.vel[i] = 0;
        this.fieldY.tgt[i] = 0;
        this.fieldY.delay[i] = this.time + span * (this.fieldZ.tgt[r] / this.rows) + 0.06 * hash(c, r);
      }
  }

  /** Tint whole columns (the planted pair) over their value shades; null clears. */
  tintColumns(tints: Record<number, Tint> | null) {
    this.tints = new Map(Object.entries(tints ?? {}).map(([k, v]) => [Number(k), v]));
    this.recolour(0);
  }

  private recolour(wave: number) {
    const C = this.cols.length;
    for (let r = 0; r < this.rows; r++)
      for (let c = 0; c < C; c++) {
        const i = r * C + c;
        const tint = this.tints.get(c);
        if (tint) tint(this.zValues[r]?.[c] ?? 0, this.c);
        else this.c.setRGB(this.baseRGB[i * 3], this.baseRGB[i * 3 + 1], this.baseRGB[i * 3 + 2]).multiplyScalar(this.tints.size ? 0.6 : 1);
        this.fieldRGB.tgt.set([this.c.r, this.c.g, this.c.b], i * 3);
        // tints run down their column from the front, like a highlighter
        const d = this.time + wave * (this.fieldZ.tgt[r] / this.rows) + (tint ? 0.5 * (this.fieldZ.tgt[r] / this.rows) : 0);
        this.fieldRGB.delay[i * 3] = this.fieldRGB.delay[i * 3 + 1] = this.fieldRGB.delay[i * 3 + 2] = d;
      }
  }

  /** Row order: order[r] is the depth position of row r (null restores the original order). */
  setOrder(order: number[] | null, stagger = 0.5) {
    for (let r = 0; r < this.rows; r++) {
      const to = order ? order[r] : r;
      if (to !== this.fieldZ.tgt[r]) this.fieldZ.delay[r] = this.time + stagger * (to / this.rows);
      this.fieldZ.tgt[r] = to;
    }
  }

  /** How faded each row is: 1 for patients not yet bought. Newly bought rows lift and settle. */
  setGhost(ghost: (r: number) => number, stagger = 0) {
    const C = this.cols.length;
    for (let r = 0; r < this.rows; r++) {
      const g = ghost(r);
      const d = this.time + stagger * (this.fieldZ.tgt[r] / this.rows);
      if (g < this.ghost.tgt[r] - 0.5)
        for (let c = 0; c < C; c++) {
          const i = r * C + c;
          this.fieldY.cur[i] = 1.6;
          this.fieldY.vel[i] = 0;
          this.fieldY.tgt[i] = 0;
          this.fieldY.delay[i] = d;
        }
      this.ghost.tgt[r] = g;
      this.ghost.delay[r] = d;
    }
  }

  /** Post-outcome columns slide aside and turn hollow. */
  setPost(apart: boolean) {
    this.shadowDirty = true;
    this.cols.forEach((c, i) => {
      this.fieldX.tgt[i] = this.colX[i] + (apart && c.post ? 1.6 : 0);
      this.fieldX.delay[i] = this.time + (c.post ? 0.1 * (i % 3) : 0);
      this.hollow[i] = apart && c.post ? 1 : 0;
    });
    this.dirty = true;
  }

  setOutcome(outcome: number[] | null) {
    for (let r = 0; r < this.rows; r++) {
      this.stripH.tgt[r] = outcome ? (outcome[r] ? 1.1 : 0.22) : 0;
      this.stripH.delay[r] = this.time + 0.6 * (r / this.rows);
    }
  }

  /**
   * Evidence towers (-log10 p per measurement) and the line they are judged against. "benchmark"
   * draws the red threshold; "agent" draws the agent's own line in ink. A tower turns red, and a
   * collar appears, only while its current height is above the line.
   */
  setTowers(evidence: number[] | null, line: number | null, kind: "benchmark" | "agent" = "benchmark") {
    const ink = kind === "agent";
    (this.plane.material as MeshBasicMaterial).color.copy(ink ? PAL.ink : PAL.signal);
    (this.plane.material as MeshBasicMaterial).opacity = ink ? 0.06 : 0.16;
    (this.planeEdge.material as MeshBasicMaterial).color.copy(ink ? PAL.ink : PAL.signal);
    this.line = evidence && line != null ? line : null;
    this.planeY.tgt[0] = (line ?? 0) * this.towerScale;
    this.planeOn.tgt[0] = this.line == null ? 0 : 1;
    const groups = [...new Set(this.cols.map((k) => k.group))];
    this.cols.forEach((c, i) => {
      const v = evidence ? evidence[i] : 0;
      this.towerH.tgt[i] = evidence ? Math.max(0.06, v * this.towerScale) : 0;
      this.towerH.delay[i] = this.time + 0.035 * i;
      const base = c.post ? PAL.blue : groups.indexOf(c.group) % 2 ? this.c.copy(PAL.deep).lerp(PAL.blue, 0.4) : PAL.blue;
      this.towerBase.set([base.r, base.g, base.b], i * 3);
    });
  }

  /** Lower every tower to the floor while the line stays, so each one drains of red only at the
   * moment it actually passes back through the threshold. */
  lowerTowers() {
    for (let i = 0; i < this.cols.length; i++) {
      this.towerH.tgt[i] = 0;
      this.towerH.delay[i] = this.time + 0.02 * i;
    }
  }

  /** True once every tower stands below the line. */
  allBelowLine() {
    const y = this.planeY.cur[0];
    return this.towerH.cur.every((h) => h <= y + 0.005);
  }

  setCamera(pose: CameraPose) {
    this.camPos.tgt.set(pose.pos);
    this.camLook.tgt.set(pose.look);
    this.camFov.tgt[0] = pose.fov ?? 30;
  }

  /** Jump the camera to a pose without travelling (first paint). */
  placeCamera(pose: CameraPose) {
    this.setCamera(pose);
    for (const s of [this.camPos, this.camLook, this.camFov]) s.snap();
  }

  /** Fog pulls back from near to its resting distance: the scene emerging from the dark. */
  reveal(from = 6) {
    this.fogFar.cur[0] = from;
    this.fogFar.vel[0] = 0;
    this.fogFar.tgt[0] = this.fogRange[1];
  }

  /** A slow periodic camera sway, starting at zero at `start` and returning to it every `period`
   * seconds, so a loop that lasts a whole number of periods closes without a seam. */
  setDrift(amount: number, period = 40, start = this.time) {
    this.drift = amount;
    this.driftPeriod = period;
    this.driftStart = start;
  }

  setLabels(labels: Label[]) {
    this.labels = labels;
  }

  /** The current labels projected into the frame, as fractions of its width and height. */
  placedLabels(): PlacedLabel[] {
    const out: PlacedLabel[] = [];
    for (const l of this.labels) {
      const [x, y, z] = l.at();
      this.v.set(x, y, this.wz(z)).project(this.camera);
      if (this.v.z > 1) continue;
      out.push({ key: l.key, text: l.text, id: l.id, cls: l.cls, x: +((this.v.x + 1) / 2).toFixed(4), y: +((1 - this.v.y) / 2).toFixed(4) });
    }
    return out;
  }

  frame(fn: (t: number) => void) {
    this.onFrame = fn;
  }

  /** Jump straight to the targets (reduced motion). */
  snap() {
    this.dirty = true;
    this.shadowDirty = true;
    for (const s of this.all()) s.snap();
  }

  private all() {
    return [this.focus, this.fieldH, this.fieldY, this.fieldZ, this.fieldX, this.ghost, this.fieldRGB, this.fieldDim, this.fieldScale, this.stripH, this.towerH, this.towerRGB, this.collar, this.planeY, this.planeOn, this.fogFar, this.camPos, this.camLook, this.camFov];
  }

  /* ---------------------------------------------------------------- label anchors (front-relative) */

  towerTop(i: number): [number, number, number] {
    return [this.colX[i] + 0.5, this.towerH.cur[i] + 0.2, TOWER_Z];
  }
  columnHead(i: number): [number, number, number] {
    return [this.fieldX.cur[i] + 0.5, 0.55, this.rows + 0.4];
  }
  /** Each data type's band of columns: its name and centre, for labelling what the cubes measure. */
  groups(): { group: string; x: number }[] {
    const out: { group: string; x: number }[] = [];
    this.cols.forEach((c) => {
      const last = out.at(-1);
      if (last && last.group === c.group) return;
      const span = this.cols.map((k, j) => (k.group === c.group ? j : -1)).filter((j) => j >= 0);
      out.push({ group: c.group, x: (this.fieldX.cur[span[0]] + this.fieldX.cur[span.at(-1)!]) / 2 + 0.5 });
    });
    return out;
  }

  /** The threshold's current height, for anchoring its label. */
  lineY() {
    return this.planeY.cur[0];
  }
  /** The left end of the threshold plane. */
  planeLabel(): [number, number, number] {
    return [-0.8, this.planeY.cur[0], TOWER_Z]; // the plane's left edge, so the leader meets it
  }

  /* ---------------------------------------------------------------- render loop */

  /** Front-relative depth to world z: the camera looks down -z, so measurement columns read left to right. */
  private wz(z: number) {
    return this.rows - z;
  }

  /**
   * Advance the scene by one fixed step and draw it. Returns whether anything is still moving,
   * so a clip can end when its scene has settled.
   */
  advance(dt: number): boolean {
    this.time += dt;
    this.onFrame?.(this.time);
    for (const p of this.pending.filter((q) => q.due <= this.time)) p.apply();
    this.pending = this.pending.filter((q) => q.due > this.time);
    // towers are red, and collared, only while above the line
    const C = this.cols.length;
    const lineY = this.planeY.cur[0];
    for (let i = 0; i < C; i++) {
      const over = this.line != null && !this.cols[i].post && this.towerH.cur[i] > lineY + 0.02;
      if (over) this.towerRGB.tgt.set([PAL.signal.r, PAL.signal.g, PAL.signal.b], i * 3);
      else this.towerRGB.tgt.set(this.towerBase.subarray(i * 3, i * 3 + 3), i * 3);
      this.collar.tgt[i] = over ? 1 : 0;
    }
    const step = (ss: Springs[]) => ss.map((s) => s.step(this.time, dt)).some(Boolean);
    const swapping = !!this.crest && this.time - this.crest.start < this.crest.span + 1;
    if (this.crest && !swapping) this.crest = null;
    const fieldMoved = step([this.fieldH, this.fieldY, this.fieldZ, this.fieldX, this.ghost, this.fieldRGB, this.fieldDim, this.fieldScale]) || swapping;
    const stripMoved = step([this.stripH]);
    const towersMoved = step([this.towerH, this.towerRGB, this.collar, this.planeY, this.planeOn]);
    const scene = step([this.fogFar, this.camPos, this.camLook, this.camFov, this.focus]);
    if (fieldMoved || stripMoved || towersMoved || this.shadowDirty || this.dirty) {
      this.renderer.shadowMap.needsUpdate = true;
      this.shadowDirty = false;
    }
    if (fieldMoved || this.dirty) this.writeField();
    if (stripMoved || fieldMoved || this.dirty) this.writeStrip();
    if (towersMoved || this.dirty) this.writeTowers();
    this.dirty = false;
    this.writeCamera();
    const f = this.focus.cur;
    this.v.set(f[0], f[1], this.wz(f[2])).applyMatrix4(this.camera.matrixWorldInverse);
    (this.bokeh.uniforms as Record<string, { value: number }>).focus.value = Math.max(1, -this.v.z);
    this.composer.render(dt);
    return fieldMoved || stripMoved || towersMoved || scene || this.pending.length > 0;
  }

  /** The hot loop: written with plain arithmetic straight into the instance buffers, because on a
   * slow CPU this per-frame rewrite of every cube, not the GPU, sets the frame rate. */
  private writeField() {
    const C = this.cols.length;
    const s = 0.82;
    const M = this.field.instanceMatrix.array as Float32Array;
    const hollowA = (this.field.geometry.getAttribute("aHollow") as InstancedBufferAttribute).array as Float32Array;
    const occA = (this.field.geometry.getAttribute("aOcc") as InstancedBufferAttribute).array as Float32Array;
    const colA = this.field.instanceColor!.array as Float32Array;
    const dim = this.fieldDim.cur[0];
    const hs = this.fieldScale.cur[0];
    const H = this.fieldH.cur;
    const Y = this.fieldY.cur;
    const X = this.fieldX.cur;
    const RGB = this.fieldRGB.cur;
    const gr = PAL.ground.r, gg = PAL.ground.g, gb = PAL.ground.b;
    // the crest of a world swap: a band of rows a touch brighter as the wave passes
    const crestRow = this.crest ? ((this.time - this.crest.start) / this.crest.span) * this.rows : -1e9;
    for (let r = 0; r < this.rows; r++) {
      const g = this.ghost.cur[r];
      const zr = this.fieldZ.cur[r];
      const z = this.wz(zr + 0.5);
      const dc = (zr - crestRow) / 5;
      const k = dim * (1 + 0.45 * Math.exp(-dc * dc));
      const fade = g * 0.8;
      for (let c = 0; c < C; c++) {
        const i = r * C + c;
        // patients not yet bought flatten to low, dark tiles; bought ones stand at their value
        const h = Math.max(0.001, (H[i] * (1 - g) + 0.1 * g) * hs);
        const m = i * 16;
        M[m] = s; M[m + 1] = 0; M[m + 2] = 0; M[m + 3] = 0;
        M[m + 4] = 0; M[m + 5] = h; M[m + 6] = 0; M[m + 7] = 0;
        M[m + 8] = 0; M[m + 9] = 0; M[m + 10] = s; M[m + 11] = 0;
        M[m + 12] = X[c] + 0.5; M[m + 13] = Y[i]; M[m + 14] = z; M[m + 15] = 1;
        const q = i * 3;
        colA[q] = RGB[q] * k + (gr - RGB[q] * k) * fade;
        colA[q + 1] = RGB[q + 1] * k + (gg - RGB[q + 1] * k) * fade;
        colA[q + 2] = RGB[q + 2] * k + (gb - RGB[q + 2] * k) * fade;
        hollowA[i] = this.hollow[c];
        // occlusion: how much taller the left and right neighbours stand
        const left = c > 0 ? H[i - 1] : H[i];
        const right = c < C - 1 ? H[i + 1] : H[i];
        const o = (Math.max(0, left - H[i]) + Math.max(0, right - H[i])) * 0.9;
        occA[i] = o > 1 ? 1 : o;
      }
    }
    this.field.instanceMatrix.needsUpdate = true;
    this.field.instanceColor!.needsUpdate = true;
    (this.field.geometry.getAttribute("aHollow") as InstancedBufferAttribute).needsUpdate = true;
    (this.field.geometry.getAttribute("aOcc") as InstancedBufferAttribute).needsUpdate = true;
  }

  private writeStrip() {
    const col = this.strip.instanceColor!;
    for (let r = 0; r < this.rows; r++) {
      const h = this.stripH.cur[r];
      this.m.makeScale(0.82, Math.max(0.001, h), 0.82).setPosition(-1.3, 0, this.wz(this.fieldZ.cur[r] + 0.5));
      this.strip.setMatrixAt(r, this.m);
      const c = h > 0.6 ? PAL.ink : PAL.deep;
      col.setXYZ(r, c.r, c.g, c.b);
    }
    this.strip.visible = this.stripH.cur.some((h) => h > 0.002);
    this.strip.instanceMatrix.needsUpdate = col.needsUpdate = true;
  }

  private writeTowers() {
    const C = this.cols.length;
    const col = this.towers.instanceColor!;
    const hollow = this.towers.geometry.getAttribute("aHollow") as InstancedBufferAttribute;
    const lineY = this.planeY.cur[0];
    for (let i = 0; i < C; i++) {
      const x = this.colX[i] + 0.5;
      this.m.makeScale(0.64, Math.max(0.001, this.towerH.cur[i]), 0.64).setPosition(x, 0, this.wz(TOWER_Z));
      this.towers.setMatrixAt(i, this.m);
      col.setXYZ(i, this.towerRGB.cur[i * 3], this.towerRGB.cur[i * 3 + 1], this.towerRGB.cur[i * 3 + 2]);
      hollow.setX(i, this.cols[i].post ? 1 : 0);
      // the collar snaps in slightly wide and closes onto the tower
      const k = this.collar.cur[i];
      const scale = Math.max(0.001, k) * (1 + 0.6 * (1 - Math.min(1, k)));
      this.m.compose(this.v.set(x, lineY + 0.01, this.wz(TOWER_Z)), this.q.identity(), this.sc.set(scale, 1, scale));
      this.collars.setMatrixAt(i, this.m);
    }
    this.towers.visible = this.towerH.cur.some((h) => h > 0.002);
    this.towers.instanceMatrix.needsUpdate = col.needsUpdate = hollow.needsUpdate = this.collars.instanceMatrix.needsUpdate = true;
    const on = this.planeOn.cur[0];
    this.plane.position.y = this.planeEdge.position.y = this.planeBack.position.y = lineY;
    this.plane.visible = this.planeEdge.visible = this.planeBack.visible = this.collars.visible = on > 0.02;
    (this.planeEdge.material as MeshBasicMaterial).opacity = on;
    (this.collars.material as MeshBasicMaterial).opacity = 0.95 * on;
  }

  private writeCamera() {
    const p = this.camPos.cur;
    const l = this.camLook.cur;
    const phase = (2 * Math.PI * (this.time - this.driftStart)) / this.driftPeriod;
    const sway = this.drift ? Math.sin(phase) * this.drift : 0;
    const lift = this.drift ? Math.sin(2 * phase) * this.drift * 0.18 : 0;
    this.camera.position.set(p[0] + sway, p[1] + lift, this.wz(p[2]));
    this.camera.lookAt(this.v.set(l[0], l[1], this.wz(l[2])));
    this.camera.updateMatrixWorld();
    if (Math.abs(this.camera.fov - this.camFov.cur[0]) > 1e-3) {
      this.camera.fov = this.camFov.cur[0];
      this.camera.updateProjectionMatrix();
    }
    this.fog.far = this.fogFar.cur[0];
    this.fog.near = Math.min(this.fogRange[0], this.fog.far * 0.4);
  }


}
