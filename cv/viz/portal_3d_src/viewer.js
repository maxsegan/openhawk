// viewer.js — WebGL 3D point-reconstruction viewer (three.js r160, vendored locally).
// Renders one exported point (export_point_3d.py JSON) as a pan-around scene: regulation
// court + net, ball flight tube coloured by speed/height/spin, bounce marks, contact markers
// (serve / rally / dead tiers), metric camera-ray player skeletons, plus physics-plausibility
// overlays a human eyeballs instantly (apex heights, net clearance, human-reach band, spin
// arrows). Missing / low-confidence spans are rendered honestly (dashed / faded), never
// smoothed over.
//
// World mapping is right-handed: court (x,y,z) -> three (-x,z,y). Three's +Y is court height;
// from the near-baseline camera, court x=0 stays broadcast-left instead of being reflected.

import * as THREE from 'three';
import { OrbitControls } from './vendor/OrbitControls.js';
import Core from './viewer_core.js';

const V = (cx, cy, cz) => new THREE.Vector3(...Core.courtToWorld(cx, cy, cz));

// ---- palette -------------------------------------------------------------------------
const COL = {
  surface: 0x0f1115, court: 0x1d3a2a, line: 0xcfd4cc, net: 0xe8eae6,
  near: 0x6f63e6, far: 0x1fb98a, ball: 0xf2b705, ballEdge: 0x111111,
  bounceIn: 0x3fb27f, bounceOut: 0xe0503f, serve: 0xffd23f, rally: 0xe34948,
  dead: 0x8a8a8a, apex: 0xffd23f, netClear: 0x59c3ff, netImpact: 0xff7ac2,
  reach: 0x59c3ff, spin: 0xff7ac2,
  gap: 0x6b7280,
};
const SKELETON_EDGES = [
  ['nose', 'left_shoulder'], ['nose', 'right_shoulder'],
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
  ['racket_grip', 'racket_head'],
];

let renderer, scene, camera, controls, clock;
let doc = null;                 // current point document
let dynamicGroup = null;        // rebuilt per point
let ball, ballGap, spinArrow, netCrossGroup;
let players = { near: null, far: null };
let playhead = { t: 0, playing: false, speed: 1.0 };
let metric = 'speed';           // speed | height | spin
let overlays = {
  apex: true, netclear: true, reach: false, spin: true, missing: true, players: true,
  family: true,
};
const DEFAULT_COURT = { width: 10.97, singles_inset: 1.37 };
const DEFAULT_SINGLES_NET = Core.singlesNetGeometry(
  DEFAULT_COURT.width,
  DEFAULT_COURT.singles_inset,
);
let netHeightFn = Core.makeSinglesNetHeightFn(
  DEFAULT_COURT.width,
  DEFAULT_COURT.singles_inset,
  0.914,
  1.07,
);
let timeBounds = { lo: 0, hi: 1 };

// Follow (progressive-reveal) mode: the full-point spaghetti collapses to a progressive reveal —
// the flight draws only up to the playhead, a recent window stays full-opacity, older stretches
// fade to a low ghost, and future shots + their markers/labels are hidden entirely. This is the
// DEFAULT state for long points (>8 shots or >8 s) on LOAD and while SCRUBBING — not just during
// play — so a 31-shot rally is followable instead of an unreadable tangle. A clearly-labelled
// "Follow" toggle flips it (off restores the classic full view); the choice persists per session.
// We never rebuild tube geometry for this: buildTrajectory records each vertex's time + an RGBA
// colour attribute, and updateTrailVisual() rewrites only the alpha channel per frame.
// recent: seconds behind the playhead kept at the bright window; full: the bright-window alpha
// (kept BELOW 1 on purpose so overlapping tubes read as translucent layers, not opaque solids);
// ghost: the low floor older shots fade to (dropped from 0.18 — 0.18 over 30 shots was still a
// tangle); fade: the transition band between them.
const DEFAULT_TRAIL = { active: false, recent: 1.5, ghost: 0.06, fade: 0.6, full: 0.75 };
let trail = { ...DEFAULT_TRAIL };
// Annotation text chips (apex height / net clearance / contact speed) are pure clutter in a long
// rally and add no value the owner wants — so they are OFF by default (zero text in the scene).
// A single "labels" toggle brings them back for anyone who wants the numbers. Player name chips
// (near/far) are orientation, not clutter, so they always show.
let showLabels = false;
// Suppressed / low-trust clutter means dashed gap bridges and diagnostic centrelines. Rejected
// point tubes remain visible but translucent: the point-level banner and low-confidence frame
// styling communicate the gate honestly without making a useful reconstruction disappear.
let showSuppressed = false;
const FOLLOW_PREF_KEY = 'portal3d.follow';
function readFollowPref() { try { return sessionStorage.getItem(FOLLOW_PREF_KEY); } catch (e) { return null; } }
function writeFollowPref(v) { try { sessionStorage.setItem(FOLLOW_PREF_KEY, v); } catch (e) { /* private mode */ } }
let trailTubes = [];   // { mesh, colorAttr, vertTimes:Float32Array, rejected }
let trailMarks = [];   // { t, mats:[{mat, baseOpacity, baseTransparent}] } — 3D markers/lines

// broadcast reference-monitor state
let videoOn = true;
let lastFrameShown = null;      // frame currently displayed
let desiredFrame = null;        // most-recent frame the playhead wants
let videoLoading = false;       // a frame decode is in flight (double-buffer)
let videoFront = 0;             // which of the two <img> buffers is currently shown
let selectedFlight = null;      // flight-review deep link, seeked on load
let navState = { set: '', match: '', point: '', query: '' };
let readoutFrame = null;
let flightEntries = [];
let lastIndex = null;
let videoGeneration = 0;        // invalidates callbacks from the previously selected point
let frameWarm = new Set();      // URLs already prefetched (Image() warming)
let frameWarmCap = Infinity;    // last frame known to exist on disk (learned from 404s)
let videoOverlayByFrame = new Map(); // native frame -> labeled front + fitted projection
let servePoseByFrame = new Map();    // native frame -> pose overlay + side elevation
let indexPoints = [];           // ordered index entries (accepted first)
let activeEntry = null;
let originalSource = null;
let selectedSpan = null;
let pointGeneration = 0;
let spanGeneration = 0;
let CURRENT_COHORT = false;
let COHORT_LABEL = 'Current S6';         // current point or shot index entry
let EMBED = false;              // true when hosted in the compare (2D+3D) view
let SHOT_REVIEW = false;        // true when every picker row is one flight
let POINT_REVIEW = false;       // true when every picker row is a complete point
let REVIEW_ACTIVE = false;
let REVIEW_PAUSED = false;
let embedFrame = null;          // last frame requested by the compare parent
const REVIEW_KEY = 'portal3d.shot_quality_review_v4';
let reviewState = { records: {} };

function activatePointReview() {
  SHOT_REVIEW = false;
  POINT_REVIEW = true;
  REVIEW_ACTIVE = true;
  document.body.classList.add('review');
  document.getElementById('review-title').textContent = 'Blinded 3D full-point review';
  document.getElementById('review-help').textContent =
    'Loading the current point-review contract…';
  document.getElementById('picker-title').textContent = 'Full point';
}

// =====================================================================================
// scene bootstrap
// =====================================================================================
function initScene() {
  const canvas = document.getElementById('scene');
  const renderStatus = document.getElementById('render-status');
  canvas.dataset.renderState = 'initializing';
  canvas.addEventListener('webglcontextlost', (event) => {
    event.preventDefault();
    canvas.dataset.renderState = 'lost';
    renderStatus.textContent = '3D temporarily unavailable: graphics context lost. Video controls remain usable.';
    renderStatus.hidden = false;
  });
  canvas.addEventListener('webglcontextrestored', () => {
    canvas.dataset.renderState = 'restoring';
    renderStatus.textContent = 'Restoring 3D…';
  });
  renderer = new THREE.WebGLRenderer({ canvas, antialias: true });
  renderer.setPixelRatio(Math.min(window.devicePixelRatio, 2));
  scene = new THREE.Scene();
  scene.background = new THREE.Color(COL.surface);
  scene.fog = new THREE.Fog(COL.surface, 40, 90);

  camera = new THREE.PerspectiveCamera(50, 2, 0.1, 500);
  camera.position.copy(V(5.485, -12, 10));
  controls = new OrbitControls(camera, renderer.domElement);
  controls.enableDamping = true;
  controls.dampingFactor = 0.08;
  controls.target.copy(V(5.485, 11.885, 1.2));
  controls.maxPolarAngle = Math.PI * 0.52; // don't go under the court

  scene.add(new THREE.AmbientLight(0xffffff, 0.75));
  const key = new THREE.DirectionalLight(0xffffff, 0.9);
  key.position.set(6, 20, -8);
  scene.add(key);

  buildCourt();
  clock = new THREE.Clock();
  window.addEventListener('resize', onResize);
  onResize();
  animate();
}

function onResize() {
  const wrap = document.getElementById('stage');
  const w = wrap.clientWidth, h = wrap.clientHeight;
  renderer.setSize(w, h, false);
  camera.aspect = w / h;
  camera.updateProjectionMatrix();
}

// =====================================================================================
// static court + net
// =====================================================================================
function buildCourt() {
  const W = 10.97, L = 23.77, netY = L / 2, inset = 1.37, svc = 6.40;
  // playing surface + a margin apron
  const apron = new THREE.Mesh(
    new THREE.PlaneGeometry(W + 12, L + 14),
    new THREE.MeshStandardMaterial({ color: 0x141a17, roughness: 1 }));
  apron.rotation.x = -Math.PI / 2;
  apron.position.copy(V(W / 2, netY, -0.01));
  scene.add(apron);
  const surf = new THREE.Mesh(
    new THREE.PlaneGeometry(W, L),
    new THREE.MeshStandardMaterial({ color: COL.court, roughness: 1 }));
  surf.rotation.x = -Math.PI / 2;
  surf.position.copy(V(W / 2, netY, 0));
  scene.add(surf);

  const seg = [
    [[0, 0], [W, 0]], [[0, L], [W, L]], [[0, 0], [0, L]], [[W, 0], [W, L]],
    [[inset, 0], [inset, L]], [[W - inset, 0], [W - inset, L]],
    [[inset, netY - svc], [W - inset, netY - svc]], [[inset, netY + svc], [W - inset, netY + svc]],
    [[W / 2, netY - svc], [W / 2, netY + svc]],
    [[W / 2, 0], [W / 2, 0.4]], [[W / 2, L], [W / 2, L - 0.4]],
  ];
  const pts = [];
  for (const [[x0, y0], [x1, y1]] of seg) { pts.push(V(x0, y0, 0.01), V(x1, y1, 0.01)); }
  const lines = new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(pts),
    new THREE.LineBasicMaterial({ color: COL.line }));
  scene.add(lines);

  buildNet(W, netY, inset);
}

function buildNet(W, netY, singlesInset) {
  // Regulation singles posts sit 0.914 m outside the singles sidelines.
  const postH = 1.07, centerH = 0.914;
  const geometry = Core.singlesNetGeometry(W, singlesInset);
  const { xMin, xMax } = geometry;
  netHeightFn = Core.makeSinglesNetHeightFn(W, singlesInset, centerH, postH);
  const xs = [xMin, (xMin + xMax) / 2, xMax];
  const tape = [];
  for (const x of xs) {
    const h = netHeightFn(x);
    tape.push(V(x, netY, h));
  }
  // top tape
  const topPts = [];
  for (let i = 0; i < tape.length - 1; i++) { topPts.push(tape[i], tape[i + 1]); }
  scene.add(new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(topPts),
    new THREE.LineBasicMaterial({ color: COL.net })));
  // net mesh (faint grid) between tape and ground
  const grid = [];
  for (let i = 0; i <= 24; i++) {
    const x = xMin + geometry.width * i / 24;
    const h = netHeightFn(x);
    grid.push(V(x, netY, 0), V(x, netY, h));
  }
  for (let j = 1; j <= 3; j++) {
    const frac = j / 4;
    const row = [];
    for (let i = 0; i <= 24; i++) {
      const x = xMin + geometry.width * i / 24;
      const h = netHeightFn(x);
      row.push(V(x, netY, h * frac));
    }
    for (let i = 0; i < row.length - 1; i++) grid.push(row[i], row[i + 1]);
  }
  scene.add(new THREE.LineSegments(
    new THREE.BufferGeometry().setFromPoints(grid),
    new THREE.LineBasicMaterial({ color: COL.net, transparent: true, opacity: 0.28 })));
  // posts
  for (const x of [xMin, xMax]) {
    const post = new THREE.Mesh(
      new THREE.CylinderGeometry(0.05, 0.05, postH, 8),
      new THREE.MeshStandardMaterial({ color: 0xdedede }));
    post.position.copy(V(x, netY, postH / 2));
    scene.add(post);
  }
  // center strap label anchor handled by overlays
}

// =====================================================================================
// per-point dynamic content
// =====================================================================================
function clearDynamic() {
  if (dynamicGroup) { scene.remove(dynamicGroup); disposeGroup(dynamicGroup); }
  dynamicGroup = new THREE.Group();
  scene.add(dynamicGroup);
  trailTubes = [];
  trailMarks = [];
  clearLabels();
}

// Register a 3D marker (its materials) as a trail element keyed to event time `t`, so trail
// mode can fade it in/out via material.opacity without touching geometry.
function regMark(t, materials) {
  const mats = materials.filter(Boolean).map((m) => ({
    mat: m, baseOpacity: m.opacity != null ? m.opacity : 1, baseTransparent: m.transparent === true,
  }));
  if (mats.length) trailMarks.push({ t, mats });
}
function disposeGroup(g) {
  g.traverse((o) => {
    if (o.geometry) o.geometry.dispose();
    if (o.material) (Array.isArray(o.material) ? o.material : [o.material]).forEach((m) => m.dispose());
  });
}

function metricValues() {
  const f = doc.frames;
  if (metric === 'height') return f.z;
  if (metric === 'spin') return f.spin_mag;
  return f.speed;
}

function buildTrajectory() {
  const f = doc.frames;
  const vals = metricValues();
  const range = Core.metricRange(vals);
  const runs = Core.trajectoryRuns(doc);
  // Rejected points remain inspectable, but their tubes are translucent and the diagnostic
  // banner is unmissable. Dashed centrelines and missing-span bridges remain opt-in clutter.
  // A point that carries a per-flight verdict is judged flight by flight, so its
  // accepted flights draw as ordinary accepted flights even when the whole point
  // is not complete.  Without that verdict the old whole-point read still holds.
  const rejected = doc.per_flight
    ? doc.per_flight.accepted_flight_count === 0
    : (doc.quality && doc.quality.accepted === false);
  // A point may be partially valid: every flight carries its own verdict. An
  // accepted flight draws as the normal metric tube; a rejected one draws as a
  // translucent dashed red segment with a GAP label, so the hole the owner is
  // being asked to judge is visible rather than implied.
  const runRejected = (i) => Array.isArray(f.acceptance) && f.acceptance[i] === 'rejected';
  const drawTubes = true;
  // one tube per contiguous run, per-vertex coloured by the chosen metric
  if (drawTubes) for (const [a, b] of runs) {
    if (b - a < 1) { continue; }
    const path = [];
    const idxs = [];
    for (let i = a; i <= b; i++) {
      if (f.x[i] === null || f.z[i] === null) continue;
      path.push(V(f.x[i], f.y[i], f.z[i]));
      idxs.push(i);
    }
    if (path.length < 2) continue;
    const flightContext = doc.review_context?.kind === 'local_s6_competitive'
      && Core.reviewPhase(doc, f.t[idxs[0]]) !== 'scored';
    const flightRejected = !flightContext && runRejected(idxs[0]);
    const curve = new THREE.CatmullRomCurve3(path);
    // TubeGeometry samples curves by arc length (`getPointAt`), while the trail reveal is
    // indexed by native video time. Override the arc-length mapping so both the moving ball
    // and tube rings use the same frame-uniform parameterization.
    curve.getPointAt = (u, target) => curve.getPoint(u, target);
    curve.getTangentAt = (u, target) => curve.getTangent(u, target);
    const tubular = Math.max(8, path.length * 3);
    const geo = new THREE.TubeGeometry(curve, tubular, 0.045, 8, false);
    const pos = geo.attributes.position;
    // RGBA per-vertex colour: alpha is the trail channel (1 in the static full view; the trail
    // window rewrites it per frame). vertTimes carries each vertex's clip time for that mapping.
    const colors = new Float32Array(pos.count * 4);
    const vertTimes = new Float32Array(pos.count);
    const ringVerts = 9; // radialSegments+1
    for (let vi = 0; vi < pos.count; vi++) {
      const ring = Math.floor(vi / ringVerts);
      const along = ring / tubular;             // 0..1 along the run
      const srcI = idxs[Math.min(idxs.length - 1, Math.round(along * (idxs.length - 1)))];
      let val = vals[srcI];
      let t = val === null ? 0 : (val - range.lo) / (range.hi - range.lo);
      const c = flightContext ? { r: 0.55, g: 0.55, b: 0.55 } : Core.ramp(t);
      const lowConf = f.method[srcI] === 'low_conf';
      const k = (lowConf || flightRejected) ? 0.45 : 1.0; // fade low-confidence or rejected flight
      colors[vi * 4] = c.r * k + (1 - k) * 0.35;
      colors[vi * 4 + 1] = c.g * k + (1 - k) * 0.35;
      colors[vi * 4 + 2] = c.b * k + (1 - k) * 0.35;
      colors[vi * 4 + 3] = 1;
      vertTimes[vi] = (f.t[srcI] != null) ? f.t[srcI] : 0;
    }
    const colorAttr = new THREE.BufferAttribute(colors, 4);
    geo.setAttribute('color', colorAttr);
    // A rejected fit draws as a translucent tube (never a confident solid). In follow mode the
    // per-vertex reveal alpha multiplies this, so keep the base high enough that the CURRENT shot
    // of a diagnostic point stays followable (~0.55 * 0.75 bright window) while ghosts vanish.
    const dim = rejected || flightRejected || flightContext;
    const mesh = new THREE.Mesh(geo, new THREE.MeshBasicMaterial({
      vertexColors: true, transparent: dim,
      opacity: flightContext ? 0.35 : (flightRejected ? 0.3 : (rejected ? 0.55 : 1)), depthWrite: !dim }));
    dynamicGroup.add(mesh);
    trailTubes.push({ mesh, colorAttr, vertTimes, rejected: dim });
    if (flightRejected) {
      const dgeo = new THREE.BufferGeometry().setFromPoints(path);
      const dln = new THREE.Line(dgeo, new THREE.LineDashedMaterial({
        color: 0xff4d4d, dashSize: 0.3, gapSize: 0.22, transparent: true, opacity: 0.95 }));
      dln.computeLineDistances();
      dynamicGroup.add(dln);
      const mid = path[Math.floor(path.length / 2)];
      const flight = (doc.per_flight && doc.per_flight.flights || [])
        .find((row) => `flight_${String(row.flight_index).padStart(2, '0')}` === f.segment[idxs[0]]);
      addLabel(
        V(mid.x, mid.y, mid.z + 0.3),
        flight ? `GAP ${flight.role}: ${flight.failures.join(', ')}` : 'GAP: flight not accepted',
        'lab-out',
        f.t[idxs[0]],
      );
    }
    // rejected fits also get a dashed centreline so the "not trusted" read is unmissable in 3D —
    // but it is low-trust clutter, so it only draws when "suppressed" geometry is enabled.
    if (rejected && !flightContext && showSuppressed) {
      const dgeo = new THREE.BufferGeometry().setFromPoints(path);
      const dln = new THREE.Line(dgeo, new THREE.LineDashedMaterial({
        color: 0xe0716a, dashSize: 0.28, gapSize: 0.2, transparent: true, opacity: 0.85 }));
      dln.computeLineDistances();
      dynamicGroup.add(dln);
    }
  }
  // honest missing spans: dashed grey bridge across each gap. Dashed = low-trust clutter, so it
  // is part of the "suppressed" set and hidden by default (off unless the viewer asks for it).
  if (overlays.missing && showSuppressed) {
    for (const g of Core.gapBridges(f.frame, f.segment)) {
      if (f.x[g.fromIdx] === null || f.x[g.toIdx] === null) continue;
      const geo = new THREE.BufferGeometry().setFromPoints([
        V(f.x[g.fromIdx], f.y[g.fromIdx], f.z[g.fromIdx]),
        V(f.x[g.toIdx], f.y[g.toIdx], f.z[g.toIdx]),
      ]);
      const mat = new THREE.LineDashedMaterial({ color: COL.gap, dashSize: 0.3, gapSize: 0.25, transparent: true, opacity: 0.7 });
      const ln = new THREE.Line(geo, mat); ln.computeLineDistances();
      dynamicGroup.add(ln);
      regMark(f.t[g.fromIdx], [mat]);
    }
  }
}

// Surviving legal-depth family members are not interchangeable with the selected member.
// Keep the selected path as the normal metric-coloured tube and draw every other member as
// a thin, dashed, labelled alternate.  The side panel repeats the depth labels even when the
// optional in-scene annotation chips are hidden.
function buildAlternates() {
  if (!overlays.family) return;
  const palette = [0x59c3ff, 0xff7ac2, 0xb9f55d, 0xff9f5d, 0xb5a7ff];
  for (const [alternateIndex, alternate] of (doc.alternates || []).entries()) {
    const f = alternate.frames || {};
    if (!Array.isArray(f.frame)) continue;
    for (const [a, b] of Core.splitRuns(f.frame, f.segment)) {
      const path = [];
      for (let i = a; i <= b; i++) {
        if (f.x[i] == null || f.y[i] == null || f.z[i] == null) continue;
        path.push(V(f.x[i], f.y[i], f.z[i]));
      }
      if (path.length < 2) continue;
      const material = new THREE.LineDashedMaterial({
        color: palette[alternateIndex % palette.length],
        dashSize: 0.23,
        gapSize: 0.15,
        transparent: true,
        opacity: 0.72,
      });
      const line = new THREE.Line(new THREE.BufferGeometry().setFromPoints(path), material);
      line.computeLineDistances();
      dynamicGroup.add(line);
    }
    const first = f.x.findIndex((value) => value != null);
    if (first >= 0) {
      addLabel(
        V(f.x[first], f.y[first], f.z[first] + 0.18),
        alternate.label || `serve depth ${alternate.serve_depth_m} m`,
        'lab-net',
        f.t[first],
      );
    }
  }
}

function discMark(cx, cy, r, color, opacity) {
  const m = new THREE.Mesh(
    new THREE.CircleGeometry(r, 24),
    new THREE.MeshBasicMaterial({ color, transparent: true, opacity: opacity ?? 0.85, side: THREE.DoubleSide }));
  m.rotation.x = -Math.PI / 2;
  m.position.copy(V(cx, cy, 0.02));
  return m;
}

function buildBounces() {
  for (const b of doc.bounces) {
    if (b.x === null) continue;
    const col = b.in_court ? COL.bounceIn : COL.bounceOut;
    const disc = discMark(b.x, b.y, 0.16, col, 0.9);
    dynamicGroup.add(disc);
    // ring
    const ring = new THREE.Mesh(
      new THREE.RingGeometry(0.18, 0.24, 24),
      new THREE.MeshBasicMaterial({ color: col, transparent: true, opacity: 0.7, side: THREE.DoubleSide }));
    ring.rotation.x = -Math.PI / 2;
    ring.position.copy(V(b.x, b.y, 0.02));
    dynamicGroup.add(ring);
    regMark(b.t, [disc.material, ring.material]);
    if (!b.in_court) {
      addLabel(V(b.x, b.y, 0.05), `OUT`, 'lab-out', b.t);
    }
  }
}

function buildContacts() {
  for (const c of doc.contacts) {
    if (c.x === null || c.z === null) continue;
    const cls = Core.contactClass(c);
    const color = cls === 'serve' ? COL.serve : (cls === 'dead' ? COL.dead : COL.rally);
    // vertical stick from ground to contact height (height validation)
    const stick = new THREE.Line(
      new THREE.BufferGeometry().setFromPoints([V(c.x, c.y, 0), V(c.x, c.y, c.z)]),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: cls === 'dead' ? 0.4 : 0.8 }));
    dynamicGroup.add(stick);
    // marker at contact
    const geo = cls === 'serve'
      ? new THREE.ConeGeometry(0.13, 0.26, 12)
      : new THREE.OctahedronGeometry(0.15);
    const mk = new THREE.Mesh(geo, new THREE.MeshStandardMaterial({
      color, emissive: color, emissiveIntensity: 0.35,
      transparent: cls === 'dead', opacity: cls === 'dead' ? 0.5 : 1,
    }));
    mk.position.copy(V(c.x, c.y, c.z));
    dynamicGroup.add(mk);
    const mats = [stick.material, mk.material];
    // spin arrow (physics-implied) from the outgoing arc
    if (overlays.spin && c.spin_out && c.spin_out.mag) {
      addSpinArrow(V(c.x, c.y, c.z), c.spin_out, mats);
    }
    regMark(c.t, mats);
    // height label
    let lab = `${c.z.toFixed(2)} m`;
    if (c.speed_out !== null) lab += ` · ${c.speed_out.toFixed(0)} m/s*`;
    addLabel(V(c.x, c.y, c.z + 0.25), lab, 'lab-contact ' + cls, c.t);
  }
}

function buildNetCollisions() {
  for (const collision of (doc.net_collisions || [])) {
    const geometry = new THREE.TorusGeometry(0.18, 0.045, 10, 24);
    const material = new THREE.MeshStandardMaterial({
      color: COL.netImpact,
      emissive: COL.netImpact,
      emissiveIntensity: 0.5,
    });
    const marker = new THREE.Mesh(geometry, material);
    marker.position.copy(V(collision.x, collision.y, collision.z));
    marker.rotation.x = Math.PI / 2;
    dynamicGroup.add(marker);
    regMark(collision.t, [material]);
    addLabel(
      V(collision.x, collision.y, collision.z + 0.22),
      'NET IMPACT',
      'lab-contact dead',
      collision.t,
    );
  }
}

function addSpinArrow(origin, spin, matSink) {
  const dir = new THREE.Vector3(spin.x || 0, spin.z || 0, spin.y || 0);
  if (dir.lengthSq() < 1e-9) return;
  dir.normalize();
  const arrow = new THREE.ArrowHelper(dir, origin, 0.9, COL.spin, 0.25, 0.15);
  dynamicGroup.add(arrow);
  // fold the arrow's line + cone materials into the caller's trail-mark material set
  if (matSink && arrow.line && arrow.cone) { matSink.push(arrow.line.material, arrow.cone.material); }
}

function buildReachBand() {
  if (!overlays.reach) return;
  // translucent slab across the court between 1.5 and 3.5 m — the human-reach band
  const W = 10.97, L = 23.77;
  for (const z of [1.5, 3.5]) {
    const plane = new THREE.Mesh(
      new THREE.PlaneGeometry(W, L),
      new THREE.MeshBasicMaterial({ color: COL.reach, transparent: true, opacity: 0.06, side: THREE.DoubleSide }));
    plane.rotation.x = -Math.PI / 2;
    plane.position.copy(V(W / 2, L / 2, z));
    dynamicGroup.add(plane);
    addLabel(V(0.2, L / 2, z), `${z.toFixed(1)} m`, 'lab-reach');
  }
}

function buildApexes() {
  if (!overlays.apex) return;
  for (const ap of Core.findApexes(doc.frames)) {
    const dot = new THREE.Mesh(
      new THREE.SphereGeometry(0.07, 10, 10),
      new THREE.MeshBasicMaterial({ color: COL.apex }));
    dot.position.copy(V(ap.x, ap.y, ap.z));
    dynamicGroup.add(dot);
    const at = (doc.frames.t[ap.index] != null) ? doc.frames.t[ap.index] : null;
    regMark(at, [dot.material]);
    addLabel(V(ap.x, ap.y, ap.z + 0.18), `apex ${ap.z.toFixed(2)} m`, 'lab-apex', at);
  }
}

function buildNetClearance() {
  netCrossGroup = new THREE.Group();
  dynamicGroup.add(netCrossGroup);
  if (!overlays.netclear) return;
  for (const cr of Core.netCrossings(doc.frames, doc.court.net_y, netHeightFn)) {
    const col = cr.aboveNet ? COL.netClear : COL.bounceOut;
    const dot = new THREE.Mesh(new THREE.SphereGeometry(0.08, 10, 10),
      new THREE.MeshBasicMaterial({ color: col }));
    dot.position.copy(V(cr.x, doc.court.net_y, cr.z));
    netCrossGroup.add(dot);
    regMark(cr.t, [dot.material]);
    addLabel(V(cr.x, doc.court.net_y, cr.z + 0.2),
      `net +${cr.clearance.toFixed(2)} m`, 'lab-net ' + (cr.aboveNet ? '' : 'below'), cr.t);
  }
}

function buildPlayers() {
  players = { near: null, far: null };
  for (const side of ['near', 'far']) {
    const color = side === 'near' ? COL.near : COL.far;
    const g = new THREE.Group();
    const body = new THREE.Mesh(
      new THREE.CylinderGeometry(0.22, 0.28, 1.7, 16),
      new THREE.MeshStandardMaterial({ color, transparent: true, opacity: 0.18 }));
    body.position.y = 0.85;
    g.add(body);
    const disc = discMark(0, 0, 0.32, color, 0.5);
    disc.position.set(0, 0.02, 0);
    disc.rotation.x = -Math.PI / 2;
    g.add(disc);
    const linePositions = new Float32Array(SKELETON_EDGES.length * 2 * 3);
    const skeletonLine = new THREE.LineSegments(
      new THREE.BufferGeometry().setAttribute(
        'position',
        new THREE.BufferAttribute(linePositions, 3),
      ),
      new THREE.LineBasicMaterial({ color, transparent: true, opacity: 0.95 }),
    );
    skeletonLine.geometry.setDrawRange(0, 0);
    g.add(skeletonLine);
    const pointPositions = new Float32Array(32 * 3);
    const skeletonPoints = new THREE.Points(
      new THREE.BufferGeometry().setAttribute(
        'position',
        new THREE.BufferAttribute(pointPositions, 3),
      ),
      new THREE.PointsMaterial({ color, size: 0.09, sizeAttenuation: true }),
    );
    skeletonPoints.geometry.setDrawRange(0, 0);
    g.add(skeletonPoints);
    const nm = (doc.players.names && doc.players.names[side]) || side;
    g.userData = { body, skeletonLine, skeletonPoints, displayName: nm };
    g.visible = false;
    dynamicGroup.add(g);
    players[side] = g;
    addPlayerLabel(side, nm);
  }
}

function updatePlayerSkeleton(group, sample) {
  const { body, skeletonLine, skeletonPoints } = group.userData;
  body.visible = !sample;
  if (!sample) {
    skeletonLine.geometry.setDrawRange(0, 0);
    skeletonPoints.geometry.setDrawRange(0, 0);
    return;
  }
  // All three components are metric root-relative court coordinates. In particular joint[1]
  // is camera-ray depth, so an arm can reach toward/away from the net instead of remaining a
  // vertical billboard. The final racket edge is an ordinary 3D grip-to-face segment.
  const jointVector = (joint) => Core.courtVectorToWorld(joint[0], joint[1], joint[2]);
  const lineArray = skeletonLine.geometry.attributes.position.array;
  let lineVertex = 0;
  for (const [startName, endName] of SKELETON_EDGES) {
    const start = sample.joints[startName];
    const end = sample.joints[endName];
    if (!start || !end) continue;
    for (const value of [...jointVector(start), ...jointVector(end)]) {
      lineArray[lineVertex++] = value;
    }
  }
  skeletonLine.geometry.setDrawRange(0, lineVertex / 3);
  skeletonLine.geometry.attributes.position.needsUpdate = true;
  const pointArray = skeletonPoints.geometry.attributes.position.array;
  let pointVertex = 0;
  for (const joint of Object.values(sample.joints)) {
    for (const value of jointVector(joint)) pointArray[pointVertex++] = value;
  }
  skeletonPoints.geometry.setDrawRange(0, pointVertex / 3);
  skeletonPoints.geometry.attributes.position.needsUpdate = true;
}

// moving ball + spin
function buildBall() {
  ball = new THREE.Mesh(
    new THREE.SphereGeometry(Core.FLIGHT_BALL_RADIUS_M, 16, 16),
    new THREE.MeshStandardMaterial({ color: COL.ball, emissive: COL.ball, emissiveIntensity: 0.3 }));
  dynamicGroup.add(ball);
  spinArrow = null;
}

// =====================================================================================
// HTML overlay labels (billboarded via CSS, projected each frame)
// =====================================================================================
let labels = [];
function clearLabels() {
  const layer = document.getElementById('labels');
  layer.innerHTML = '';
  labels = [];
  playerLabels = {};
}
function addLabel(worldPos, text, cls, t) {
  const el = document.createElement('div');
  el.className = 'label ' + (cls || '');
  el.textContent = text;
  document.getElementById('labels').appendChild(el);
  // t (optional) ties this label to an event time so trail mode reveals it as the ball passes
  // and dims it after; labels without a t (reach band, reference chips) stay full opacity.
  labels.push({ el, pos: worldPos.clone(), t: (t != null ? t : null) });
}
let playerLabels = {};
function addPlayerLabel(side, text) {
  const el = document.createElement('div');
  el.className = 'label lab-player ' + side;
  el.textContent = text;
  document.getElementById('labels').appendChild(el);
  playerLabels[side] = el;
}
function projectLabels() {
  const stage = document.getElementById('stage');
  const w = stage.clientWidth, h = stage.clientHeight;
  const tmp = new THREE.Vector3();
  for (const l of labels) {
    // Annotation chips are OFF by default (owner: "the text labels add no value; remove them").
    if (!showLabels) { l.el.style.display = 'none'; continue; }
    tmp.copy(l.pos).project(camera);
    let vis = tmp.z < 1;
    if (vis && trail.active && l.t != null) {
      // Only the current reveal window carries a label; ghosted/older shots show none, so the
      // labels that DO appear all belong to the shot the ball is on right now.
      const a = Core.trailAlpha(l.t, playhead.t, trail);
      if (a >= trail.full - 1e-3) l.el.style.opacity = '1'; // inside the bright recent window
      else vis = false;                                     // future OR ghosted: hide entirely
    } else {
      l.el.style.opacity = '1';
    }
    l.el.style.display = vis ? 'block' : 'none';
    l.el.style.left = ((tmp.x * 0.5 + 0.5) * w) + 'px';
    l.el.style.top = ((-tmp.y * 0.5 + 0.5) * h) + 'px';
  }
  for (const side of ['near', 'far']) {
    const g = players[side], el = playerLabels[side];
    if (!g || !el) continue;
    if (!g.visible) { el.style.display = 'none'; continue; }
    tmp.setFromMatrixPosition(g.matrixWorld); tmp.y += 1.9;
    tmp.project(camera);
    const vis = tmp.z < 1;
    el.style.display = vis ? 'block' : 'none';
    el.style.left = ((tmp.x * 0.5 + 0.5) * w) + 'px';
    el.style.top = ((-tmp.y * 0.5 + 0.5) * h) + 'px';
  }
}

// =====================================================================================
// time / animation
// =====================================================================================
function setTime(t) {
  if (!doc) return;
  playhead.t = Core.clamp(t, timeBounds.lo, timeBounds.hi);
  // Local S6 context styling describes the native exposure shown in the monitor;
  // the fractional scrubber can otherwise put its badge on the other side of an ending.
  const phaseTime = doc.review_context?.kind === 'local_s6_competitive'
    ? nativeTimeForFrame(nativeFrameAtTime(playhead.t)) : playhead.t;
  const reviewPhase = Core.reviewPhase(doc, phaseTime);
  const context = reviewPhase !== 'scored';
  document.body.classList.toggle('review-context', context);
  const contextBadge = document.getElementById('shot-context');
  if (contextBadge) {
    contextBadge.textContent = doc.review_context?.kind === 'local_s6_competitive'
      ? Core.reviewCaption(doc, phaseTime) : reviewPhase === 'pre'
      ? 'PRE-SHOT CONTEXT'
      : reviewPhase === 'post' ? 'POST-SHOT CONTEXT' : 'SCORED SHOT';
  }
  // 3D ball + players — only when WebGL is up (video/scrubber below run regardless)
  if (renderer) {
    const bp = Core.reviewPhase(doc, playhead.t) === 'scored'
      ? Core.ballAtTime(doc.frames, playhead.t) : null;
    if (bp && bp.x !== null) {
      ball.visible = !bp.gap; // ball vanishes over a missing span (honest)
      const display = Core.ballPresentation(bp.z);
      ball.position.copy(V(bp.x, bp.y, display.centreHeightM));
      const scale = display.radiusM / Core.FLIGHT_BALL_RADIUS_M;
      ball.scale.setScalar(scale);
    } else { ball.visible = false; }
    const poseMissing = [];
    for (const side of ['near', 'far']) {
      if (!overlays.players) { players[side].visible = false; continue; }
      const p = Core.playerAtTime(doc.players[side], playhead.t);
      const playerLabel = playerLabels[side];
      if (p) {
        const sample = Core.skeletonAtTime(
          (doc.skeletons && doc.skeletons[side]) || [], playhead.t,
        );
        players[side].visible = true;
        players[side].position.copy(V(p.x, p.y, 0));
        updatePlayerSkeleton(players[side], sample);
        const boxOnly = p.pose_status === 'box';
        if (playerLabel) {
          playerLabel.textContent = sample
            ? players[side].userData.displayName
            : boxOnly
              ? `${players[side].userData.displayName} · box`
              : `${players[side].userData.displayName} · NO POSE`;
          playerLabel.classList.toggle('no-pose', !sample && !boxOnly);
        }
        // A flight-review box root is the player mark. It is not a missing skeleton.
        if (!sample && !boxOnly) poseMissing.push({ side, reason: p.pose_status || 'no_pose_at_frame' });
      } else {
        players[side].visible = false;
        if (playerLabel) playerLabel.classList.remove('no-pose');
        poseMissing.push({ side, reason: 'player_track_missing' });
      }
    }
    const poseNote = document.getElementById('pose-note');
    if (poseNote) {
      const frame = nativeFrameAtTime(playhead.t);
      const reviewNote = doc && doc.flight_overlay && doc.flight_overlay.pose_note;
      poseNote.textContent = poseMissing.length
        ? `NO POSE at native frame ${frame}: ${poseMissing.map((row) => `${row.side} (${row.reason.replaceAll('_', ' ')})`).join(' + ')}. No pose or stale player position is invented.`
        : (reviewNote || '');
      poseNote.classList.toggle('show', poseMissing.length > 0 || !!reviewNote);
    }
  }
  const sc = document.getElementById('scrub');
  sc.value = ((playhead.t - timeBounds.lo) / (timeBounds.hi - timeBounds.lo) * 1000) | 0;
  document.getElementById('clock').textContent =
    `t = ${playhead.t.toFixed(2)} s / ${timeBounds.hi.toFixed(2)} s`;
  updateTrailVisual(playhead.t);
  updateFlightReadout(nativeFrameAtTime(playhead.t));
  updateVideo(playhead.t);
}

// =====================================================================================
// playback trail — progressive reveal driven by Core.trailAlpha (pure window mapping).
// Per frame we ONLY rewrite alpha (tube vertex alpha channel + marker material.opacity);
// geometry is never rebuilt. Toggling the mode flips material.transparent once (a shader
// recompile), not every frame.
// =====================================================================================
// Flip the MATERIAL state of every current trail element for follow on/off. In follow mode a
// tube/marker must be transparent with depthWrite off so its per-vertex reveal alpha (tubes) /
// material.opacity (markers) actually blends; out of follow it returns to its base look (a
// rejected tube stays translucent, everything else opaque). Split out from setTrailMode so a
// REBUILD (metric / overlay / suppressed toggle) can re-arm the freshly built meshes — see
// rebuild(): without this the new confident tubes come back transparent:false and render every
// shot fully opaque while follow is on ("scrubbing shows all trajectories").
function setTrailMaterials(on) {
  for (const tt of trailTubes) {
    tt.mesh.material.transparent = on ? true : tt.rejected;
    tt.mesh.material.depthWrite = on ? false : !tt.rejected;
    tt.mesh.material.needsUpdate = true;
  }
  for (const mk of trailMarks) for (const e of mk.mats) {
    e.mat.transparent = on ? true : e.baseTransparent;
    e.mat.needsUpdate = true;
  }
}

function setTrailMode(on) {
  if (trail.active === on) { updateFollowButton(on); return; }
  trail.active = on;
  setTrailMaterials(on);
  updateFollowButton(on);
  if (on) updateTrailVisual(playhead.t); else restoreFull();
}

// The "Follow" toggle: reflects and controls whether follow (progressive-reveal) mode is on.
function updateFollowButton(on) {
  const btn = document.getElementById('follow');
  if (!btn) return;
  btn.classList.toggle('armed', on);
  const locked = !!(doc && doc.force_follow);
  btn.disabled = locked;
  btn.textContent = locked ? 'Follow: locked' : (on ? 'Follow: on' : 'Follow: off');
  if (locked) {
    btn.title = 'This reconstruction always hides future ball paths while scrubbing.';
    return;
  }
  btn.title = on
    ? 'Follow mode ON — the point reveals up to the scrubber, older shots ghost out, future shots stay hidden. Click for the whole point at once.'
    : 'Follow mode OFF — the whole point is shown at once. Click to follow the scrubber (reveal shots one at a time).';
}

// Turn follow mode on/off and (optionally) remember the choice for the rest of the session.
function applyFollowMode(on, persist) {
  if (doc && doc.force_follow) on = true;
  setTrailMode(on);
  updateFollowButton(on);
  if (persist) writeFollowPref(on ? 'on' : 'off');
}

// Full-opacity restore (leaving trail mode): reset every trail element to its base look. Called
// once on exit — the per-frame path (updateTrailVisual) does nothing while inactive.
function restoreFull() {
  for (const tt of trailTubes) {
    const arr = tt.colorAttr.array;
    for (let vi = 0; vi < tt.vertTimes.length; vi++) arr[vi * 4 + 3] = 1;
    tt.colorAttr.needsUpdate = true;
  }
  for (const mk of trailMarks) for (const e of mk.mats) e.mat.opacity = e.baseOpacity;
}

// Per-frame trail update: rewrite alpha for the current playhead. No-op when trail is off.
function updateTrailVisual(t) {
  if (!renderer || !trail.active) return;
  for (const tt of trailTubes) {
    const arr = tt.colorAttr.array, times = tt.vertTimes;
    for (let vi = 0; vi < times.length; vi++) {
      arr[vi * 4 + 3] = Core.trailAlpha(times[vi], t, trail);
    }
    tt.colorAttr.needsUpdate = true;
  }
  for (const mk of trailMarks) {
    const a = Core.trailAlpha(mk.t, t, trail);
    for (const e of mk.mats) e.mat.opacity = e.baseOpacity * a;
  }
}

// =====================================================================================
// broadcast reference monitor — the actual frame at the current playhead, synced to the
// 3D scrubber and play loop. Frame numbers == f_%04d.jpg file numbers (t == frame/fps).
// =====================================================================================
// Request the frame for playhead time t. We double-buffer: decode into an offscreen Image
// first, then swap the visible <img> to the (now-cached) URL on load — so the monitor never
// blanks mid-load. Under fast play we chase only the LATEST desired frame and drop the
// intermediates the network can't keep up with, which keeps the swap smooth instead of
// thrashing img.src every RAF.
function updateVideo(t) {
  if (!doc || !videoOn) return;
  const wrap = document.getElementById('video');
  if (!doc.frames_available) { wrap.classList.add('no-frames'); return; }
  const n = nativeFrameAtTime(t);
  if (n === null) return;
  desiredFrame = n;
  pumpVideo();
}

function pumpVideo() {
  if (videoLoading || desiredFrame === null || desiredFrame === lastFrameShown) return;
  const n = desiredFrame;
  const url = Core.frameUrl(doc, n);
  const wrap = document.getElementById('video');
  if (!url) { wrap.classList.add('no-frames'); return; }
  videoLoading = true;
  const generation = videoGeneration;
  const imgs = [document.getElementById('vframe'), document.getElementById('vframeB')];
  const back = imgs[1 - videoFront];   // load into the hidden buffer, reveal on load
  back.onload = () => {
    if (generation !== videoGeneration) return;
    const reveal = () => {
      if (generation !== videoGeneration) return;
      videoLoading = false;
      lastFrameShown = n;
      back.classList.add('show');
      imgs[videoFront].classList.remove('show');
      const ovIds = ['ovframe', 'ovframeB'];
      const ovShown = document.getElementById(ovIds[videoFront]);
      const ovNext = document.getElementById(ovIds[1 - videoFront]);
      videoFront = 1 - videoFront;
      document.getElementById('vcap').textContent =
        `${Core.reviewCaption(doc, nativeTimeForFrame(n))} · `
        + `${doc.point ?? doc.clip} · frame ${n} · native t=${nativeTimeForFrame(n).toFixed(3)}s`;
      const revealOverlay = () => {
        if (generation !== videoGeneration) return;
        if (ovNext) ovNext.classList.add('show');
        if (ovShown && ovShown !== ovNext) ovShown.classList.remove('show');
        const ovcap = document.getElementById('ovcap');
        if (ovcap) ovcap.textContent = document.getElementById('vcap').textContent;
      };
      if (ovNext && !ovNext.complete) ovNext.addEventListener('load', revealOverlay, { once: true });
      else revealOverlay();
      updateVideoOverlay(n);
      for (let k = 1; k <= 5; k++) { // warm a few frames ahead so play stays smooth
        const fn = n + k;
        if (fn > frameWarmCap) break;      // don't 404 past the last frame on disk
        const wu = Core.frameUrl(doc, fn);
        if (wu && !frameWarm.has(wu)) {
          frameWarm.add(wu);
          const im = new Image();
          im.onerror = () => { frameWarmCap = Math.min(frameWarmCap, fn - 1); };
          im.src = wu;
        }
      }
      pumpVideo(); // chase the latest desired frame
    };
    if (typeof back.decode === 'function') back.decode().then(reveal, reveal);
    else reveal();
  };
  back.onerror = () => {
    if (generation !== videoGeneration) return;
    videoLoading = false;
    wrap.classList.add('no-frames');
    const overlay = document.getElementById('overlay-video');
    if (overlay) overlay.classList.add('no-frames');
  };
  const ovIds = ['ovframe', 'ovframeB'];
  const ovBack = document.getElementById(ovIds[1 - videoFront]);
  if (ovBack) ovBack.src = url;
  back.src = url;
}

function nativeFrameAtTime(t) {
  return Core.nativeFrameAtTime(doc, t);
}

function nativeTimeForFrame(frame) {
  return Core.nativeTimeForFrame(doc, frame);
}

function updateVideoOverlay(frame) {
  const row = videoOverlayByFrame.get(frame) || {};
  document.querySelector('#video-ball-legend .front').textContent = Core.observedBallLegend(doc);
  const svg = document.getElementById('video-overlay');
  const size = (doc?.video_overlay && doc.video_overlay.native_size) || [1920, 1080];
  svg.setAttribute('viewBox', `0 0 ${size[0]} ${size[1]}`);
  const labeled = document.getElementById('vlabel');
  const fitted = document.getElementById('vfit');
  const cross = document.getElementById('vfit-cross');
  const front = row.labeled_front;
  const fit = row.fitted_projection;
  labeled.setAttribute('visibility', front ? 'visible' : 'hidden');
  if (front) {
    labeled.setAttribute('cx', front[0]);
    labeled.setAttribute('cy', front[1]);
    labeled.setAttribute('r', Math.max(8, Number(row.labeled_radius_px) || 8));
  }
  fitted.setAttribute('visibility', fit ? 'visible' : 'hidden');
  cross.setAttribute('visibility', fit ? 'visible' : 'hidden');
  if (fit) {
    fitted.setAttribute('cx', fit[0]);
    fitted.setAttribute('cy', fit[1]);
    cross.setAttribute('d', `M ${fit[0] - 12} ${fit[1]} H ${fit[0] + 12} `
      + `M ${fit[0]} ${fit[1] - 12} V ${fit[1] + 12}`);
  }
  const pose = servePoseByFrame.get(frame);
  const edges = (doc?.video_overlay && doc.video_overlay.serve_pose_edges) || SKELETON_EDGES;
  drawSvgPose(document.getElementById('vpose'), pose && pose.native_joints, edges, (point) => point);
  updateServeSideElevation(pose, edges);
  const flight = doc && doc.flight_overlay;
  document.getElementById('video-ball-legend').classList.toggle('show', !flight && !!(front || fit));
  drawFlightOverlay(frame);
}

const FLIGHT_SKELETON = [
  ['nose', 'left_shoulder'], ['nose', 'right_shoulder'],
  ['left_shoulder', 'right_shoulder'],
  ['left_shoulder', 'left_elbow'], ['left_elbow', 'left_wrist'],
  ['right_shoulder', 'right_elbow'], ['right_elbow', 'right_wrist'],
  ['left_shoulder', 'left_hip'], ['right_shoulder', 'right_hip'],
  ['left_hip', 'right_hip'],
  ['left_hip', 'left_knee'], ['left_knee', 'left_ankle'],
  ['right_hip', 'right_knee'], ['right_knee', 'right_ankle'],
];

function flightLayer(name) {
  const box = document.getElementById('fl-' + name);
  return !box || box.checked;
}

function overlayRecord(frame) {
  const rows = (doc && doc.flight_overlay && doc.flight_overlay.frames) || [];
  for (let i = rows.length - 1; i >= 0; i--) {
    if (rows[i].frame === frame) return rows[i];
  }
  return null;
}

function svgEl(name, attrs) {
  const node = document.createElementNS('http://www.w3.org/2000/svg', name);
  for (const [key, value] of Object.entries(attrs)) node.setAttribute(key, value);
  return node;
}

function drawFlightOverlay(frame) {
  const overlay = doc && doc.flight_overlay;
  const court = document.getElementById('ov-court');
  const fitLayer = document.getElementById('ov-fit');
  const players = document.getElementById('ov-players');
  const events = document.getElementById('ov-events');
  if (!court || !fitLayer || !players || !events) return;
  court.replaceChildren();
  fitLayer.replaceChildren();
  players.replaceChildren();
  events.replaceChildren();
  const legacy = !overlay;
  for (const id of ['vlabel', 'vfit', 'vfit-cross', 'vpose']) {
    const node = document.getElementById(id);
    if (node) node.style.display = legacy ? '' : 'none';
  }
  if (!overlay) return;
  const record = overlayRecord(frame) || {};
  const size = overlay.native_size || [1920, 1080];
  document.getElementById('video-overlay').setAttribute('viewBox', `0 0 ${size[0]} ${size[1]}`);
  const kind = record.kind || 'no camera';
  const cam = document.getElementById('ovcam');
  if (cam) cam.textContent = `${kind} · ${overlay.arm || ''}`;
  const svg = document.getElementById('video-overlay');
  const box = svg.viewBox.baseVal;
  const units = (px) => Core.overlayUserUnits(
    box.width || size[0] || 1920,
    box.height || size[1] || 1080,
    svg.clientWidth,
    svg.clientHeight,
    px,
  );
  const ballR = units(8);
  const fitR = units(7);
  const crossArm = units(9);
  const markSize = units(11);
  const fontSize = units(14);
  if (flightLayer('court')) {
    const cls = record.kind === 'held' ? 'court-held' : 'court-ok';
    for (const segment of record.court || []) {
      court.appendChild(svgEl('line', {
        class: cls, x1: segment[0], y1: segment[1], x2: segment[2], y2: segment[3],
      }));
    }
  }
  const addPath = (points, cls) => {
    if (!points || points.length < 2) return;
    fitLayer.appendChild(svgEl('polyline', {
      class: cls, fill: 'none', points: points.map((point) => point.join(',')).join(' '),
    }));
  };
  if (flightLayer('fit')) addPath(record.fit_path, 'fit-ok');
  if (flightLayer('rejected')) addPath(record.rejected_path, 'fit-reject');
  const mark = (point, cls, radius) => {
    if (!point) return;
    fitLayer.appendChild(svgEl('circle', { class: cls, cx: point[0], cy: point[1], r: radius }));
  };
  if (flightLayer('auto')) mark(record.auto, 'ball-auto', ballR);
  if (flightLayer('labelled') && record.labelled) {
    const [x, y] = record.labelled;
    fitLayer.appendChild(svgEl('path', {
      class: 'ball-label', fill: 'none',
      d: `M ${x - crossArm} ${y} H ${x + crossArm} M ${x} ${y - crossArm} V ${y + crossArm}`,
    }));
  }
  if (flightLayer('fit')) mark(record.fit, 'fit-ok', fitR);
  if (flightLayer('rejected')) mark(record.rejected, 'fit-reject', fitR);
  for (const player of record.players || []) {
    const cls = player.side === 'far' ? 'player-far' : 'player-near';
    if (flightLayer('boxes') && player.box) {
      const [x0, y0, x1, y1] = player.box;
      players.appendChild(svgEl('rect', { class: cls, x: x0, y: y0, width: x1 - x0, height: y1 - y0 }));
    }
    if (flightLayer('skeleton') && player.joints) {
      for (const [start, end] of FLIGHT_SKELETON) {
        if (!player.joints[start] || !player.joints[end]) continue;
        players.appendChild(svgEl('line', {
          class: 'skel',
          x1: player.joints[start][0], y1: player.joints[start][1],
          x2: player.joints[end][0], y2: player.joints[end][1],
        }));
      }
    }
  }
  if (flightLayer('events')) {
    for (const event of overlay.events || []) {
      if (Math.abs(Number(event.frame) - frame) > 6) continue;
      const at = event.xy || record.auto || record.labelled || record.fit || record.rejected;
      if (!at) continue;
      events.appendChild(svgEl('rect', {
        class: 'event-mark',
        x: at[0] - markSize / 2, y: at[1] - markSize / 2, width: markSize, height: markSize,
        transform: `rotate(45 ${at[0]} ${at[1]})`,
        stroke: event.status === 'Astra label' ? '#ffd228' : '#50dc50',
      }));
      if (Math.abs(Number(event.frame) - frame) <= 1.5) {
        const label = svgEl('text', {
          x: units(12), y: units(28) + events.childElementCount * fontSize, fill: '#f4f7fb', 'font-size': fontSize,
        });
        label.textContent = `${event.type} f${Math.round(event.frame)} ${Number(event.time_s).toFixed(2)}s ${event.source} ${event.status}`;
        events.appendChild(label);
      }
    }
  }
}

function updateFlightReadout(frame) {
  const panel = document.getElementById('flight-readout');
  if (!panel) return;
  const overlay = doc && doc.flight_overlay;
  panel.hidden = !overlay;
  if (!overlay) {
    readoutFrame = null;
    return;
  }
  if (frame === readoutFrame) return;
  readoutFrame = frame;
  const events = document.getElementById('flight-events');
  const checks = document.getElementById('flight-checks');
  events.replaceChildren();
  for (const event of overlay.events || []) {
    const button = document.createElement('button');
    button.type = 'button';
    const near = Math.abs(Number(event.frame) - frame) <= 1.5;
    button.classList.toggle('active', near);
    button.textContent = `${event.type} · f${Math.round(event.frame)} · ${Number(event.time_s).toFixed(2)}s · ${event.source} · ${event.status}`;
    button.addEventListener('click', () => setTime(nativeTimeForFrame(Number(event.frame))));
    events.appendChild(button);
  }
  if (!events.childElementCount) events.textContent = 'No events in this clip.';
  const lines = [];
  if (overlay.hold_reason) lines.push(`Hold: ${overlay.hold_reason}`);
  if (overlay.pose_note) lines.push(overlay.pose_note);
  for (const flight of overlay.flights || []) {
    const failed = (flight.failed_checks || []).join(', ');
    const bits = [
      flight.id,
      flight.outcome || (flight.accepted ? 'accepted' : 'no accept'),
      flight.cause || '',
      failed ? `failed ${failed}` : (flight.accepted ? 'accepted fit' : 'no retained fit'),
    ].filter(Boolean);
    lines.push(bits.join(' · '));
  }
  checks.replaceChildren();
  for (const line of lines) {
    const row = document.createElement('div');
    row.className = 'check';
    row.textContent = line;
    checks.appendChild(row);
  }
}

function drawSvgPose(group, joints, edges, transform) {
  group.replaceChildren();
  if (!joints) return;
  const ns = 'http://www.w3.org/2000/svg';
  for (const [first, second] of edges) {
    if (!joints[first] || !joints[second]) continue;
    const a = transform(joints[first]), b = transform(joints[second]);
    const line = document.createElementNS(ns, 'line');
    line.setAttribute('class', 'pose-limb');
    line.setAttribute('x1', a[0]); line.setAttribute('y1', a[1]);
    line.setAttribute('x2', b[0]); line.setAttribute('y2', b[1]);
    group.appendChild(line);
  }
  for (const point of Object.values(joints)) {
    const p = transform(point);
    const circle = document.createElementNS(ns, 'circle');
    circle.setAttribute('class', 'pose-joint');
    circle.setAttribute('cx', p[0]); circle.setAttribute('cy', p[1]); circle.setAttribute('r', 4);
    group.appendChild(circle);
  }
}

function updateServeSideElevation(pose, edges) {
  const panel = document.getElementById('serve-pose-audit');
  panel.classList.toggle('show', !!pose);
  const group = document.getElementById('serve-side-pose');
  if (!pose) { group.replaceChildren(); return; }
  const transform = (point) => [180 + Number(point[0]) * 95, 165 - Number(point[1]) * 52];
  drawSvgPose(group, pose.side_elevation_m, edges, transform);
  const step = Number.isFinite(pose.root_step_m) ? `${Number(pose.root_step_m).toFixed(3)} m` : 'n/a';
  document.getElementById('serve-pose-caption').textContent =
    `Serve side elevation · frame ${pose.frame} · ${pose.support} · root step ${step}`;
}

function applyVideoVisibility() {
  const missing = !!doc && !doc.frames_available;
  for (const id of ['video', 'overlay-video']) {
    const wrap = document.getElementById(id);
    if (!wrap) continue;
    wrap.classList.toggle('hidden', !videoOn);
    wrap.classList.toggle('no-frames', missing);
  }
}

function resetVideoPanel() {
  videoGeneration += 1;
  lastFrameShown = null;
  desiredFrame = null;
  videoLoading = false;
  frameWarm = new Set();
  frameWarmCap = Array.isArray(doc && doc.frame_range) && Number.isFinite(doc.frame_range[1])
    ? doc.frame_range[1] : Infinity;
  videoOverlayByFrame = new Map(
    ((doc && doc.video_overlay && doc.video_overlay.frames) || [])
      .map((row) => [row.frame, row]),
  );
  servePoseByFrame = new Map(
    ((doc && doc.video_overlay && doc.video_overlay.serve_pose_frames) || [])
      .map((row) => [row.frame, row]),
  );
  updateVideoOverlay(-1);
  const wrap = document.getElementById('video');
  wrap.classList.toggle('no-frames', !doc || !doc.frames_available);
  const overlayWrap = document.getElementById('overlay-video');
  if (overlayWrap) overlayWrap.classList.toggle('no-frames', !doc || !doc.frames_available);
  document.getElementById('vcam').textContent =
    doc?.extension_schema === 'native_source_context_point3d_v1' ? 'original native pictures · exact PTS'
      : doc?.native_timebase?.mapping === 'piecewise_linear_original_native_pts'
        ? `${Number(doc.fps.toFixed(3))} fps · exact source PTS`
        : doc && doc.fps ? `${doc.fps} fps · native cadence` : '';
  // Keep the current decoded image visible until the next point's first frame is ready.
  // The generation token prevents a stale in-flight request from winning the swap.
}

function animate() {
  requestAnimationFrame(animate);
  const dt = clock.getDelta();
  if (doc && playhead.playing) {
    let nt = playhead.t + dt * playhead.speed;
    if (nt >= timeBounds.hi) {
      nt = timeBounds.hi; playhead.playing = false; document.getElementById('play').textContent = '▶';
      // Do NOT flip follow mode off here — it is a persistent per-session choice, not a
      // play-only affordance. The scrubber can be dragged back to re-follow from any point.
    }
    setTime(nt);
  }
  controls.update();
  if (renderer.getContext().isContextLost()) return;
  renderer.render(scene, camera);
  renderer.domElement.dataset.renderState = 'ready';
  document.getElementById('render-status').hidden = true;
  if (doc) projectLabels();
}

// =====================================================================================
// loading + UI
// =====================================================================================
async function loadPoint(entryOrFile, requestedComponent = null) {
  const generation = ++pointGeneration;
  ++spanGeneration;
  document.getElementById('supported-spans').hidden = true;
  const entry = typeof entryOrFile === 'string'
    ? indexPoints.find((candidate) => pickerValue(candidate) === entryOrFile)
      || indexPoints.find((candidate) => candidate.file === entryOrFile)
    : entryOrFile;
  if (!entry) throw new Error('unknown reconstruction selection');
  if (entry.held && !entry.frames_available) { showHeldAttempt(entry); return; }
  document.getElementById('err').style.display = 'none';
  document.getElementById('native-context').hidden = false;
  const file = entry.file;
  const res = await fetch('data/' + file);
  if (!res.ok) throw new Error('failed to load ' + file + ' (' + res.status + ')');
  const source = await res.json();
  if (generation !== pointGeneration) return;
  activeEntry = entry;
  originalSource = source;
  selectedSpan = null;
  // Establish full native video before validating or fetching a child.
  showDocument(source, entry);
  const spans = Core.supportedSpans(source);
  const panel = document.getElementById('supported-spans');
  const picker = document.getElementById('span-picker');
  panel.hidden = spans.length === 0;
  picker.replaceChildren(new Option('Whole attempt · video and missing gaps', ''));
  for (const span of spans) {
    const option = new Option(`Frames ${span.startFrame}–${span.endFrame}${span.available ? '' : ' · no fitted output'}`, String(span.component_index));
    option.disabled = !span.available;
    picker.add(option);
  }
  const missing = source.original_source_plan?.original_slots
    ?.filter((slot) => slot.component_index == null).map((slot) => `F${slot.start_frame}`) || [];
  document.getElementById('span-scope').textContent =
    `Whole-attempt video remains available. Unmodeled origins: ${missing.join(', ') || 'none'}. Spans are independent; the original attempt is incomplete.`;
  picker.onchange = () => loadSupportedSpan(picker.value).catch(showError);
  if (spans.length) {
    const requested = requestedComponent;
    const initial = requested === 'overview' ? null
      : spans.find((span) => String(span.component_index) === requested && span.available)
        || spans.find((span) => span.available);
    await loadSupportedSpan(initial ? String(initial.component_index) : '');
    return;
  }
}

async function loadSupportedSpan(value) {
  const source = originalSource;
  const entry = activeEntry;
  const sourceGeneration = pointGeneration;
  const generation = ++spanGeneration;
  const current = () => sourceGeneration === pointGeneration && generation === spanGeneration;
  try {
    let view = source;
    let selected = null;
    if (value !== '') {
      const bound = Core.supportedSpans(source).find((span) => String(span.component_index) === value && span.available);
      if (!bound) throw new Error('Unknown span for this original attempt');
      const response = await fetch('data/' + bound.file);
      if (!response.ok) throw new Error('Failed to load the bound supported span');
      const child = await response.json();
      if (!current()) return;
      view = Core.componentView(source, bound, child);
      selected = bound.component_index;
    }
    if (!current()) return;
    selectedSpan = selected;
    document.getElementById('span-picker').value = selected === null ? '' : String(selected);
    showDocument(view, entry);
  } catch (error) {
    if (!current()) return;
    document.getElementById('span-picker').value = selectedSpan === null ? '' : String(selectedSpan);
    throw error;
  }
}

function showDocument(source, entry) {
  readoutFrame = null;
  if (source.contact_component_scope && !source.component_source) {
    throw new Error('A child scene requires its original source binding');
  }
  doc = SHOT_REVIEW
    ? Core.sliceDocumentToShot(source, entry)
    : source;
  trail = { ...DEFAULT_TRAIL, ...(doc.trail || {}), active: false };
  netHeightFn = Core.makeSinglesNetHeightFn(
    doc.court.width,
    doc.court.singles_inset,
    doc.court.net_center_h,
    doc.court.net_post_h,
  );
  timeBounds = Core.timeBounds(doc);
  rebuildTimelineTicks();
  resetVideoPanel();
  applyVideoVisibility();
  trail.active = false;   // reset; follow mode is decided below once the scene is rebuilt
  rebuild();
  fillSidePanel();
  updateReviewPanel();
  // Start the playhead where the ball is actually observed. The scrubber still spans the
  // whole point, but opening in a leading players-only span made PLAY look dead (the ball
  // was invisible for seconds). firstBallTime lands us on visible motion immediately.
  let start = selectedSpan !== null ? doc.review_context.score_start_t
    : doc.review_context ? timeBounds.lo : Core.firstBallTime(doc);
  const flightStart = flightSeekTime();
  if (flightStart != null) start = flightStart;
  playhead.t = (start != null) ? Core.clamp(start, timeBounds.lo, timeBounds.hi) : timeBounds.lo;
  playhead.playing = false;
  document.getElementById('play').textContent = '▶';
  // Decide follow mode BEFORE the first setTime so long points open already revealed only up
  // to the playhead (which sits on first ball motion): no future spaghetti, older shots ghosted.
  // The per-session toggle preference overrides the length heuristic. This applies IN THE EMBED
  // (compare) view too — the parent owns the shared clock, but the reveal itself is driven here
  // by the frames the parent posts, and we report our resolved default to the parent on ready so
  // its "Follow" toggle mirrors this pane. Without this the compare view never revealed at all.
  const wantFollow = Core.shouldFollow(doc, readFollowPref());
  applyFollowMode(wantFollow, false);
  setTime(playhead.t);
  frameCamera();
  const sel = document.getElementById('picker');
  const flightValue = selectedFlight && entry ? `${entry.file}|${selectedFlight}` : null;
  const selectedValue = flightValue && [...sel.options].some((opt) => opt.value === flightValue)
    ? flightValue : pickerValue(entry);
  if (sel.value !== selectedValue && [...sel.options].some((opt) => opt.value === selectedValue)) {
    sel.value = selectedValue;
  }
  syncUrl();
  embedReady();
  document.body.classList.remove('landing');
}

function flightSeekTime() {
  if (!selectedFlight || !doc || !doc.flight_overlay) return null;
  const flight = (doc.flight_overlay.flights || []).find((row) => row.id === selectedFlight);
  if (!flight || !Number.isFinite(Number(flight.origin_frame))) return null;
  return nativeTimeForFrame(Number(flight.origin_frame));
}

function splitSelection(value) {
  const text = String(value);
  const bar = text.indexOf('|');
  if (bar < 0) return { file: text, flight: null };
  return { file: text.slice(0, bar), flight: text.slice(bar + 1) || null };
}

function syncUrl() {
  if (!doc) return;
  try {
    const u = new URL(window.location.href);
    if (u.searchParams.get('point') !== String(doc.point)) u.searchParams.delete('frame');
    u.searchParams.set('point', doc.point);
    u.searchParams.set('match', doc.match);
    if (['local_s6_cold', 'automatic_shared_s6', 'opened50_partial_s6'].includes(activeEntry.source_group)) u.searchParams.delete('file');
    else u.searchParams.set('file', activeEntry.file);
    if (originalSource?.component_scenes?.length) u.searchParams.set('component', selectedSpan === null ? 'overview' : String(selectedSpan));
    else u.searchParams.delete('component');
    if (doc.review_shot) u.searchParams.set('shot', doc.review_shot.shot_index);
    else u.searchParams.delete('shot');
    if (activeEntry && activeEntry.arm) u.searchParams.set('arm', activeEntry.arm);
    else u.searchParams.delete('arm');
    if (selectedFlight) u.searchParams.set('flight', selectedFlight);
    else u.searchParams.delete('flight');
    const audit = activeEntry && activeEntry.source_group === 'flight_review'
      ? Core.entryAuditSet(activeEntry) : (navState.set === 'points' ? 'points' : '');
    if (audit) u.searchParams.set('set', audit);
    else u.searchParams.delete('set');
    const source = activeEntry && activeEntry.source_group === 'flight_review'
      ? Core.matchIdentity(activeEntry).source : '';
    if (source) u.searchParams.set('source', source);
    else u.searchParams.delete('source');
    window.history.replaceState(null, '', u);
  } catch (e) { /* deep-linking is best-effort */ }
}

function rebuild() {
  if (!renderer) return;  // WebGL init failed — data panels only
  clearDynamic();
  buildTrajectory();
  buildAlternates();
  buildBounces();
  buildContacts();
  buildNetCollisions();
  buildApexes();
  buildNetClearance();
  buildReachBand();
  buildPlayers();
  buildBall();
  // Follow mode persists across a rebuild: rebuildKeepTime (metric / overlay / suppressed
  // toggles) calls rebuild() WITHOUT resetting trail.active, and setTrailMode would early-return
  // because the flag is unchanged. The freshly built meshes default to their full-view material
  // state (confident tubes opaque, markers at base opacity), so without re-arming here they'd
  // ignore the reveal window and draw every shot at once. Re-arm now; the caller's setTime()
  // then paints the correct per-vertex alpha. (On LOAD trail.active is false here, so this is a
  // no-op and applyFollowMode does the arming once the scene is built.)
  if (trail.active) setTrailMaterials(true);
}

function frameCamera() {
  if (!renderer) return;  // WebGL init failed — data panels only
  // point camera at the court centre; keep a stable, informative default
  controls.target.copy(V(doc.court.width / 2, doc.court.net_y, 1.0));
  camera.position.copy(V(doc.court.width / 2, -12, 10));
  controls.update();
}

function fillSidePanel() {
  const fmt = (value, digits = 0, fallback = '—') =>
    Number.isFinite(value) ? value.toFixed(digits) : fallback;
  const q = doc.quality || {};
  const branches = doc.interpretation_branches || {};
  const topology = doc.event_topology_branches || {};
  const review = reviewState.records[reviewKey()] || {};
  const blinded = REVIEW_ACTIVE && !review.quality;
  const reconstruction = Core.reconstructionStatus(doc);
  // unmissable state for rejected reconstructions (the picker/default used to bury this)
  const banner = document.getElementById('diagbanner');
  banner.classList.toggle('show', !blinded && reconstruction.held);
  const bannerTitle = banner.querySelector('b');
  const bannerSub = banner.querySelector('.sub');
  if (bannerTitle && bannerSub) {
    bannerTitle.textContent = reconstruction.title;
    bannerSub.textContent = reconstruction.detail;
  }
  const contextPanel = document.getElementById('native-context');
  contextPanel.hidden = !doc.review_context;
  document.getElementById('native-scope').textContent = Core.nativeContextSummary(doc);
  document.getElementById('native-first').onclick = () => setTime(timeBounds.lo);
  document.getElementById('native-contact').onclick = () => setTime(doc.review_context.score_start_t);
  document.getElementById('native-contact').hidden = doc.review_context?.kind === 'automatic_observations_only';
  document.getElementById('native-last').onclick = () => setTime(timeBounds.hi);
  const meta = document.getElementById('meta');
  let accepted = 'n/a';
  if (blinded) {
    accepted = '<span class="muted">automatic gate hidden until verdict is saved</span>';
  } else if (doc.extension_schema === 'connected_point3d_v1') {
    accepted = '';
  } else if (q.accepted === true) {
    accepted = '<span class="ok">Passed the acceptance gate</span>';
  } else if (q.accepted === false) {
    accepted = '<span class="warn">Reconstruction held — see gates and review below</span>';
  }
  const shot = doc.review_shot;
  const family = doc.family || {};
  const familyMembers = [];
  if (Number.isFinite(family.selected_depth_y_m)) {
    familyMembers.push(`<span class="family-member selected ok">selected ${family.selected_depth_y_m.toFixed(1)} m</span>`);
  }
  for (const alternate of (doc.alternates || [])) {
    familyMembers.push(`<span class="family-member" style="color:#59c3ff">${alternate.label}</span>`);
  }
  const familyRange = Array.isArray(family.depth_y_range_m)
    ? `${family.depth_y_range_m[0].toFixed(1)}–${family.depth_y_range_m[1].toFixed(1)} m`
    : 'none';
  meta.innerHTML = `
    <div class="row"><b>${doc.match} · point ${doc.point ?? doc.clip}${shot ? ` · shot ${shot.shot_index + 1}` : ''}</b></div>
    ${shot ? `<div class="row">frames ${shot.start_frame}–${shot.end_frame} · ${shot.start_side} to ${shot.end_side}${shot.terminal ? ' · terminal' : ''}</div>` : ''}
    <div class="row muted">tag ${doc.tag}</div>
    ${accepted ? `<div class="row">${accepted}</div>` : ''}
    ${q.verdict ? `<div class="row ${q.accepted ? 'ok' : 'warn'}"><b>${q.verdict}</b></div>` : ''}
    ${Number.isFinite(q.diagnostic_depth_m) ? `<div class="row warn">best diagnostic serve depth ${q.diagnostic_depth_m.toFixed(1)} m</div>` : ''}
    ${q.blocker_text ? `<div class="row ${q.accepted ? 'muted' : 'warn'}">${q.blocker_text}</div>` : ''}
    ${doc.per_flight ? `<div class="row ${reconstruction.visualIncorrect ? 'warn' : 'muted'}">
      <b>${reconstruction.gateSummary}</b>
      · ${reconstruction.counts}
      · rung ${doc.per_flight.rung}${doc.per_flight.gaps.length ? ` · ${doc.per_flight.gaps.length} gap(s)` : ''}</div>
      ${doc.per_flight.flights.map((row) => `<div class="row ${row.accepted ? 'muted' : 'warn'}">flight ${row.flight_index + 1} (${row.role}) f${row.start_frame}–${row.end_frame}: ${row.accepted ? 'accepted' : row.failures.join(', ')}</div>`).join('')}` : ''}
    ${doc.topology && doc.topology.summary ? `<div class="row muted">${doc.topology.summary}</div>` : ''}
    <div class="row">${doc.counts.frames} ball frames · ${doc.counts.contacts} contacts
      (${doc.counts.contacts_fit} fit) · ${doc.counts.bounces} bounces
      (${doc.counts.bounces_in_court} in court) · ${doc.counts.net_collisions || 0} net impacts</div>
    <div class="row">spin estimate: ${doc.spin_present ? 'present' : 'none'}</div>
    ${doc.confidence_status ? `<div class="row muted">${doc.confidence_status}</div>` : ''}
    ${doc.extension_schema === 'connected_point3d_v1' ? `<div class="family-summary">
      <div class="row"><b>legal depth family</b> · ${family.count || 0} member(s) · range ${familyRange}
        · width ${fmt(family.width_m, 1)} m · midpoint ${fmt(family.midpoint_m, 2)} m</div>
      <div class="family-members">${familyMembers.join('')}</div>
    </div>` : ''}
    ${q.footnote ? `<div class="row caveat"><sup>*</sup> ${q.footnote}</div>` : ''}
    ${!blinded && branches.enabled ? `<div class="row ${branches.decisive ? 'ok' : 'warn'}">
      whole-point branches: ${branches.decisive ? 'decisive' : 'ambiguous'}
      ${Number.isFinite(branches.margin) ? `· margin ${branches.margin.toFixed(2)}` : ''}
      · ${branches.alternatives.length} refit
    </div>` : ''}
    ${!blinded && topology.enabled ? `<div class="row ${topology.decisive ? 'ok' : 'warn'}">
      event topology: ${topology.selected || 'baseline'}
      · ${topology.decisive ? 'adopted' : 'baseline / abstain'}
      ${Number.isFinite(topology.margin) ? `· margin ${topology.margin.toFixed(2)}` : ''}
      · ${topology.scoring_stage || 'initial fit'}
    </div>` : ''}
    <div class="row">RPM prior deviations: ${doc.counts.spin_deviations || 0}</div>
    <div class="row caveat">${doc.speed_caveat}</div>
    ${doc.skeleton_caveat ? `<div class="row caveat">${doc.skeleton_caveat}</div>` : ''}`;

  const rows = doc.contacts.map((c) => {
    const cls = Core.contactClass(c);
    const spin = c.spin_out && Number.isFinite(c.spin_out.rpm)
      ? `${fmt(c.spin_out.rpm)}±${fmt(c.spin_out.rpm_ci, 0, '?')}` : '—';
    const components = c.spin_components_rpm;
    const spinDetail = components
      ? `${c.spin_profile || 'untyped'} · T ${fmt(components.topspin)}
        / S ${fmt(components.sidespin)} / R ${fmt(components.rifle)}`
      : '—';
    const sin = fmt(c.speed_in);
    const sout = fmt(c.speed_out);
    const z = fmt(c.z, 2);
    const interval = (Number.isFinite(c.frame_lo) && Number.isFinite(c.frame_hi))
      ? `${fmt(c.frame_lo)}–${fmt(c.frame_hi)}` : fmt(c.frame);
    const fittedAndLabeled = Number.isFinite(c.labeled_frame)
      ? `${fmt(c.frame, 1)} / ${fmt(c.labeled_frame, 1)}` : fmt(c.frame, 1);
    return `<tr class="ct-${cls}" data-t="${c.t}">
      <td>${fittedAndLabeled}</td><td>${c.side}</td><td>${cls}</td>
      <td>${sin}/${sout}</td><td>${z}</td><td>${spin}</td><td>${spinDetail}</td>
      <td class="muted">${interval}</td></tr>`;
  }).join('');
  document.getElementById('contacts').innerHTML = `
    <table><thead><tr><th>fit / label frame</th><th>side</th><th>tier</th><th>v in/out*</th>
      <th>ht m</th><th>RPM±CI</th><th>profile · T/S/R RPM</th>
      <th>fr interval</th></tr></thead><tbody>${rows}</tbody></table>`;
  document.querySelectorAll('#contacts tr[data-t]').forEach((tr) => {
    tr.addEventListener('click', () => { setTime(parseFloat(tr.dataset.t)); });
  });
}

function showHeldAttempt(held) {
  ++pointGeneration;
  ++spanGeneration;
  selectedSpan = null;
  originalSource = null;
  document.getElementById('supported-spans').hidden = true;
  playhead.playing = false;
  document.getElementById('play').textContent = '▶';
  doc = null;
  activeEntry = held;
  if (renderer) clearDynamic();
  resetVideoPanel();
  document.getElementById('native-context').hidden = true;
  document.getElementById('pose-note').classList.remove('show');
  document.getElementById('clock').textContent = 'No fitted output';
  document.getElementById('timeline-ticks')?.replaceChildren();
  document.getElementById('err').style.display = 'none';
  document.getElementById('contacts').textContent = '';
  document.getElementById('diagbanner').classList.remove('show');
  const readout = document.getElementById('flight-readout');
  if (readout) readout.hidden = true;
  document.getElementById('shot-context').textContent = 'NO CURRENT 3D';
  document.getElementById('picker').value = `held:${held.key}`;
  const status = String(held.status || 'held').replaceAll('_', ' ');
  document.getElementById('meta').textContent = `${held.key} · ${COHORT_LABEL}: ${status}. No fitted trajectory was produced. This attempt remains in the declared source roster.`;
  document.getElementById('vcap').textContent = `NO CURRENT 3D · ${status}`;
  const note = document.getElementById('vnote');
  note.textContent = 'No fitted output for this attempt.';
  note.style.display = 'block';
  const u = new URL(window.location.href);
  u.search = new URLSearchParams({point: held.key}).toString();
  window.history.replaceState(null, '', u);
}

function populateFlightFilters(entries) {
  const panels = [...new Set(entries.map((entry) => entry.panel).filter(Boolean))].sort();
  const causes = [...new Set(entries.flatMap((entry) => entry.causes || entry.flights?.map((flight) => flight.cause) || []))].filter(Boolean).sort();
  const panel = document.getElementById('flt-panel');
  const cause = document.getElementById('flt-cause');
  if (!panel || !cause) return;
  const keep = (select, values) => {
    const current = select.value;
    select.replaceChildren(new Option('all', 'all'));
    for (const value of values) select.add(new Option(value, value));
    if ([...select.options].some((opt) => opt.value === current)) select.value = current;
  };
  keep(panel, panels);
  keep(cause, causes);
}

function readFlightFilters() {
  const value = (id) => document.getElementById(id)?.value || 'all';
  return {
    panel: value('flt-panel'),
    arm: value('flt-arm'),
    outcome: value('flt-outcome'),
    cause: value('flt-cause'),
  };
}

function fillFlightPicker() {
  const sel = document.getElementById('picker');
  sel.innerHTML = '';
  const choices = Core.flightReviewChoices(flightEntries, readFlightFilters());
  indexPoints = [];
  const seen = new Set();
  for (const choice of choices) {
    const flight = choice.flight;
    const value = flight && flight.id ? `${choice.entry.file}|${flight.id}` : choice.entry.file;
    if (seen.has(value)) continue;
    seen.add(value);
    const opt = document.createElement('option');
    opt.value = value;
    opt.textContent = flight
      ? `${choice.entry.panel} · ${choice.entry.arm} · ${flight.outcome || 'flight'} · ${flight.id}${flight.cause ? ' · ' + flight.cause : ''}`
      : (choice.entry.label || choice.entry.point);
    sel.appendChild(opt);
    indexPoints.push(choice.entry);
  }
  const title = document.getElementById('picker-title');
  if (title) title.textContent = `Flight reviews (${choices.length})`;
  const scope = document.getElementById('cohort-scope');
  if (scope) scope.textContent = 'Filter by panel, arm, matched / missed / extra, and cause. A row seeks to that flight.';
}

function refillFlightOrCurrent() {
  const library = document.getElementById('lib-select');
  if (library && library.value === 'flights') {
    fillFlightPicker();
    return;
  }
  fillCurrentPicker(lastIndex);
}

function buildPicker(index) {
  lastIndex = index;
  flightEntries = (index.points || []).filter((entry) => entry.source_group === 'flight_review' && !entry.parent_source_key);
  populateFlightFilters(flightEntries);
  const library = document.getElementById('lib-select');
  if (library && !library.dataset.bound) {
    library.dataset.bound = '1';
    for (const id of ['lib-select', 'flt-panel', 'flt-arm', 'flt-outcome', 'flt-cause']) {
      const control = document.getElementById(id);
      if (!control) continue;
      control.addEventListener('change', () => {
        if (id !== 'lib-select' && library.value !== 'flights') library.value = 'flights';
        refillFlightOrCurrent();
      });
    }
    const sel = document.getElementById('picker');
    sel.addEventListener('change', () => {
      const parsed = splitSelection(sel.value);
      if (parsed.flight) selectedFlight = parsed.flight;
      else if (!SHOT_REVIEW) selectedFlight = null;
      loadPoint(SHOT_REVIEW ? sel.value : parsed.file).catch(showError);
    });
  }
  refillFlightOrCurrent();
}

function fillCurrentPicker(index) {
  if (!index) return;
  index = { ...index, points: index.points.filter((entry) => !entry.parent_source_key) };
  const sel = document.getElementById('picker');
  sel.innerHTML = '';
  if (CURRENT_COHORT) {
    indexPoints = Core.currentAttempts(index);
    const labels = ['Reviewed useful · complete', 'Gate complete · quality pending / uncertain / issues',
      'Incomplete fitted attempts', 'No fitted output / not run'];
    labels.forEach((label, rank) => {
      const rows = indexPoints.filter((p) => Core.currentAttemptGroup(p) === rank);
      if (!rows.length) return;
      const group = document.createElement('optgroup');
      group.label = `${label} (${rows.length})`;
      rows.forEach((entry) => {
        const option = document.createElement('option');
        option.value = entry.file;
        option.textContent = ['opened50_partial_s6', 'opened50_current_s6'].includes(entry.source_group) && !entry.held ? entry.label
          : `${entry.source_group === 'automatic_shared_s6' ? 'Automatic · ' : entry.source_group === 'opened50_current_s6' ? 'Opened50 current · ' : ''}${entry.point} · ${entry.held ? (entry.status === 'not_run_cap' ? 'not run (cap)' : 'no output') : `${entry.accepted_flights}/${entry.flight_count} flights pass gates`}`;
        group.append(option);
      });
      sel.append(group);
    });
    document.getElementById('picker-title').textContent = `${COHORT_LABEL} · full attempts`;
    document.getElementById('cohort-scope').textContent =
      `${Core.currentSelectionLabel(index)} · ${indexPoints.filter((p) => !p.held).length} fitted · ${indexPoints.filter((p) => p.held && p.status !== 'not_run_cap').length} without fitted output · ${Core.currentHeldAttempts(index).filter((p) => p.status === 'not_run_cap').length} not run (cap). Reviewed useful first; numerical gates and visual quality are separate. Historical lift sets are archived outside this picker.`;
    document.title = `${COHORT_LABEL} · full tennis attempts`;
    document.querySelector('.cross-nav b').textContent = `${COHORT_LABEL} · 3D attempts`;
    document.querySelector('.cross-nav a').href = index.validation_panel ? 'sources.html' : '/portal/progress/review.html';
    document.querySelector('.cross-nav a').textContent = index.validation_panel ? '50 source cases' : 'Human review checklist';
    document.getElementById('review-link').hidden = true;
    return;
  }
  if (SHOT_REVIEW) {
    indexPoints = (index.shots || []).slice().sort((left, right) =>
      left.match.localeCompare(right.match)
      || Number(left.point) - Number(right.point)
      || left.shot_index - right.shot_index);
    for (const shot of indexPoints) {
      const opt = document.createElement('option');
      opt.value = pickerValue(shot);
      opt.textContent = `${shot.match} · pt${String(shot.point).padStart(4, '0')} · shot ${shot.shot_index + 1} · f${shot.start_frame}–${shot.end_frame}`;
      sel.appendChild(opt);
    }
    return;
  }
  if (POINT_REVIEW) {
    indexPoints = (index.points || []).slice();
    for (const point of indexPoints) {
      const opt = document.createElement('option');
      opt.value = pickerValue(point);
      opt.textContent = `${point.match} · pt${String(point.point).padStart(4, '0')} · full point`;
      sel.appendChild(opt);
    }
    return;
  }
  const ordered = Core.orderPoints(index.points);
  const sourceGroups = index.source_groups || [
    { id: 'curated_real_attempts', label: 'Curated real attempts' },
  ];
  const knownIds = new Set(sourceGroups.map((group) => group.id));
  const bySource = new Map(sourceGroups.map((group) => [group.id, []]));
  bySource.set('other', []);
  for (const point of ordered) {
    const source = knownIds.has(point.source_group) ? point.source_group : 'other';
    bySource.get(source).push(point);
  }
  indexPoints = sourceGroups.flatMap((group) => bySource.get(group.id) || [])
    .concat(bySource.get('other'));
  const mkOpt = (p) => {
    const opt = document.createElement('option');
    opt.value = p.file;
    opt.textContent = p.label || Core.pointLabel(p, null);
    return opt;
  };
  const addGroup = (label, arr) => {
    if (!arr.length) return;
    const g = document.createElement('optgroup');
    g.label = label;
    arr.forEach((p) => g.appendChild(mkOpt(p)));
    sel.appendChild(g);
  };
  for (const group of sourceGroups) {
    const points = bySource.get(group.id) || [];
    addGroup(`${group.label} (${points.length})`, points);
  }
  addGroup(`Other still-valid sets (${bySource.get('other').length})`, bySource.get('other'));
}

function pickerValue(entry) {
  return SHOT_REVIEW ? `${entry.file}::${entry.shot_index}` : entry.file;
}

function entryReviewKey(entry) {
  if (!entry) return '';
  return `${entry.match}__${entry.point ?? entry.clip}__${entry.tag || 'default'}__shot_${entry.shot_index}`;
}

function reviewKey() {
  if (!doc) return '';
  return SHOT_REVIEW ? entryReviewKey(activeEntry)
    : `${doc.match}__${doc.point ?? doc.clip}__${doc.tag || 'default'}`;
}

function readReviewState() {
  try { reviewState = JSON.parse(localStorage.getItem(REVIEW_KEY) || '{"records":{}}'); }
  catch (e) { reviewState = { records: {} }; }
  if (!reviewState.records) reviewState.records = {};
}

function persistReviewState() {
  localStorage.setItem(REVIEW_KEY, JSON.stringify(reviewState));
}

function updateReviewPanel() {
  const panel = document.getElementById('review-panel');
  if (!panel || !doc) return;
  const record = reviewState.records[reviewKey()] || {};
  panel.querySelectorAll('[data-quality]').forEach((button) => {
    button.classList.toggle('active', button.dataset.quality === record.quality);
  });
  panel.querySelectorAll('[data-reason]').forEach((button) => {
    button.classList.toggle('active', (record.failure_reasons || []).includes(button.dataset.reason));
  });
  document.getElementById('review-notes').value = record.notes || '';
  const reviewed = indexPoints.filter((entry) => !!reviewState.records[
    SHOT_REVIEW ? entryReviewKey(entry)
      : `${entry.match}__${entry.point ?? entry.clip}__${entry.tag || 'default'}`
  ]?.quality).length;
  if (POINT_REVIEW) {
    document.getElementById('review-progress').textContent =
      `${reviewed} / ${indexPoints.length} full points complete`;
  } else {
    const groupedPoints = new Map();
    for (const entry of indexPoints) {
      const key = `${entry.match}__${entry.point}`;
      const entries = groupedPoints.get(key) || [];
      entries.push(entry);
      groupedPoints.set(key, entries);
    }
    const reviewedPoints = [...groupedPoints.values()].filter((entries) => entries.every((entry) =>
      !!reviewState.records[entryReviewKey(entry)]?.quality)).length;
    document.getElementById('review-progress').textContent =
      `${reviewed} / ${indexPoints.length} shots · ${reviewedPoints} / ${groupedPoints.size} points complete`;
  }
  fillSidePanel();
}

function currentReviewRecord() {
  if (REVIEW_PAUSED) return null;
  const key = reviewKey();
  return reviewState.records[key] || (reviewState.records[key] = {
    match_id: doc.match,
    point: doc.point ?? doc.clip,
    tag: doc.tag || '',
    shot_index: doc.review_shot?.shot_index ?? '',
    start_frame: doc.review_shot?.start_frame ?? '',
    end_frame: doc.review_shot?.end_frame ?? '',
    flight_id: doc.review_shot?.shot_id ?? '',
    review_unit: POINT_REVIEW ? 'full_live_play_point' : 'contact_to_contact_or_terminal_shot',
    quality: '',
    failure_reasons: [],
    notes: '',
    review_frame: null,
  });
}

function wireReviewControls() {
  const panel = document.getElementById('review-panel');
  if (!panel) return;
  panel.querySelectorAll('[data-quality]').forEach((button) => {
    button.addEventListener('click', () => {
      const record = currentReviewRecord();
      if (!record) return;
      record.quality = button.dataset.quality;
      if (record.quality !== 'not_good_enough') record.failure_reasons = [];
      record.review_frame = doc?.fps ? Math.round(playhead.t * doc.fps) : null;
      persistReviewState();
      updateReviewPanel();
    });
  });
  panel.querySelectorAll('[data-reason]').forEach((button) => {
    button.addEventListener('click', () => {
      const record = currentReviewRecord();
      if (!record) return;
      record.quality = 'not_good_enough';
      const reasons = new Set(record.failure_reasons || []);
      if (reasons.has(button.dataset.reason)) reasons.delete(button.dataset.reason);
      else reasons.add(button.dataset.reason);
      record.failure_reasons = [...reasons].sort();
      persistReviewState();
      updateReviewPanel();
    });
  });
  document.getElementById('review-notes').addEventListener('input', (event) => {
    const record = currentReviewRecord();
    if (!record) return;
    record.notes = event.target.value;
    persistReviewState();
  });
  document.getElementById('review-clear').addEventListener('click', () => {
    if (REVIEW_PAUSED) return;
    delete reviewState.records[reviewKey()];
    persistReviewState();
    updateReviewPanel();
  });
  document.getElementById('review-save-next').addEventListener('click', () => {
    const record = currentReviewRecord();
    if (!record) return;
    if (!record.quality) { alert('Choose one verdict first.'); return; }
    persistReviewState();
    const currentIndex = indexPoints.findIndex(
      (entry) => pickerValue(entry) === document.getElementById('picker').value,
    );
    const next = indexPoints.slice(currentIndex + 1).concat(indexPoints.slice(0, currentIndex + 1))
      .find((entry) => !reviewState.records[
        SHOT_REVIEW ? entryReviewKey(entry)
          : `${entry.match}__${entry.point ?? entry.clip}__${entry.tag || 'default'}`
      ]?.quality);
    if (next) loadPoint(next).catch(showError);
  });
  document.getElementById('review-download').addEventListener('click', () => {
    const header = ['review_unit', 'flight_id', 'match_id', 'point', 'tag', 'shot_index', 'start_frame', 'end_frame', 'quality', 'failure_reasons', 'review_frame', 'notes'];
    const activeKeys = new Set(indexPoints.map((entry) => (
      SHOT_REVIEW ? entryReviewKey(entry)
        : `${entry.match}__${entry.point ?? entry.clip}__${entry.tag || 'default'}`
    )));
    const activeRecords = Object.entries(reviewState.records)
      .filter(([key]) => activeKeys.has(key))
      .map(([, record]) => record);
    const rows = [header, ...activeRecords.map((record) => [
      record.review_unit || 'contact_to_contact_or_terminal_shot', record.flight_id,
      record.match_id, record.point, record.tag, record.shot_index,
      record.start_frame, record.end_frame, record.quality,
      (record.failure_reasons || []).join('|'), record.review_frame ?? '', record.notes || '',
    ])];
    const escape = (value) => `"${String(value ?? '').replaceAll('"', '""')}"`;
    const blob = new Blob([rows.map((row) => row.map(escape).join(',')).join('\n') + '\n'],
      { type: 'text/csv' });
    const link = document.createElement('a');
    link.href = URL.createObjectURL(blob);
    link.download = POINT_REVIEW
      ? 'reconstruction_3d_point_labels.csv' : 'reconstruction_3d_shot_labels.csv';
    link.click(); URL.revokeObjectURL(link.href);
  });
}

function stepFrame(delta) {
  if (!doc || !doc.fps) return;
  playhead.playing = false;
  document.getElementById('play').textContent = '▶';
  const frame = nativeFrameAtTime(playhead.t);
  if (frame != null) setTime(nativeTimeForFrame(frame + delta));
}

function rebuildTimelineTicks() {
  const lane = document.getElementById('timeline-ticks');
  if (!lane) return;
  lane.replaceChildren();
  for (const tick of Core.timelineTicks(doc, timeBounds)) {
    const marker = document.createElement('i');
    marker.className = `timeline-tick ${tick.source} ${tick.kind}`;
    marker.style.left = `${tick.fraction * 100}%`;
    marker.title = `${tick.source} ${tick.kind} · frame ${tick.frame}`;
    lane.appendChild(marker);
  }
}

function wireControls() {
  document.getElementById('play').addEventListener('click', () => {
    if (!doc) return;
    if (playhead.t >= timeBounds.hi) playhead.t = timeBounds.lo;
    if (!playhead.playing) {
      // starting playback: if we're sitting in a leading span with no ball yet, jump to the
      // first ball observation so play always produces visible motion (the reported "does
      // nothing" root cause on points whose ball track starts seconds into the window).
      const fb = Core.firstBallTime(doc);
      if (fb != null && playhead.t < fb - 1e-6) setTime(fb);
      // Play respects the current follow toggle (no longer force-enables the reveal): follow
      // mode already IS the default for long points on load, and an explicit "off" must stay off.
    }
    playhead.playing = !playhead.playing;
    document.getElementById('play').textContent = playhead.playing ? '❚❚' : '▶';
  });
  document.getElementById('step-back').addEventListener('click', () => stepFrame(-1));
  document.getElementById('step-forward').addEventListener('click', () => stepFrame(1));
  const follow = document.getElementById('follow');
  if (follow) follow.addEventListener('click', () => {
    if (doc && !doc.force_follow) applyFollowMode(!trail.active, true);
  });
  document.getElementById('scrub').addEventListener('input', (e) => {
    playhead.playing = false; document.getElementById('play').textContent = '▶';
    setTime(timeBounds.lo + (e.target.value / 1000) * (timeBounds.hi - timeBounds.lo));
  });
  document.getElementById('speed').addEventListener('change', (e) => { playhead.speed = parseFloat(e.target.value); });
  document.getElementById('metric').addEventListener('change', (e) => { metric = e.target.value; rebuildKeepTime(); });
  for (const key of Object.keys(overlays)) {
    const el = document.getElementById('ov-' + key);
    if (el) el.addEventListener('change', () => { overlays[key] = el.checked; rebuildKeepTime(); });
  }
  // labels toggle: pure visibility (projectLabels reads showLabels every frame) — no rebuild.
  const labTog = document.getElementById('ov-labels');
  if (labTog) { labTog.checked = showLabels; labTog.addEventListener('change', () => { showLabels = labTog.checked; }); }
  // suppressed toggle: changes what geometry is built, so rebuild while holding the playhead.
  const supTog = document.getElementById('ov-suppressed');
  if (supTog) { supTog.checked = showSuppressed; supTog.addEventListener('change', () => { showSuppressed = supTog.checked; rebuildKeepTime(); }); }
  for (const name of ['court', 'auto', 'labelled', 'boxes', 'skeleton', 'fit', 'rejected', 'events']) {
    const layer = document.getElementById('fl-' + name);
    if (layer) layer.addEventListener('change', () => {
      if (doc) drawFlightOverlay(nativeFrameAtTime(playhead.t));
    });
  }
  const vtog = document.getElementById('ov-video');
  if (vtog) vtog.addEventListener('change', () => {
    videoOn = vtog.checked;
    applyVideoVisibility();
    if (videoOn) { lastFrameShown = null; updateVideo(playhead.t); }
  });
  document.getElementById('reset-cam').addEventListener('click', frameCamera);
  document.addEventListener('keydown', (event) => {
    if (EMBED || event.altKey || event.ctrlKey || event.metaKey) return;
    const tag = event.target && event.target.tagName;
    if ((tag === 'INPUT' && event.target.type !== 'range') || tag === 'TEXTAREA' || tag === 'SELECT') return;
    if (event.key === 'ArrowLeft' || event.key === 'ArrowRight') {
      event.preventDefault();
      stepFrame(event.key === 'ArrowLeft' ? -1 : 1);
    }
    if (event.key === '[' || event.key === ']') {
      event.preventDefault();
      stepReviewFlight(event.key === '[' ? -1 : 1);
    }
  });
}

function rebuildKeepTime() {
  if (!renderer) return;  // WebGL init failed — data panels only
  const t = playhead.t;
  rebuild();
  setTime(t);
}

function showError(err) {
  console.error(err);
  document.getElementById('err').textContent = String(err && err.message || err);
  document.getElementById('err').style.display = 'block';
}

// ------------------------------------------------------------------- embedded mode
// When hosted inside the compare (2D+3D) view, the parent owns the single shared scrubber
// and play clock: it posts {source:'compare', kind:'seek'|'play'|'pause'|'follow', frame|on}
// and this pane maps the frame through the native timebase and renders it. Progressive-reveal
// (follow) is its OWN persistent state driven by the parent's 'follow' toggle — NOT tied to
// play/pause (the old play->on / pause->off coupling meant scrubbing, which pauses, always
// killed the reveal, so the compare view never followed). 'seek' renders the requested frame;
// when follow is on, setTime's reveal window paints it progressively. The listener is only
// wired when ?embed=1 — inert on the standalone page.
function frameToTime(frame) {
  if (!doc || !doc.fps) return timeBounds.lo;
  return Core.clamp(nativeTimeForFrame(frame), timeBounds.lo, timeBounds.hi);
}
function setupEmbed() {
  document.body.classList.add('embed');
  window.addEventListener('message', (ev) => {
    const m = ev.data;
    if (!m || m.source !== 'compare') return;
    if (m.kind === 'seek') {
      embedFrame = m.frame;
      if (doc) {
        playhead.playing = false;
        const pb = document.getElementById('play'); if (pb) pb.textContent = '▶';
        setTime(frameToTime(m.frame));
      }
    } else if (m.kind === 'follow') {
      // parent's Follow toggle: set (and persist) the reveal state, then repaint the current frame
      if (doc) applyFollowMode(!!m.on, true);
    }
    // 'play' / 'pause' from the parent no longer touch the reveal — follow is its own state.
  });
}
function embedReady() {
  if (!EMBED || !doc) return;
  // report our resolved follow default so the parent's Follow toggle mirrors this pane on load
  try { parent.postMessage({ source: 'viewer', kind: 'ready', fps: doc.fps, follow: trail.active }, '*'); }
  catch (e) { /* not embedded / cross-origin parent — ignore */ }
  if (embedFrame != null) setTime(frameToTime(embedFrame));
}

const NAV_SET_LABEL = {
  fresh: 'Fresh panels 0 and A',
  development: 'Development',
  panel_c: 'Panel C',
  error_themes: 'Error-theme audits',
  all: 'All flight reviews',
  points: 'Existing points',
};

function scopedFlightEntries() {
  const set = !navState.set || navState.set === 'points' ? 'all' : navState.set;
  return Core.searchFlightEntries(Core.flightEntriesForSet(flightEntries, set), navState.query);
}

function showLanding() {
  document.body.classList.add('landing');
}

function fillArchivePicker() {
  if (!lastIndex) return;
  const sel = document.getElementById('picker');
  sel.innerHTML = '';
  const groups = lastIndex.source_groups || [];
  const labels = new Map(groups.map((group) => [group.id, group.label]));
  const points = (lastIndex.points || []).filter((entry) => entry.source_group !== 'flight_review' && !entry.parent_source_key);
  const by = new Map();
  for (const entry of points) {
    const key = entry.source_group || 'other';
    if (!by.has(key)) by.set(key, []);
    by.get(key).push(entry);
  }
  indexPoints = points;
  for (const [key, rows] of by) {
    const group = document.createElement('optgroup');
    group.label = `${labels.get(key) || key} (${rows.length})`;
    for (const entry of rows) {
      const option = document.createElement('option');
      option.value = entry.file;
      option.textContent = entry.label || entry.point;
      group.appendChild(option);
    }
    sel.appendChild(group);
  }
  const title = document.getElementById('picker-title');
  if (title) title.textContent = `Existing points (${points.length})`;
  const scope = document.getElementById('cohort-scope');
  if (scope) scope.textContent = 'Older reconstructions stay here. A flight review is chosen above.';
}

function selectReview(entry, flightId) {
  if (!entry) return;
  const library = document.getElementById('lib-select');
  if (entry.source_group === 'flight_review' && library) library.value = 'flights';
  navState.set = entry.source_group === 'flight_review' ? Core.entryAuditSet(entry) : 'points';
  navState.match = entry.source_group === 'flight_review' ? Core.matchIdentity(entry).source : '';
  navState.point = entry.point;
  const setSelect = document.getElementById('nav-set');
  if (setSelect) setSelect.value = navState.set;
  if (flightId) selectedFlight = flightId;
  else if (entry.source_group === 'flight_review') {
    const flights = Core.orderedFlights(entry);
    selectedFlight = flights.length ? flights[0].id : null;
  }
  refillFlightOrCurrent();
  loadPoint(entry).catch(showError);
  renderNavigator();
}

function stepReviewFlight(direction) {
  if (!activeEntry || activeEntry.source_group !== 'flight_review') return;
  const next = Core.stepReview(scopedFlightEntries(), activeEntry.file, selectedFlight, direction);
  if (!next || !next.entry) return;
  if (next.entry.file !== activeEntry.file) {
    selectReview(next.entry, next.flight ? next.flight.id : null);
    return;
  }
  selectedFlight = next.flight ? next.flight.id : null;
  if (next.flight && Number.isFinite(Number(next.flight.origin_frame))) {
    setTime(nativeTimeForFrame(Number(next.flight.origin_frame)));
  }
  renderNavigator();
  syncUrl();
}

function renderNavigator() {
  const set = navState.set;
  const flightMode = !!set && set !== 'points';
  for (const id of ['nav-match-label', 'nav-point-label', 'nav-arms', 'nav-flights', 'nav-move']) {
    const node = document.getElementById(id);
    if (node) node.hidden = !flightMode;
  }
  const crumb = document.getElementById('nav-crumb');
  if (!crumb) return;
  crumb.replaceChildren();
  const crumbButton = (label, run) => {
    const button = document.createElement('button');
    button.type = 'button';
    button.textContent = label;
    button.addEventListener('click', run);
    crumb.appendChild(button);
  };
  crumbButton('Reviews', () => {
    navState = { set: '', match: '', point: '', query: navState.query };
    const setSelect = document.getElementById('nav-set');
    if (setSelect) setSelect.value = '';
    renderNavigator();
    showLanding();
  });
  if (set) crumbButton(NAV_SET_LABEL[set] || set, () => {
    navState.match = '';
    navState.point = '';
    renderNavigator();
  });
  const entries = flightMode ? scopedFlightEntries() : [];
  const matches = Core.matchChoices(entries);
  if (navState.match && !matches.some((row) => row.source === navState.match)) navState.match = '';
  const match = matches.find((row) => row.source === navState.match);
  if (match) crumbButton(match.title, () => { navState.point = ''; renderNavigator(); });
  const attempts = navState.match ? Core.attemptChoices(entries, navState.match) : [];
  if (navState.point && !attempts.some((row) => row.point === navState.point)) navState.point = '';
  if (navState.point) crumbButton(navState.point, () => renderNavigator());
  const matchSel = document.getElementById('nav-match');
  const pointSel = document.getElementById('nav-point');
  if (matchSel) {
    const current = navState.match;
    matchSel.replaceChildren(new Option(matches.length ? 'Choose a match' : 'No matches', ''));
    for (const row of matches) {
      const option = new Option(`${row.title} (${row.attempts})`, row.source);
      matchSel.add(option);
    }
    matchSel.value = current;
  }
  if (pointSel) {
    const current = navState.point;
    pointSel.replaceChildren(new Option(attempts.length ? 'Choose a point' : 'No points', ''));
    for (const row of attempts) pointSel.add(new Option(row.point, row.point));
    pointSel.value = current;
  }
  const arms = document.getElementById('nav-arms');
  const flights = document.getElementById('nav-flights');
  if (arms) arms.replaceChildren();
  if (flights) flights.replaceChildren();
  const attempt = attempts.find((row) => row.point === navState.point);
  if (!attempt || !arms || !flights) return;
  for (const chip of Core.armChips(attempt.arms)) {
    const button = document.createElement('button');
    button.type = 'button';
    button.classList.toggle('active', activeEntry && activeEntry.file === chip.file);
    button.append(chip.arm);
    for (const outcome of ['matched', 'missed', 'extra']) {
      if (!chip.counts[outcome]) continue;
      const mark = document.createElement('span');
      mark.className = `chip ${outcome}`;
      mark.textContent = `${chip.counts[outcome]} ${outcome}`;
      button.append(mark);
    }
    if (chip.causes.length) {
      const cause = document.createElement('span');
      cause.className = 'hint';
      cause.textContent = ` ${chip.causes.slice(0, 2).join(', ')}`;
      button.append(cause);
    }
    button.addEventListener('click', () => {
      const entry = attempt.arms.find((row) => row.file === chip.file);
      selectReview(entry, null);
    });
    arms.appendChild(button);
  }
  const current = attempt.arms.find((row) => activeEntry && row.file === activeEntry.file) || attempt.arms[0];
  for (const flight of Core.orderedFlights(current)) {
    const button = document.createElement('button');
    button.type = 'button';
    button.classList.toggle('active', selectedFlight === flight.id);
    button.textContent = `${flight.outcome || 'flight'} · ${flight.id}${flight.cause ? ' · ' + flight.cause : ''}`;
    button.addEventListener('click', () => selectReview(current, flight.id));
    flights.appendChild(button);
  }
}

function wireNavigator() {
  if (wireNavigator.done) return;
  wireNavigator.done = true;
  document.getElementById('nav-set').addEventListener('change', (event) => {
    navState.set = event.target.value;
    navState.match = '';
    navState.point = '';
    if (navState.set === 'points') fillArchivePicker();
    else if (navState.set) {
      const library = document.getElementById('lib-select');
      if (library) library.value = 'flights';
      refillFlightOrCurrent();
    }
    renderNavigator();
    if (!activeEntry || (navState.set && navState.set !== 'points' && Core.entryAuditSet(activeEntry) !== navState.set)) {
      showLanding();
    }
  });
  document.getElementById('nav-search').addEventListener('input', (event) => {
    navState.query = event.target.value;
    renderNavigator();
  });
  document.getElementById('nav-match').addEventListener('change', (event) => {
    navState.match = event.target.value;
    navState.point = '';
    renderNavigator();
  });
  document.getElementById('nav-point').addEventListener('change', (event) => {
    navState.point = event.target.value;
    renderNavigator();
  });
  document.getElementById('flight-prev').addEventListener('click', () => stepReviewFlight(-1));
  document.getElementById('flight-next').addEventListener('click', () => stepReviewFlight(1));
}

async function boot() {
  const params = new URLSearchParams(window.location.search);
  const reviewLink = document.getElementById('review-link');
  if (reviewLink) reviewLink.href = `${window.location.protocol}//${window.location.hostname}:5031/`;
  EMBED = params.get('embed') === '1';
  const reviewMode = params.get('review');
  SHOT_REVIEW = params.has('review') && reviewMode !== 'points';
  POINT_REVIEW = reviewMode === 'points';
  REVIEW_ACTIVE = SHOT_REVIEW || POINT_REVIEW;
  if (REVIEW_ACTIVE) document.body.classList.add('review');
  if (POINT_REVIEW) activatePointReview();
  readReviewState();
  if (EMBED) setupEmbed();
  // Scene init is the fragile part (WebGL availability varies by browser/GPU settings).
  // Never let it take the picker and side panel down with it — and always say WHY.
  let sceneOk = true;
  try { initScene(); } catch (e) {
    sceneOk = false;
    showError('3D rendering failed in this browser (' + String(e && e.message || e)
      + '). The point list and contact tables still work; try enabling hardware '
      + 'acceleration / WebGL, or a different browser.');
  }
  const videoExpand = document.getElementById('video-expand');
  if (videoExpand) videoExpand.addEventListener('click', () => {
    const panel = document.getElementById('video');
    const expanded = panel.classList.toggle('expanded');
    videoExpand.textContent = expanded ? 'shrink' : 'expand';
  });
  try { wireControls(); } catch (e) { if (sceneOk) showError(e); }
  try { wireReviewControls(); } catch (e) { if (sceneOk) showError(e); }
  try {
    const res = await fetch('data/index.json?v=' + Date.now());
    if (!res.ok) throw new Error('index.json ' + res.status);
    const index = await res.json();
    CURRENT_COHORT = !!(index.local_s6 || index.automatic_s6);
    COHORT_LABEL = Core.currentCohortLabel(index);
    if (CURRENT_COHORT) {
      SHOT_REVIEW = false; POINT_REVIEW = false; REVIEW_ACTIVE = false;
      document.body.classList.remove('review');
    }
    if (!params.has('review') && [
      'owner_calibration_candidate',
      'diagnostic_known_defects',
    ].includes(index.point_selection_status)) {
      activatePointReview();
    }
    REVIEW_PAUSED = REVIEW_ACTIVE && String(index.point_selection_status || '').startsWith('paused_');
    if (POINT_REVIEW) {
      const reviewHelp = document.getElementById('review-help');
      if (index.point_selection_status === 'owner_calibration_candidate') {
        reviewHelp.textContent = 'Post-repair calibration pilot. Review the complete 3D path against the video; these points intentionally sit near current physics and spatial gate boundaries.';
      } else if (index.point_selection_status === 'diagnostic_known_defects') {
        reviewHelp.textContent = 'Known-defect diagnostic queue. Notes are useful for engineering, but these held points are not released 3D output and do not estimate quality or yield.';
      } else {
        reviewHelp.textContent = 'PAUSED after systematic player-pose, hard-bounce, missing-flight, and net-impact defects. Existing labels remain exportable; no new owner labeling is requested for this queue.';
      }
    }
    if (REVIEW_PAUSED) {
      document.querySelectorAll('#review-panel button:not(#review-download), #review-panel textarea')
        .forEach((control) => { control.disabled = true; });
    }
    if (!index.points.length && !Core.currentHeldAttempts(index).length) throw new Error('no points in index');
    buildPicker(index);
    wireNavigator();
    if (!indexPoints.length && !(index.points || []).some((entry) => entry.source_group === 'flight_review')) {
      throw new Error('no reviewable reconstructions in index');
    }
    // deep link: ?point=N (&match=...) — the inspector links here; else default to the first
    // ACCEPTED point (never a rejected fit), else the best-ranked diagnostic.
    const wantPoint = params.get('point'), wantMatch = params.get('match');
    const wantShot = params.get('shot');
    const wantArm = params.get('arm');
    const wantFlight = params.get('flight');
    const wantSet = params.get('set');
    if (wantSet) {
      navState.set = wantSet;
      const setSelect = document.getElementById('nav-set');
      if (setSelect) setSelect.value = wantSet;
    }
    if (params.get('source')) navState.match = params.get('source');
    if (wantFlight) selectedFlight = wantFlight;
    const deepLink = !!(wantPoint || params.get('file') || wantArm || wantFlight);
    if (!deepLink) {
      if (wantSet === 'points') fillArchivePicker();
      renderNavigator();
      showLanding();
      return;
    }
    let requested = Core.resolvePointRequest(indexPoints,
      Core.currentHeldAttempts(index), {
        point: wantPoint, match: wantMatch, shot: wantShot,
        shotReview: SHOT_REVIEW, file: params.get('file'), arm: wantArm,
      });
    if (!requested.entry && !requested.held && wantPoint) {
      const flightHit = Core.resolvePointRequest(
        (index.points || []).filter((entry) => entry.source_group === 'flight_review'),
        [],
        {
          point: wantPoint, match: null, shot: wantShot,
          file: params.get('file'), arm: wantArm,
        },
      );
      if (flightHit.entry) requested = flightHit;
    }
    if (requested.missingFile) throw new Error('This document is not in the current cohort. Choose a current full attempt below.');
    if (requested.held) { showHeldAttempt(requested.held); return; }
    if (CURRENT_COHORT && wantPoint && !requested.entry && !wantArm && !wantFlight) {
      throw new Error(`This point is not in the ${COHORT_LABEL.toLowerCase()} cohort. Historical lift sets are archived; choose a current full attempt below.`);
    }
    if ((wantArm || wantFlight) && wantPoint && !requested.entry) {
      throw new Error('This flight review is not in the portal index.');
    }
    let target = requested.entry;
    if (target && target.source_group === 'flight_review') {
      const library = document.getElementById('lib-select');
      if (library) library.value = 'flights';
      refillFlightOrCurrent();
    }
    if (!target) {
      target = SHOT_REVIEW ? indexPoints[0]
        : Core.defaultPoint(indexPoints);
    }
    await loadPoint(target, params.get('component'));
    // Audit links address original native pictures, including offset timebases.
    const requestedFrame = params.get('frame');
    const frame = requestedFrame === null ? NaN : Number(requestedFrame);
    if (!EMBED && Number.isSafeInteger(frame) && frame > 0) {
      const time = nativeTimeForFrame(frame);
      if (time >= timeBounds.lo && time <= timeBounds.hi) setTime(time);
    }
    if (target && target.source_group === 'flight_review') {
      navState.set = Core.entryAuditSet(target);
      navState.match = Core.matchIdentity(target).source;
      navState.point = target.point;
      const setSelect = document.getElementById('nav-set');
      if (setSelect) setSelect.value = navState.set;
    }
    renderNavigator();
  } catch (e) { showError(e); }
}

window.addEventListener('DOMContentLoaded', boot);
