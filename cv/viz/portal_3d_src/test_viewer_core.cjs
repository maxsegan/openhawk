// Node-VM test for viewer_core.js — validates the pure viewer logic (gap splitting, color
// ramp, apex finding, net crossings, scrubber lookups) without a browser or three.js.
// Run: node cv/viz/portal_3d_src/test_viewer_core.cjs
//
// The task calls for "testing the page's script logic in a Node VM"; we load the ES-module
// source, neutralise the `export` keywords, and evaluate it in a vm sandbox, then assert.

const fs = require('fs');
const path = require('path');
const vm = require('vm');
const assert = require('assert');

const srcPath = path.join(__dirname, 'viewer_core.js');
let src = fs.readFileSync(srcPath, 'utf8');
// strip ES-module syntax so it runs as a plain script in the VM sandbox
src = src
  .replace(/export default [\s\S]*$/m, '')
  .replace(/export function/g, 'function')
  .replace(/export const/g, 'const');

const sandbox = { globalThis: {}, Math, Number, Infinity, NaN, console };
sandbox.globalThis = sandbox;
vm.createContext(sandbox);
vm.runInContext(src, sandbox);
const C = sandbox.ViewerCore;
assert(C, 'ViewerCore api present');

let passed = 0;
function ok(name, fn) { fn(); passed++; console.log('  ok  ' + name); }
// cross-realm-safe structural equality (VM arrays have a different Array prototype)
function eq(a, b) { assert.strictEqual(JSON.stringify(a), JSON.stringify(b)); }

ok('mixed-input legends follow the marker contract and ball source independently of events', () => {
  const mixed = { provenance: { observation_origin: 'labeled' }, input_swap_scope: { ball: 'automatic' } };
  assert.strictEqual(C.observedBallLegend(mixed), '● automatic observed center');
  mixed.video_overlay = { labeled_marker: 'blue dot: automatic observed native center' };
  assert.strictEqual(C.observedBallLegend(mixed), '● automatic observed native center');
  assert.strictEqual(C.observedBallLegend({ video_overlay: { labeled_marker: 'blue dot: frozen human leading visible front' } }), '● frozen human leading visible front');
  assert.strictEqual(C.observedBallLegend({ provenance: { observation_origin: 'automatic' } }), '● automatic center');
  assert.strictEqual(C.observedBallLegend({}), '● labeled click');
});

ok('additional current cohorts preserve existing ordering and keep held slots separate', () => {
  const index = { automatic_s6: {}, source_groups: [
    { id: 'fresh_swaps', current: true, held_before_fitting: [{ key: 'fresh_held', status: 'preparation_held' }] },
    { id: 'old_archive' },
  ], points: [
    { point: 'old_useful', source_group: 'local_s6_cold', gate_accepted: true, visual_review_status: 'useful_complete' },
    { point: 'old_pending', source_group: 'automatic_shared_s6', gate_accepted: true },
    { point: 'fresh_fit', source_group: 'fresh_swaps', gate_accepted: false },
    { point: 'fresh_video', source_group: 'fresh_swaps', held: true },
    { point: 'archived', source_group: 'old_archive' },
  ] };
  const rows = C.currentAttempts(index);
  eq(rows.map((row) => row.point), ['old_useful', 'old_pending', 'fresh_fit', 'fresh_held', 'fresh_video']);
  assert(C.currentSelectionLabel(index).includes('1 curated labeled deliveries'));
  assert(C.currentSelectionLabel(index).includes('3 controlled-input views'));
  assert.strictEqual(C.resolvePointRequest(rows, C.currentHeldAttempts(index), { point: 'fresh_held' }).held.key, 'fresh_held');
});

ok('court mapping keeps the known near-left landmark on broadcast left', () => {
  const nearLeft = C.courtToWorld(0, 0, 0);
  const nearRight = C.courtToWorld(10.97, 0, 0);
  // A near-baseline camera looking along world +Z sees +world-X on screen left.
  assert(nearLeft[0] > nearRight[0]);
  eq(nearLeft, [0, 0, 0]);
  eq(nearRight, [-10.97, 0, 0]);
});

ok('ball uses its true radius at court height without poking through the floor', () => {
  const impact = C.ballPresentation(0.0325);
  assert.strictEqual(impact.radiusM, 0.033);
  assert.strictEqual(impact.centreHeightM, 0.033);
  assert(impact.centreHeightM - impact.radiusM >= 0);
  const flight = C.ballPresentation(1.2);
  assert.strictEqual(flight.radiusM, 0.09);
  assert.strictEqual(flight.centreHeightM, 1.2);
});

ok('splitRuns breaks on frame gaps', () => {
  eq(C.splitRuns([1, 2, 3, 7, 8, 20]), [[0, 2], [3, 4], [5, 5]]);
  eq(C.splitRuns([]), []);
  eq(C.splitRuns([5]), [[0, 0]]);
});

ok('splitRuns breaks at physical flight boundaries', () => {
  eq(C.splitRuns([1, 2, 3, 4], ['0', '0', '1', '1']), [[0, 1], [2, 3]]);
});

ok('gapBridges pairs run ends to next starts', () => {
  eq(C.gapBridges([1, 2, 5, 6]), [{ fromIdx: 1, toIdx: 2 }]);
  eq(C.gapBridges([1, 2, 3]), []);
});

ok('gapBridges exposes unresolved physical junctions', () => {
  eq(
    C.gapBridges([1, 2, 3, 4], ['0', '0', '1', '1']),
    [{ fromIdx: 1, toIdx: 2 }],
  );
});

ok('metricRange ignores nulls', () => {
  const r = C.metricRange([null, 3, 7, NaN, 5]);
  assert.strictEqual(r.lo, 3); assert.strictEqual(r.hi, 7);
  const flat = C.metricRange([2, 2, 2]);
  assert.strictEqual(flat.lo, 2); assert.strictEqual(flat.hi, 3);
});

ok('ramp endpoints and monotone channel movement', () => {
  const a = C.ramp(0), b = C.ramp(1), m = C.ramp(0.5);
  for (const c of [a, b, m]) {
    for (const ch of ['r', 'g', 'b']) { assert(c[ch] >= 0 && c[ch] <= 1); }
  }
  assert(b.r > a.r); // high end is warmer/redder than low end
});

ok('findApexes finds the peak of a parabola', () => {
  const frames = {
    frame: [10, 11, 12, 13, 14],
    x: [1, 1, 1, 1, 1], y: [5, 6, 7, 8, 9],
    z: [1.0, 2.0, 2.6, 2.0, 1.0], t: [0, 0.02, 0.04, 0.06, 0.08],
  };
  const ap = C.findApexes(frames);
  assert.strictEqual(ap.length, 1);
  assert.strictEqual(ap[0].index, 2);
  assert.strictEqual(ap[0].z, 2.6);
});

ok('findApexes respects run boundaries (no apex across a gap)', () => {
  const frames = {
    frame: [10, 11, 30, 31, 32],
    x: [0, 0, 0, 0, 0], y: [0, 0, 0, 0, 0],
    z: [1, 3, 1, 3, 1], t: [0, 1, 2, 3, 4],
  };
  const ap = C.findApexes(frames);
  assert.strictEqual(ap.length, 1); // only the interior peak of the 2nd run at index 3
  assert.strictEqual(ap[0].index, 3);
});

ok('netCrossings interpolates height at the net plane', () => {
  const netY = 11.885;
  const frames = {
    frame: [1, 2], x: [5, 5], y: [10.885, 12.885],
    z: [1.5, 1.7], t: [0, 0.02],
  };
  const netH = C.makeNetHeightFn(10.97, 0.914, 1.07);
  const cr = C.netCrossings(frames, netY, netH);
  assert.strictEqual(cr.length, 1);
  assert(Math.abs(cr[0].z - 1.6) < 1e-6); // midway between 1.5 and 1.7
  assert.strictEqual(cr[0].aboveNet, true);
  assert(cr[0].clearance > 0);
});

ok('makeNetHeightFn: center lower than posts', () => {
  const f = C.makeNetHeightFn(10.97, 0.914, 1.07);
  assert(Math.abs(f(5.485) - 0.914) < 1e-6);
  assert(Math.abs(f(0) - 1.07) < 1e-6);
  assert(Math.abs(f(10.97) - 1.07) < 1e-6);
});

ok('singles net uses regulation singles-post width', () => {
  const net = C.singlesNetGeometry(10.97, 1.37);
  assert(Math.abs(net.xMin - 0.456) < 1e-9);
  assert(Math.abs(net.xMax - 10.514) < 1e-9);
  assert(Math.abs(net.width - 10.058) < 1e-9);
});

ok('singles net has no collision surface beyond its posts', () => {
  const f = C.makeSinglesNetHeightFn(10.97, 1.37, 0.914, 1.07);
  assert.strictEqual(f(0.4), 0);
  assert.strictEqual(f(10.6), 0);
  assert(Math.abs(f(5.485) - 0.914) < 1e-6);
  assert(Math.abs(f(0.456) - 1.07) < 1e-6);
  assert(Math.abs(f(10.514) - 1.07) < 1e-6);
});

ok('frameIndexAtTime binary search', () => {
  const times = [0, 0.5, 1.0, 1.5, 2.0];
  assert.strictEqual(C.frameIndexAtTime(times, -1), 0);
  assert.strictEqual(C.frameIndexAtTime(times, 0.7), 1);
  assert.strictEqual(C.frameIndexAtTime(times, 1.0), 2);
  assert.strictEqual(C.frameIndexAtTime(times, 99), 4);
});

ok('ballAtTime interpolates within a run, flags gap edges', () => {
  const frames = {
    frame: [1, 2, 10], x: [0, 2, 100], y: [0, 0, 0], z: [1, 3, 9],
    t: [0.0, 0.02, 0.18],
  };
  const mid = C.ballAtTime(frames, 0.01);
  assert(Math.abs(mid.x - 1) < 1e-6 && Math.abs(mid.z - 2) < 1e-6);
  assert(!mid.gap);
  const inGap = C.ballAtTime(frames, 0.05); // between frame 2 and 10 (non-contiguous)
  assert.strictEqual(inGap.gap, true);
});

ok('playerAtTime interpolates supported rows and refuses stale gaps', () => {
  const side = {
    frame: [1, 2, 250], t: [0.0, 0.02, 5.0], x: [1, 2, 9], y: [0, 0, 0],
    pose_status: ['pose', 'lift_rejected', 'pose'], track_id: ['a', 'a', 'b'],
  };
  const near = C.playerAtTime(side, 0.01);
  assert(Math.abs(near.x - 1.5) < 1e-6);
  assert.strictEqual(near.pose_status, 'pose');
  assert.strictEqual(near.track_id, 'a');
  assert.strictEqual(C.playerAtTime(side, 0.5), null);
  assert.strictEqual(C.playerAtTime(side, 5.1), null);
});

ok('skeletonAtTime interpolates shared joints and drops stale poses', () => {
  const rows = [
    { t: 1.0, joints: { nose: [0, 0, 1, 0.9], left_wrist: [1, 0, 1, 0.8] } },
    { t: 1.04, joints: { nose: [0.2, 0, 1.2, 0.7] } },
  ];
  const pose = C.skeletonAtTime(rows, 1.02);
  assert(Math.abs(pose.joints.nose[0] - 0.1) < 1e-9);
  assert.strictEqual(pose.joints.nose[3], 0.7);
  assert.strictEqual(pose.joints.left_wrist, undefined);
  assert.strictEqual(C.skeletonAtTime(rows, 1.5), null);
});

ok('skeletonAtTime does not invent a pose across a missing native frame', () => {
  const rows = [
    { frame: 10, t: 0.4, joints: { nose: [0, 0, 1, 1] } },
    { frame: 12, t: 0.48, joints: { nose: [1, 0, 1, 1] } },
  ];
  assert.strictEqual(C.skeletonAtTime(rows, 0.44), null);
  assert(C.skeletonAtTime(rows, 0.4));
});

ok('contactClass tiers', () => {
  assert.strictEqual(C.contactClass({ fit: true, phase: 'serve' }), 'serve');
  assert.strictEqual(C.contactClass({ fit: true, phase: 'rally' }), 'rally');
  assert.strictEqual(C.contactClass({ fit: false, phase: 'rally' }), 'dead');
});

ok('timeBounds spans all event types', () => {
  const doc = {
    frames: { t: [1, 2, 3] },
    contacts: [{ t: 0.5 }], bounces: [{ t: 4.0 }],
    players: { near: { t: [0.2] }, far: { t: [4.5] } },
  };
  const b = C.timeBounds(doc);
  assert.strictEqual(b.lo, 0.2); assert.strictEqual(b.hi, 4.5);
});

ok('timelineTicks exposes label/fit offsets on the native scrubber', () => {
  const doc = {
    frames: { t: [1, 3] }, contacts: [], bounces: [],
    players: { near: { t: [] }, far: { t: [] } },
    timeline_ticks: [
      { frame: 25, t: 2, source: 'label', kind: 'contact' },
      { frame: 26, t: 2.04, source: 'fit', kind: 'contact' },
    ],
  };
  const ticks = C.timelineTicks(doc);
  assert.strictEqual(ticks.length, 2);
  assert.strictEqual(ticks[0].fraction, 0.5);
  assert(ticks[1].fraction > ticks[0].fraction);
});

ok('firstBallTime skips a leading no-ball span (the play-button root cause)', () => {
  // scrubber starts at t=5.0 (a player sample) but the ball is not observed until t=9.08;
  // firstBallTime finds where pressing PLAY actually shows a moving ball.
  const doc = {
    frames: { t: [5.0, 5.02, 9.08, 9.10], x: [null, null, 1.2, 1.3] },
  };
  assert(Math.abs(C.firstBallTime(doc) - 9.08) < 1e-9);
  assert.strictEqual(C.firstBallTime({ frames: { t: [], x: [] } }), null);
});

ok('frameNumberAtTime maps seconds -> 1-based frame file number', () => {
  assert.strictEqual(C.frameNumberAtTime(0.12, 50), 6);   // 0.12*50 = 6 -> f_0006
  assert.strictEqual(C.frameNumberAtTime(0.0, 50), 1);    // clamp to >= 1
  assert.strictEqual(C.frameNumberAtTime(1.0, 59.94005994), 60);
  assert.strictEqual(C.frameNumberAtTime(0.5, 0), null);  // no fps
});

ok('frameFileName zero-pads per pattern', () => {
  assert.strictEqual(C.frameFileName(6, 'f_%04d.jpg'), 'f_0006.jpg');
  assert.strictEqual(C.frameFileName(1234, 'f_%04d.jpg'), 'f_1234.jpg');
});

ok('frameUrl builds root-relative path, null when unavailable', () => {
  const doc = { match: 'rg2025f', clip: 'pt0006', frames_dir: 'rally_frames_50_1080_v2',
    frames_pattern: 'f_%04d.jpg', frames_available: true };
  assert.strictEqual(C.frameUrl(doc, 6), '/rg2025f/rally_frames_50_1080_v2/pt0006/f_0006.jpg');
  assert.strictEqual(
    C.frameUrl({ ...doc, frames_base: '/cross_match_event_audit_v4/rg2025f' }, 6),
    '/cross_match_event_audit_v4/rg2025f/rally_frames_50_1080_v2/pt0006/f_0006.jpg',
  );
  assert.strictEqual(C.frameUrl({ ...doc, frames_available: false }, 6), null);
  assert.strictEqual(C.frameUrl(doc, null), null);
});

ok('orderPoints: accepted first, then diagnostics best-junction-first', () => {
  const pts = [
    { point: 1, accepted: false, junction_med: 0.9, rms_med: 14 },
    { point: 2, accepted: true, junction_med: 0.01, rms_med: 3 },
    { point: 3, accepted: false, junction_med: 0.2, rms_med: 4 },
    { point: 4, accepted: false, junction_med: null, rms_med: null }, // no junction -> last
  ];
  const ord = C.orderPoints(pts);
  eq(ord.map((p) => p.point), [2, 3, 1, 4]);
  // input not mutated
  assert.strictEqual(pts[0].point, 1);
});

ok('shouldFollow honors a point-level forced progressive reveal', () => {
  const doc = {
    force_follow: true,
    counts: { contacts_fit: 2 },
    frames: { t: [0, 1], x: [1, 2] },
    contacts: [],
    bounces: [],
    players: { near: { t: [] }, far: { t: [] } },
  };
  assert.strictEqual(C.shouldFollow(doc, 'off'), true);
});

ok('defaultPoint picks first accepted, else best diagnostic', () => {
  const ordAcc = C.orderPoints([
    { point: 1, accepted: false, junction_med: 0.2 },
    { point: 2, accepted: true, junction_med: 0.5 },
  ]);
  assert.strictEqual(C.defaultPoint(ordAcc).point, 2);
  const ordNone = C.orderPoints([
    { point: 1, accepted: false, junction_med: 0.9 },
    { point: 3, accepted: false, junction_med: 0.1 },
  ]);
  assert.strictEqual(C.defaultPoint(ordNone).point, 3); // best diagnostic
  assert.strictEqual(C.defaultPoint([]), null);
});

ok('matchTag + scoreShort', () => {
  assert.strictEqual(C.matchTag('rg2025f'), 'RG');
  assert.strictEqual(C.matchTag('uso2025f'), 'USO');
  assert.strictEqual(C.scoreShort('pts 30-15 / games 0-0'), '30-15');
  assert.strictEqual(C.scoreShort('pts 40-AD / games 1-2'), '40-AD');
  assert.strictEqual(C.scoreShort(null), null);
});

ok('pointLabel joins score context and quality mark', () => {
  const entry = { match: 'rg2025f', point: 26, clip: 'pt0026', n_contacts: 33,
    n_contacts_fit: 19, duration_s: 16.98, accepted: true };
  assert.strictEqual(
    C.pointLabel(entry, { server: 'Sinner', score_state: 'pts 30-15 / games 0-0' }),
    'RG · point 26 · Sinner serving 30-15 · 19 shots · 17s · verified');
  // graceful without score context, and a rejected fit reads "diagnostic"
  assert.strictEqual(
    C.pointLabel({ match: 'uso2025f', point: 7, n_contacts: 2, duration_s: 8, accepted: false }, null),
    'USO · point 7 · 2 shots · 8s · diagnostic (fit rejected)');
});

ok('trailAlpha: playhead-time -> reveal/recent/ghost window', () => {
  const opts = { recent: 1.5, ghost: 0.2, fade: 0.5, full: 1 };
  // ahead of the playhead -> not yet revealed
  assert.strictEqual(C.trailAlpha(11.0, 10.0, opts), 0);
  // at and just behind the playhead, inside the recent window -> full
  assert.strictEqual(C.trailAlpha(10.0, 10.0, opts), 1);
  assert.strictEqual(C.trailAlpha(9.0, 10.0, opts), 1); // age 1.0 <= recent 1.5
  // well past the recent+fade band -> ghost floor
  assert.strictEqual(C.trailAlpha(5.0, 10.0, opts), 0.2); // age 5.0
  assert.strictEqual(C.trailAlpha(8.0, 10.0, opts), 0.2); // age 2.0 >= 1.5+0.5
  // inside the fade band -> between full and ghost, monotonically decreasing with age
  const mid = C.trailAlpha(8.25, 10.0, opts); // age 1.75, halfway through the 0.5s fade
  assert(Math.abs(mid - 0.6) < 1e-9);         // full + (ghost-full)*0.5 = 1 + (-0.8)*0.5
  assert(C.trailAlpha(8.4, 10.0, opts) > C.trailAlpha(8.1, 10.0, opts)); // younger -> brighter
  // defaults applied when opts omitted
  assert.strictEqual(C.trailAlpha(5, 4), 0);   // sampleT ahead of head
  assert.strictEqual(C.trailAlpha(4, 4), 1);   // at head -> full
  // null/NaN sample time (untimed element) falls back to the ghost floor, never invisible
  assert.strictEqual(C.trailAlpha(null, 10, opts), 0.2);
});

ok('isLongPoint: many shots OR many seconds trips it', () => {
  const short = { counts: { contacts_fit: 3 }, frames: { t: [0, 1, 2] },
    contacts: [], bounces: [], players: { near: { t: [] }, far: { t: [] } } };
  assert.strictEqual(C.isLongPoint(short), false);            // 3 shots, 2 s
  // long by shot count
  const manyShots = { counts: { contacts_fit: 31 }, frames: { t: [0, 1, 2] },
    contacts: [], bounces: [], players: { near: { t: [] }, far: { t: [] } } };
  assert.strictEqual(C.isLongPoint(manyShots), true);
  // long by duration alone (few shots, 17 s)
  const longSecs = { counts: { contacts_fit: 2 }, frames: { t: [0, 8.5, 17] },
    contacts: [], bounces: [], players: { near: { t: [] }, far: { t: [] } } };
  assert.strictEqual(C.isLongPoint(longSecs), true);
  // falls back to contacts when contacts_fit absent; thresholds overridable
  const byContacts = { counts: { contacts: 9 }, frames: { t: [0, 1] },
    contacts: [], bounces: [], players: { near: { t: [] }, far: { t: [] } } };
  assert.strictEqual(C.isLongPoint(byContacts), true);
  assert.strictEqual(C.isLongPoint(byContacts, { shots: 20 }), false);
});

ok('shouldFollow: pref overrides, else auto by length', () => {
  const longDoc = { counts: { contacts_fit: 31 }, frames: { t: [0, 17] },
    contacts: [], bounces: [], players: { near: { t: [] }, far: { t: [] } } };
  const shortDoc = { counts: { contacts_fit: 3 }, frames: { t: [0, 2] },
    contacts: [], bounces: [], players: { near: { t: [] }, far: { t: [] } } };
  assert.strictEqual(C.shouldFollow(longDoc, null), true);    // auto-on for long
  assert.strictEqual(C.shouldFollow(shortDoc, null), false);  // auto-off for short
  assert.strictEqual(C.shouldFollow(longDoc, 'off'), false);  // explicit off wins
  assert.strictEqual(C.shouldFollow(shortDoc, 'on'), true);   // explicit on wins
});

ok('deriveShots: contact intervals become independent review units', () => {
  const doc = {
    fps: 25,
    frames: { t: [1, 2, 3, 4] },
    contacts: [
      { frame: 25, t: 1, fit: true, side: 'near', phase: 'serve', speed_out: 30 },
      { frame: 75, t: 3, fit: true, side: 'far', phase: 'rally', speed_out: 20 },
    ],
    bounces: [{ t: 2, in_court: true }],
    players: { near: { t: [] }, far: { t: [] } },
  };
  const shots = C.deriveShots(doc);
  assert.strictEqual(shots.length, 2);
  assert.deepStrictEqual(
    [shots[0].start_frame, shots[0].end_frame, shots[0].bounce_count, shots[0].terminal],
    [25, 75, 1, false],
  );
  assert.strictEqual(shots[1].terminal, true);
  assert.strictEqual(shots[1].end_frame, 100);
});

ok('sliceDocumentToShot: synchronized point data is clipped to one shot', () => {
  const source = {
    fps: 25,
    frames: { frame: [25, 50, 75, 100], t: [1, 2, 3, 4], x: [1, 2, 3, 4] },
    contacts: [{ t: 1, fit: true }, { t: 3, fit: true }, { t: 4, fit: true }],
    bounces: [{ t: 2, in_court: true }, { t: 3.5, in_court: false }],
    players: {
      names: { near: 'A', far: 'B' },
      near: { t: [1, 2, 3, 4], x: [0, 1, 2, 3], y: [0, 1, 2, 3] },
      far: { t: [1, 2, 3, 4], x: [4, 3, 2, 1], y: [4, 3, 2, 1] },
    },
    skeletons: { near: [{ t: 1 }, { t: 2 }, { t: 4 }], far: [] },
    counts: {},
  };
  const shot = { shot_index: 0, start_t: 1, end_t: 3, start_frame: 25, end_frame: 75 };
  const clipped = C.sliceDocumentToShot(source, shot);
  assert.strictEqual(JSON.stringify(clipped.frames.frame), JSON.stringify([25, 50, 75]));
  assert.strictEqual(JSON.stringify(clipped.players.near.x), JSON.stringify([0, 1, 2]));
  assert.strictEqual(clipped.contacts.length, 2);
  assert.strictEqual(clipped.bounces.length, 1);
  assert.strictEqual(clipped.skeletons.near.length, 2);
  assert.strictEqual(clipped.counts.frames, 3);
  assert.strictEqual(clipped.review_context.score_start_t, 1);
  assert.strictEqual(clipped.review_context.score_end_t, 3);
  assert.strictEqual(C.reviewPhase(clipped, 0.9), 'pre');
  assert.strictEqual(C.reviewPhase(clipped, 2), 'scored');
  assert.strictEqual(C.reviewPhase(clipped, 3.1), 'post');
});

ok('local S6 keeps passive replay distinct from an accepted competitive flight', () => {
  const doc = {
    frames: { frame: [73, 74, 75, 76, 77], t: [27.88, 27.92, 27.96, 28, 28.04],
      segment: ['flight_00', 'flight_00', 'flight_00', 'flight_00', 'flight_00'] },
    review_context: { kind: 'local_s6_competitive', score_start_t: 27.46, score_end_t: 27.92,
      view_start_t: 27.08, view_end_t: 28.36 },
  };
  assert.strictEqual(C.reviewPhase(doc, 27.92), 'scored');
  assert.strictEqual(C.reviewPhase(doc, 27.920000001), 'scored');
  assert.strictEqual(C.reviewPhase(doc, 27.96), 'post');
  assert.strictEqual(C.reviewCaption(doc, 27.96), 'PASSIVE CONTEXT');
  assert.strictEqual(C.reviewCaption(doc, 27.9), 'COMPETITIVE ATTEMPT');
  assert.strictEqual(C.reviewCaption(doc, 27.1), 'PRE-ATTEMPT CONTEXT');
  eq(C.trajectoryRuns(doc), [[0, 1], [2, 4]]);
  delete doc.review_context;
  eq(C.trajectoryRuns(doc), [[0, 4]]);
  assert.strictEqual(C.reviewCaption(doc, 28), 'SCORED SHOT');
});

ok('current held point keys never silently select historical fits', () => {
  const historic = { point: 'held', file: 'old.json' };
  for (const status of ['pending', 'preparation_failed', 'execution_failed']) {
    const held = { key: 'held', status };
    eq(C.resolvePointRequest([historic], [held], { point: 'held' }),
      { entry: null, held, missingFile: false });
    eq(C.resolvePointRequest([historic], [held], { point: 'held', file: 'old.json' }),
      { entry: historic, held: null, missingFile: false });
  }
});

ok('explicit historical selection is stable and invalid documents do not fall back', () => {
  const current = { point: 'p', file: 'current.json', source_group: 'local_s6_cold' };
  const historic = { point: 'p', file: 'old.json' };
  eq(C.resolvePointRequest([current, historic], [], { point: 'p' }).entry, current);
  eq(C.resolvePointRequest([current, historic], [], { point: 'p', file: 'old.json' }).entry, historic);
  eq(C.resolvePointRequest([current, historic], [], { point: 'p', file: 'missing.json' }),
    { entry: null, held: null, missingFile: true });
  const shot = { point: 'p', file: 'old.json', shot_index: 2 };
  eq(C.resolvePointRequest([historic, shot], [], { file: 'old.json', shot: 2, shotReview: true }).entry, shot);
  const split = { point: 'second_serve', file: 'second.json', aliases: ['parent'], match: 'm' };
  eq(C.resolvePointRequest([split], [], { point: 'parent', match: 'm' }).entry, split);
  eq(C.resolvePointRequest([split], [], { point: 'parent', match: 'another' }).entry, null);
});

ok('current cohort excludes historical files and sorts useful, other gates, partial, blocked', () => {
  const fit = (point, gate, status) => ({point, file: `${point}.json`, source_group: 'local_s6_cold', gate_accepted: gate, visual_review_status: status});
  const index = {points: [fit('partial', false), {point: 'old', file: 'old.json'},
    fit('pending', true), fit('useful', true, 'useful_complete'),
    fit('approximate', true, 'useful_approximate'), fit('incorrect', true, 'incorrect')],
    local_s6: {held_before_fitting: [{key:'blocked', status:'execution_failed'}]}};
  const rows = C.currentAttempts(index);
  eq(rows.map(p => p.point), ['approximate', 'useful', 'pending', 'incorrect', 'partial', 'blocked']);
  assert(C.resolvePointRequest(rows, [], {file:'old.json'}).missingFile);
  assert(C.resolvePointRequest(rows, [], {point:'old'}).entry === null);
  assert(rows[5].held && rows[5].file === 'held:blocked');
  assert(index.points.length === 6);
});
ok('native context describes original bounds without extending the available interval', () => {
  const doc = {frame_range:[138,211], review_context:{score_start_frame:149, score_end_frame:159.5}};
  const summary = C.nativeContextSummary(doc);
  assert(summary.includes('f138–211') && summary.includes('f149.0–159.5'));
  assert(summary.includes('no reconstructed ball'));
  eq(doc.frame_range,[138,211]);
});

ok('current heading follows the exported run version, including future cohorts', () => {
  assert.strictEqual(C.currentCohortLabel({local_s6:{run:{path:'/runs/labeled_s6_cold_candidate_v9/manifest.json'},scope:'old candidate v8'}}), 'Current v9');
  assert.strictEqual(C.currentCohortLabel({local_s6:{scope:'Completed cold candidate v8 2026'}}), 'Current v8');
  assert.strictEqual(C.currentCohortLabel({local_s6:{}}), 'Current S6');
});

ok('an in-bounds first bounce does not hide the observed live ball', () => {
  const doc = {frame_range: [80, 160], review_context: {
    kind: 'local_s6_competitive', score_start_t: 4, score_end_t: 5.2,
    score_start_frame: 100, score_end_frame: 130,
    display_end_t: 5.64, display_end_frame: 141, observed_live_continuation: true,
  }};
  assert.strictEqual(C.reviewPhase(doc, 5.5), 'scored');
  assert.strictEqual(C.reviewPhase(doc, 5.64), 'scored');
  assert.strictEqual(C.reviewPhase(doc, 5.68), 'post');
  assert.strictEqual(C.reviewCaption(doc, 5.5), 'OBSERVED LIVE CONTINUATION');
  assert(C.nativeContextSummary(doc).includes('not the end of live play'));
  assert.strictEqual(doc.review_context.score_end_frame, 130);
});

console.log(`\n${passed} viewer_core checks passed`);

ok('held attempts with original video resolve to a playable empty document', () => {
  const held = { key: 'blocked', file: 'local_s6_blocked.json', frames_available: true };
  const rows = C.currentAttempts({ points: [], local_s6: { held_before_fitting: [held] } });
  eq(C.resolvePointRequest(rows, [held], { point: 'blocked' }),
    { entry: rows[0], held: null, missingFile: false });
  const doc = { review_context: { kind: 'local_s6_observations_only', view_start_t: 10, view_end_t: 12 }, frame_range: [1, 51] };
  eq(C.timeBounds(doc), { lo: 10, hi: 12 });
  assert(C.nativeContextSummary(doc).includes('No fitted 3D output'));
});

ok('automatic held video and capped attempts preserve separate source identity', () => {
  const held = {key:'auto_held',file:'automatic_s6_auto_held.json',frames_available:true,status:'preparation_held'};
  const index = {points:[{key:'labeled',point:'labeled',file:'local_s6_labeled.json',source_group:'local_s6_cold',gate_accepted:true}],
    local_s6:{held_before_fitting:[]},automatic_s6:{held_before_fitting:[held,{key:'auto_cap',status:'not_run_cap'}]}};
  const rows = C.currentAttempts(index), holds = C.currentHeldAttempts(index);
  assert.strictEqual(rows.length,2);
  assert(!rows.some((row) => row.key === 'auto_cap'));
  assert.strictEqual(holds.length,2);
  assert.strictEqual(C.resolvePointRequest(rows,holds,{point:'auto_held'}).entry.source_group,'automatic_shared_s6');
  assert.strictEqual(C.resolvePointRequest(rows,holds,{point:'auto_cap'}).held.status,'not_run_cap');
  assert.strictEqual(C.currentCohortLabel({automatic_s6:{}}),'Automatic S6');
  assert(C.nativeContextSummary({frame_range:[1,20],review_context:{kind:'automatic_observations_only'}}).includes('No fitted 3D output'));
});

ok('automatic held and unknown-ending scope never announce a scored shot', () => {
  assert.strictEqual(C.reviewCaption({review_context:{kind:'automatic_observations_only'}},1),'ORIGINAL VIDEO · NO FIT');
  assert.strictEqual(C.reviewCaption({review_context:{kind:'local_s6_observation_scope'}},1),'OBSERVED SCOPE · ENDING UNKNOWN');
});

ok('unresolved physical prefix displays actual modeled support without inferring stroke or ending', () => {
  const doc = {review_context: {kind:'local_s6_observation_scope',
    score_start_t:10, score_end_t:12.0000004, display_end_t:12.0000004,
    modeled_horizon_frame:67}};
  assert.strictEqual(C.reviewCaption(doc,11), 'MODELED THROUGH f67 · ENDING UNKNOWN');
  assert.strictEqual(C.reviewCaption(doc,9), 'ORIGINAL VIDEO · BEFORE MODELED SCOPE');
  assert.strictEqual(C.reviewCaption(doc,13), 'ORIGINAL VIDEO · AFTER MODELED SCOPE');
  assert.strictEqual(C.reviewPhase(doc,12.000001), 'scored');
  assert.strictEqual(C.contactClass({fit:true,phase:'contact'}), 'rally');
});

ok('unresolved native context describes modeled support instead of competitive scope', () => {
  const summary = C.nativeContextSummary({frame_range:[1,100],review_context:{kind:'local_s6_observation_scope',score_start_frame:50.64,score_end_frame:67}});
  assert(summary.includes('Modeled support: f50.6–67.0'));
  assert(summary.includes('Competitive ending remains unknown'));
  assert(!summary.includes('Competitive scope'));
});


ok('automatic source counts distinguish split deliveries and respect visible subset', () => {
  const index = {points:[{point:'source_a__first',source_group:'automatic_shared_s6'}],
    automatic_s6:{original_parents:[{key:'source_a',children:['source_a__first','source_a__second']},
      {key:'source_b',children:[]}],held_before_fitting:[{key:'source_a__second',status:'execution_failed'}]}};
  assert.strictEqual(C.currentSelectionLabel(index),'1 source attempt · 2 delivery segments');
  index.points.push({point:'source_b',source_group:'automatic_shared_s6'});
  assert.strictEqual(C.currentSelectionLabel(index),'2 source attempts · 3 delivery segments');
  assert.strictEqual(C.currentAttempts(index).length,3);
  assert.strictEqual(C.currentSelectionLabel({points:[],local_s6:{held_before_fitting:[]}}),'0 selected attempts');
});


ok('exact zero-based native clocks preserve quantized frame epochs and interpolation', () => {
  const doc = {fps: 60000 / 1001, native_timebase: {
    mapping: 'piecewise_linear_original_native_pts', origin_frame: 0, origin_t: 2918.616,
    native_frames: [0, 1, 2, 3], native_times_seconds: [2918.616, 2918.633, 2918.649, 2918.666],
  }};
  assert.strictEqual(C.nativeFrameAtTime(doc, 2918.616), 0);
  assert.strictEqual(C.nativeFrameAtTime(doc, 2918.649), 2);
  assert.strictEqual(C.nativeFrameAtTime(doc, 2918.640), 1);
  assert.strictEqual(C.nativeFrameAtTime(doc, 2918.642), 2);
  assert(Math.abs(C.nativeTimeForFrame(doc, 1.5) - 2918.641) < 1e-9);
  assert.strictEqual(C.nativeTimeForFrame(doc, -1), 2918.616);
  assert.strictEqual(C.nativeTimeForFrame(doc, 4), 2918.666);
  assert.strictEqual(C.frameFileName(0, 'f_%04d.jpg'), 'f_0000.jpg');
});

ok('regular legacy native mapping remains one-based with its original offset', () => {
  const doc = {fps: 25, native_timebase: {origin_frame: 1, origin_t: 9}};
  assert.strictEqual(C.nativeFrameAtTime(doc, 9), 1);
  assert.strictEqual(C.nativeTimeForFrame(doc, 233), 18.28);
  assert.strictEqual(C.nativeFrameAtTime({fps: 25}, 0), 1);
});


ok('validation panel keeps fixed source cases distinct from delivery and context views', () => {
  const index = {validation_panel: {label: 'Independent validation V21', source_cases: 50, submitted_deliveries: 53, source_context_views: 17}};
  assert.strictEqual(C.currentCohortLabel(index), 'Independent validation V21');
  assert.strictEqual(C.currentSelectionLabel(index), '50 source cases · 53 submitted deliveries · 17 source context views');
});

ok('visual rejection preserves passing flight gates without claiming source completion', () => {
  const status = C.reconstructionStatus({
    quality: {accepted: false, visual_review_status: 'incorrect_complete'},
    per_flight: {complete_point: true, partial_point: false, accepted_flight_count: 7, flight_count: 7},
  });
  assert(status.held && status.visualIncorrect);
  assert(status.title.includes('VISUAL REVIEW'));
  assert.strictEqual(status.counts, '7/7 flights pass stage gates');
  assert.strictEqual(status.gateSummary, 'All modeled flights pass gates');
  assert(!status.title.includes('NO FLIGHT'));
  const gateOnly = C.reconstructionStatus({quality: {accepted: null}, per_flight: {
    complete_point: true, accepted_flight_count: 7, flight_count: 7,
  }});
  assert(!gateOnly.held);
  assert.strictEqual(gateOnly.gateSummary, status.gateSummary);
});

ok('zero and partial flight gates remain distinct from visual review', () => {
  const make = count => C.reconstructionStatus({quality: {accepted: false}, per_flight: {
    complete_point: false, accepted_flight_count: count, flight_count: 3,
  }});
  assert.strictEqual(make(0).title, 'NO FLIGHT PASSES STAGE GATES');
  assert.strictEqual(make(2).title, 'PARTIAL RECONSTRUCTION');
  assert.strictEqual(make(2).counts, '2/3 flights pass stage gates');
});

ok('opened50 prefixes require explicit inclusion and keep separate incomplete counts', () => {
  const prefix = {point:'opened', file:'opened.json', source_group:'opened50_partial_s6',
    gate_accepted:false, accepted_flights:2, flight_count:2, declared_flights:3};
  const index = {points:[
    {point:'labeled', file:'labeled.json', source_group:'local_s6_cold', gate_accepted:true},
    {point:'automatic', file:'automatic.json', source_group:'automatic_shared_s6', gate_accepted:true},
    prefix,
  ], automatic_s6:{}, local_s6:{}};
  assert.strictEqual(C.currentAttempts(index).length, 2);
  index.opened50_partial_s6 = {cases:1, original_sources:50, complete_source_credit:0};
  const rows = C.currentAttempts(index);
  assert.strictEqual(rows.length, 3);
  assert.strictEqual(C.currentAttemptGroup(prefix), 2);
  assert.strictEqual(C.resolvePointRequest(rows, [], {point:'opened'}).entry.file, 'opened.json');
  assert.strictEqual(C.currentSelectionLabel(index),
    '1 curated labeled deliveries · 1 opened50 partial reconstructions · automatic: 1 delivery segment');
});

ok('opened50 current deliveries include playable holds without replacing other cohorts', () => {
  const opened = {point:'opened', file:'opened.json', source_group:'opened50_current_s6',
    gate_accepted:true, accepted_flights:2, flight_count:2};
  const held = {key:'missing', file:'missing.json', status:'execution_failed',
    source_group:'opened50_current_s6', frames_available:true};
  const automatic = {point:'automatic', file:'automatic.json', source_group:'automatic_shared_s6'};
  const index = {points:[opened, automatic], automatic_s6:{},
    opened50_current_s6:{held_before_fitting:[held]}};
  const rows = C.currentAttempts(index);
  assert.strictEqual(rows.length, 3);
  assert.strictEqual(C.currentHeldAttempts(index)[0].file, 'missing.json');
  assert.strictEqual(rows.at(-1).point, 'missing');
  assert.strictEqual(C.currentSelectionLabel(index),
    '2 opened50 current deliveries · automatic: 1 delivery segment');
  assert.strictEqual(C.resolvePointRequest(rows, C.currentHeldAttempts(index),
    {point:'missing'}).entry.file, 'missing.json');
});

function componentFixture() {
  const mapping = {flights: [3, 4], contacts: [3, 4, 5], events: [6, 7, 8]};
  const component = {component_index: 0, original_contact_indices: [3, 4, 5], local_to_original: mapping};
  const entry = {point: 'source__component_0', parent_source_key: 'source', component_index: 0,
    file: 'source_component_0.json', count_in_source_denominator: false,
    local_to_original: mapping, n_frames: 31};
  const source = {point: 'source', match: 'match', source_clip: 'clip', clip: 'parent_clip', fps: 25,
    component_scenes: [entry], complete_original_source: false, continuity_between_components: false,
    original_source_plan: {plan_sha256: 'hash', components: [component],
      original_events: [1, 10, 20, 30, 45, 60].map(frame => ({event_type: 'contact', frame})),
      original_slots: [{start_frame: 20, component_index: null}]},
    frame_range: [1, 100], frames_dir: 'source', frames_base: 'frames', frames_pattern: 'f_{frame}.jpg',
    native_timebase: {native_frames: [1, 50, 100], native_times_seconds: [10, 12, 14]},
    review_context: {view_start_t: 10, view_end_t: 14}};
  const child = {point: entry.point, match: 'match', source_clip: 'clip', clip: 'child_clip', fps: 25,
    contact_component_scope: {plan_sha256: 'hash', component},
    native_timebase: source.native_timebase, frames_dir: 'child', frame_range: [1, 100],
    frames: {frame: [1, 30, 45, 60, 100], t: [10, 11, 11.6, 12.4, 14],
      x: [null, 1, 2, 3, null], y: [null, 1, 2, 3, null], z: [null, 1, 2, 3, null]},
    review_context: {complete_original_source: false, score_start_t: 11, score_end_t: 12.4,
      view_start_t: 10, view_end_t: 14}, per_flight: {complete_point: false, accepted_flight_count: 2, flight_count: 2}};
  return {source, entry, child};
}

ok('component navigation retains parent native clock and gaps without stitching geometry', () => {
  const {source, entry, child} = componentFixture();
  const spans = C.supportedSpans(source);
  assert.strictEqual(spans[0].startFrame, 30);
  assert.strictEqual(spans[0].endFrame, 60);
  const view = C.componentView(source, entry, child);
  assert.strictEqual(view.point, 'source');
  assert.strictEqual(view.frames_dir, source.frames_dir);
  assert.strictEqual(view.native_timebase, source.native_timebase);
  assert.strictEqual(view.frame_range, source.frame_range);
  assert.strictEqual(view.frames, child.frames);
  assert.strictEqual(view.frames.x[0], null);
  assert.strictEqual(view.frames.x.at(-1), null);
  assert.strictEqual(view.original_source_plan, source.original_source_plan);
  assert.strictEqual(C.reconstructionStatus(view).held, true);
  assert.strictEqual(C.reconstructionStatus(view).title, 'INCOMPLETE ORIGINAL ATTEMPT');
  assert.strictEqual(child.point, entry.point);
});

ok('a ball-track terminal component spans its one contact to the supplied ending', () => {
  const {source} = componentFixture();
  const spec = source.original_source_plan.components[0];
  spec.original_contact_indices = [3];
  spec.terminal_track_endpoint = {kind: 'exit', frame: 52};
  assert.strictEqual(C.supportedSpans(source)[0].endFrame, 52);
  delete spec.terminal_track_endpoint;
  assert.throws(() => C.supportedSpans(source), /Invalid original contact span/);
  spec.terminal_track_endpoint = {kind: 'exit', frame: 20};
  assert.throws(() => C.supportedSpans(source), /Invalid original contact span/);
});

ok('component loading refuses arbitrary files, swapped source identities and changed clocks', () => {
  for (const mutate of [
    f => {f.entry.file = '../foreign.json';},
    f => {f.entry.parent_source_key = 'foreign';},
    f => {f.child.point = 'foreign';},
    f => {f.child.contact_component_scope.plan_sha256 = 'foreign';},
    f => {f.child.contact_component_scope = {...f.child.contact_component_scope, component: {...f.child.contact_component_scope.component, local_to_original: {flights: [99]}}};},
    f => {f.child.native_timebase = {...f.child.native_timebase, native_times_seconds: [0, 1, 2]};},
    f => {f.child.per_flight.complete_point = true;},
    f => {f.child.frame_range = [30, 60];},
    f => {f.child.frames.x.pop();},
    f => {f.child.frames.frame[0] = 2;},
    f => {f.source.original_source_plan.components[0].original_contact_indices = [];},
    f => {f.source.original_source_plan.components[0].original_contact_indices = [3, 99];},
    f => {f.source.original_source_plan.components[0].original_contact_indices = [4, 3];},
  ]) {
    const f = componentFixture(); mutate(f);
    assert.throws(() => C.componentView(f.source, f.entry, f.child));
  }
  const f = componentFixture();
  assert.throws(() => C.componentView(f.source, {...f.entry, file: 'unlisted.json'}, f.child));
  assert.strictEqual(C.resolvePointRequest([{point: 'source', file: 'source.json'}], [],
    {file: f.entry.file}).missingFile, true);
});

ok('component children including failed children never become additional current attempts', () => {
  const f = componentFixture();
  const index = {local_s6: {held_before_fitting: [{...f.entry, held: true}]}, points: [
    {point: 'source', file: 'source.json', source_group: 'local_s6_cold'},
    {...f.entry, source_group: 'local_s6_cold'},
  ]};
  assert.strictEqual(C.currentAttempts(index).length, 1);
  assert.strictEqual(C.resolvePointRequest(index.points, [], {file: f.entry.file}).missingFile, true);
  assert.strictEqual(C.currentHeldAttempts(index).length, 0);
  f.entry.held = true;
  assert.strictEqual(C.supportedSpans(f.source)[0].available, false);
  assert.throws(() => C.componentView(f.source, f.entry, f.child));
  assert.strictEqual(C.supportedSpans({point: 'ordinary'}).length, 0);
});

ok('flight review deep links keep an existing point ahead of an arm', () => {
  const existing = {point: 'p', file: 'old.json', match: 'm'};
  const aa = {point: 'p', file: 'aa.json', match: 'm', arm: 'AA', source_group: 'flight_review'};
  const al = {point: 'p', file: 'al.json', match: 'm', arm: 'AL', source_group: 'flight_review'};
  assert.strictEqual(C.resolvePointRequest([aa, al, existing], [], {point: 'p'}).entry.file, 'old.json');
  assert.strictEqual(C.resolvePointRequest([aa, al, existing], [], {point: 'p', arm: 'AL'}).entry.file, 'al.json');
  assert.strictEqual(C.resolvePointRequest([aa], [], {point: 'p'}).entry.arm, 'AA');
  eq(C.projectCamera([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 1]], [2, 4, 1]), [1, 2]);
  const choices = C.flightReviewChoices([
    {source_group: 'flight_review', panel: 'panel_0', arm: 'AA', file: 'a.json',
      flights: [{id: 'a:e1', outcome: 'matched', cause: null}, {id: 'a:e2', outcome: 'missed', cause: 'net'}]},
    {source_group: 'flight_review', panel: 'panel_A', arm: 'AL', file: 'b.json',
      flights: [{id: 'b:e1', outcome: 'extra', cause: 'extra competitive accept'}]},
  ], {panel: 'panel_0', outcome: 'missed', cause: 'net'});
  assert.strictEqual(choices.length, 1);
  assert.strictEqual(choices[0].flight.id, 'a:e2');
});
ok('navigation walks audit set, match, point, and the next flight', () => {
  const entries = [
    {source_group: 'flight_review', panel: 'panel_0', point: 'source07_case001_short_a1', arm: 'AA',
      file: 'aa.json', source_id: 'source07', tournament: 'Indian Wells', players: 'Iga Swiatek – Maria Sakkari',
      audit_set: 'fresh', flights: [
        {id: 'source07_case001_short_a1:e004', outcome: 'matched', cause: null, origin_frame: 700},
        {id: 'source07_case001_short_a1:e005', outcome: 'missed', cause: 'bounce_count_timing', origin_frame: 800},
      ]},
    {source_group: 'flight_review', panel: 'panel_C', point: 'source04_point001_a1', arm: 'AL',
      file: 'c.json', source_id: 'source04', tournament: 'Roland Garros', players: 'Djokovic – Nadal',
      audit_set: 'panel_c', flights: [
        {id: 'source04_point001_a1:A1_C1', outcome: 'matched', cause: null, origin_frame: 10},
      ]},
    {source_group: 'local_s6_cold', panel: 'old', point: 'ao2019f_pt0001', file: 'old.json'},
  ];
  assert.strictEqual(C.auditSetForPanel('panel_A'), 'fresh');
  assert.strictEqual(C.flightEntriesForSet(entries, 'panel_c').length, 1);
  assert.strictEqual(C.flightEntriesForSet(entries, 'error_themes').length, 1);
  assert.strictEqual(C.searchFlightEntries(entries, 'swiatek').length, 1);
  const matches = C.matchChoices(C.flightEntriesForSet(entries, 'fresh'));
  assert.strictEqual(matches[0].title, 'Indian Wells · Iga Swiatek – Maria Sakkari');
  const chips = C.armChips(C.attemptChoices(entries, 'source07')[0].arms);
  assert.strictEqual(chips[0].counts.missed, 1);
  assert.strictEqual(chips[0].causes[0], 'bounce_count_timing');
  const next = C.stepReview(C.flightEntriesForSet(entries, 'all'), 'aa.json', 'source07_case001_short_a1:e004', 1);
  assert.strictEqual(next.flight.id, 'source07_case001_short_a1:e005');
  assert(C.overlayUserUnits(1920, 1080, 480, 270, 8) > 20);
});

console.log(passed + ' viewer_core checks finished');
