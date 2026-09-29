// viewer_core.js — pure data logic for the 3D point viewer (NO THREE.js, NO DOM).
// Kept dependency-free so it is unit-testable in a Node VM (see test_viewer_core.cjs)
// and so the rendering module (viewer.js) can stay focused on the WebGL scene.
//
// Court frame throughout: x across [0,10.97], y along [0,23.77], z up (RICH_CHART_SPEC).

export function clamp(v, lo, hi) { return v < lo ? lo : (v > hi ? hi : v); }
export function lerp(a, b, t) { return a + (b - a) * t; }

// Child scenes are navigable only through their original source's declared roster.
// They are views within one attempt, never additional picker points.
export function supportedSpans(source) {
  if (!Array.isArray(source?.component_scenes)) return [];
  const plan = source.original_source_plan;
  if (!plan?.plan_sha256 || !Array.isArray(plan.components)
      || source.component_scenes.length !== plan.components.length
      || source.complete_original_source !== false
      || source.continuity_between_components !== false) {
    throw new Error('Invalid original component source');
  }
  const contacts = (plan.original_events || []).filter((event) => event.event_type === 'contact');
  if (new Set(plan.components.map((c) => c.component_index)).size !== plan.components.length) {
    throw new Error('Duplicate planned component');
  }
  const seen = new Set();
  const files = new Set();
  return source.component_scenes.map((entry) => {
    const index = entry.component_index;
    const spec = plan.components.find((c) => c.component_index === index);
    if (!Number.isInteger(index) || seen.has(index) || files.has(entry.file) || !spec
        || entry.parent_source_key !== source.point
        || entry.count_in_source_denominator !== false
        || !/^[A-Za-z0-9_.-]+\.json$/.test(entry.file || '') || entry.file.includes('..')
        || JSON.stringify(entry.local_to_original) !== JSON.stringify(spec.local_to_original)) {
      throw new Error('Invalid source-bound component entry');
    }
    seen.add(index);
    files.add(entry.file);
    const indices = spec.original_contact_indices;
    // A component closed by its own ball-track terminal has one original contact
    // and ends on that supplied (still unresolved) ending, not on a contact.
    const terminal = spec.terminal_track_endpoint;
    const terminalFrame = Number(terminal?.frame);
    if (!Array.isArray(indices) || indices.length < (terminal ? 1 : 2) || indices.some((i, j) =>
      !Number.isInteger(i) || i < 0 || i >= contacts.length
      || !Number.isFinite(contacts[i].frame) || (j > 0 && i <= indices[j - 1]))
      || (terminal && !(Number.isFinite(terminalFrame) && terminalFrame > contacts[indices.at(-1)].frame))) {
      throw new Error('Invalid original contact span');
    }
    return { ...entry, startFrame: contacts[indices[0]].frame,
      endFrame: terminal ? terminalFrame : contacts[indices.at(-1)].frame,
      available: !entry.held && entry.n_frames > 0 };
  }).sort((a, b) => a.component_index - b.component_index);
}

export function componentView(source, entry, child) {
  const bound = supportedSpans(source).find((row) => row.component_index === entry.component_index);
  const scope = child?.contact_component_scope;
  if (!bound || bound.file !== entry.file || !bound.available
      || child.point !== bound.point || child.match !== source.match
      || child.source_clip !== source.source_clip
      || child.fps !== source.fps
      || JSON.stringify(child.frame_range) !== JSON.stringify(source.frame_range)
      || !Array.isArray(child.frames?.frame) || child.frames.frame.length === 0
      || ['t', 'x', 'y', 'z'].some((key) => !Array.isArray(child.frames[key])
        || child.frames[key].length !== child.frames.frame.length)
      || child.frames.frame[0] !== source.frame_range?.[0]
      || child.frames.frame.at(-1) !== source.frame_range?.[1]
      || JSON.stringify(child.native_timebase?.native_frames) !== JSON.stringify(source.native_timebase?.native_frames)
      || JSON.stringify(child.native_timebase?.native_times_seconds) !== JSON.stringify(source.native_timebase?.native_times_seconds)
      || scope?.plan_sha256 !== source.original_source_plan.plan_sha256
      || scope.component?.component_index !== bound.component_index
      || JSON.stringify(scope.component?.local_to_original) !== JSON.stringify(bound.local_to_original)
      || child.review_context?.complete_original_source !== false
      || child.per_flight?.complete_point !== false) {
    throw new Error('Child scene differs from its original source binding');
  }
  // Geometry stays exactly the selected child's, including null samples outside
  // its span. Native video, clock, source identity and gaps belong to the parent.
  const view = { ...child, point: source.point, clip: source.clip,
    component_source: source, selected_component: bound.component_index,
    complete_original_source: false, continuity_between_components: false };
  for (const key of ['frames_base', 'frames_dir', 'frames_pattern', 'frames_available',
    'frame_range', 'native_timebase', 'fps', 'component_scenes', 'original_source_plan',
    'original_source_verdict', 'reference_source_summary', 'reference_attempt_score',
    'optional_candidate_status']) view[key] = source[key];
  view.review_context = { ...child.review_context,
    view_start_t: source.review_context.view_start_t,
    view_end_t: source.review_context.view_end_t, complete_original_source: false };
  return view;
}

// Mixed-input research can use automatic ball centers with labeled event provenance.
export function observedBallLegend(doc) {
  const marker = doc?.video_overlay?.labeled_marker;
  if (typeof marker === 'string' && marker.trim()) {
    return `● ${marker.replace(/^blue dot:\s*/i, '').trim()}`;
  }
  const ball = doc?.input_swap_scope?.ball;
  if (typeof ball === 'string' && ball.trim()) {
    return `● ${ball.trim()} observed center`;
  }
  return doc?.provenance?.observation_origin === 'automatic'
    ? '● automatic center' : '● labeled click';
}

// Numerical gates and visual usefulness are different facts, including when all
// emitted flights pass but an original serve was omitted from reconstruction.
export function reconstructionStatus(doc) {
  const quality = doc.quality || {};
  const flights = doc.per_flight;
  const visualIncorrect = String(quality.visual_review_status || '').startsWith('incorrect');
  const partialSource = Array.isArray(doc.component_scenes);
  const held = quality.accepted === false || visualIncorrect || partialSource;
  const count = flights?.accepted_flight_count;
  const total = flights?.flight_count;
  const known = Number.isFinite(count) && Number.isFinite(total);
  const counts = known ? `${count}/${total} flights pass stage gates` : 'Flight gate counts unavailable';
  let title = 'RECONSTRUCTION HELD FOR REVIEW';
  let detail = 'Inspect the diagnostic reconstruction and its review notes.';
  if (visualIncorrect) {
    title = 'VISUAL REVIEW FOUND AN INCORRECT RECONSTRUCTION';
    detail = `${counts}. Passing gates do not establish a useful complete point.`;
  } else if (known && count === 0) {
    title = 'NO FLIGHT PASSES STAGE GATES';
    detail = 'The diagnostic candidate is shown to help inspect the failure.';
  } else if (known) {
    title = count === total ? 'RECONSTRUCTION HELD DESPITE PASSING FLIGHT GATES' : 'PARTIAL RECONSTRUCTION';
    detail = `${counts}. Whole-point review and completeness are separate.`;
  }
  if (partialSource) {
    title = 'INCOMPLETE ORIGINAL ATTEMPT';
    detail = 'Independent supported spans only. Unresolved origins remain missing; no complete-point claim.';
  }
  const gateSummary = partialSource ? 'Partial source · independent spans and missing gaps' : flights?.complete_point
    ? 'All modeled flights pass gates'
    : doc.review_context?.kind === 'local_s6_observation_scope'
      ? 'Partial reconstruction · ending unknown'
      : known && count > 0 ? 'Partial flight-gate coverage' : 'No passing flight gates';
  return { held, visualIncorrect, title, detail, gateSummary, counts };
}

// A current held attempt must not fall through to an older fit with the same point key.
// An explicit document selection keeps the historical archive independently accessible.
export function resolvePointRequest(points, heldRows, request) {
  points = points.filter((entry) => !entry.parent_source_key);
  heldRows = (heldRows || []).filter((entry) => !entry.parent_source_key);
  if (request.file) {
    const entry = points.find((p) => p.file === request.file
      && (!request.shotReview || request.shot == null
        || String(p.shot_index) === String(request.shot))) || null;
    return { entry, held: null, missingFile: !entry };
  }
  const held = request.point == null ? null
    : (heldRows || []).find((row) => String(row.key) === String(request.point));
  if (held) {
    const entry = held.frames_available && held.file
      ? points.find((p) => p.held && p.file === held.file) || null : null;
    return { entry, held: entry ? null : held, missingFile: false };
  }
  const matchesPoint = (p) => (String(p.point) === String(request.point)
      || (p.aliases || []).some((alias) => String(alias) === String(request.point)))
    && (!request.match || p.match === request.match)
    && (!request.shotReview || request.shot == null
      || String(p.shot_index) === String(request.shot));
  if (request.arm) {
    const entry = request.point == null ? null : points.find((p) =>
      matchesPoint(p) && p.arm === request.arm) || null;
    return { entry, held: null, missingFile: false };
  }
  const matches = request.point == null ? [] : points.filter(matchesPoint);
  // A flight-review arm must not hide an existing point that uses the same id.
  const entry = matches.find((p) => !p.arm) || matches.find((p) => p.arm === 'AA') || matches[0] || null;
  return { entry, held: null, missingFile: false };
}

// Project a court-space point through the fitter's native 3x4 camera.
export function projectCamera(matrix, xyz) {
  if (!matrix || !xyz || matrix.length < 3) return null;
  const x = Number(xyz[0]), y = Number(xyz[1]), z = Number(xyz[2]);
  const row = (index) => {
    const line = matrix[index];
    return line[0] * x + line[1] * y + line[2] * z + line[3];
  };
  const w = row(2);
  if (!Number.isFinite(w) || Math.abs(w) < 1e-9) return null;
  const u = row(0) / w, v = row(1) / w;
  if (!Number.isFinite(u) || !Number.isFinite(v)) return null;
  return [u, v];
}

// Flight-review picker rows. Outcome and cause filters apply per flight, so a
// missed flight stays visible when the rest of its attempt matched.
export function flightReviewChoices(entries, filters) {
  const wanted = filters || {};
  const choices = [];
  for (const entry of entries || []) {
    if (entry.source_group && entry.source_group !== 'flight_review') continue;
    if (wanted.panel && wanted.panel !== 'all' && entry.panel !== wanted.panel) continue;
    if (wanted.arm && wanted.arm !== 'all' && entry.arm !== wanted.arm) continue;
    let flights = entry.flights || [];
    if (wanted.outcome && wanted.outcome !== 'all') {
      flights = flights.filter((flight) => flight.outcome === wanted.outcome);
    }
    if (wanted.cause && wanted.cause !== 'all') {
      flights = flights.filter((flight) => flight.cause === wanted.cause);
    }
    if ((wanted.outcome && wanted.outcome !== 'all') || (wanted.cause && wanted.cause !== 'all')) {
      if (!flights.length) continue;
    }
    if (!flights.length) {
      choices.push({ entry, flight: null });
      continue;
    }
    for (const flight of flights) choices.push({ entry, flight });
  }
  return choices;
}

// Show only current labeled and automatic attempts, with their input origins explicit.
export function currentCohortLabel(index) {
  if (index.validation_panel) return index.validation_panel.label;
  if (index.automatic_s6 && !index.local_s6) return 'Automatic S6';
  const metadata = index.local_s6 || {};
  const source = metadata.run?.path || metadata.run_label || metadata.scope || '';
  const version = source.match(/candidate[ _-](v\d+)\b/i)?.[1];
  if (index.automatic_s6) return `${version ? `Labeled ${version.toLowerCase()}` : 'Labeled S6'} + automatic`;
  return version ? `Current ${version.toLowerCase()}` : 'Current S6';
}

export function currentAttemptGroup(entry) {
  if (entry.held) return 3;
  if (!entry.gate_accepted) return 2;
  if (['incorrect', 'incorrect_complete', 'uncertain'].includes(entry.visual_review_status)) return 2;
  return ['useful_complete', 'useful_approximate'].includes(entry.visual_review_status) ? 0 : 1;
}

export function currentHeldAttempts(index) {
  return [
    ...(index.local_s6?.held_before_fitting || []),
    ...(index.opened50_current_s6?.held_before_fitting || []),
    ...(index.automatic_s6?.held_before_fitting || []).map((row) => ({
      ...row, source_group: 'automatic_shared_s6',
    })),
    ...(index.source_groups || []).filter((group) => group.current).flatMap((group) =>
      (group.held_before_fitting || []).map((row) => ({ ...row, source_group: group.id }))),
  ].filter((row) => !row.parent_source_key);
}

export function currentAttempts(index) {
  const groups = new Set(['local_s6_cold', 'automatic_shared_s6']);
  if (index.opened50_partial_s6) groups.add('opened50_partial_s6');
  if (index.opened50_current_s6) groups.add('opened50_current_s6');
  for (const group of index.source_groups || []) if (group.current) groups.add(group.id);
  const fits = (index.points || []).filter((p) => groups.has(p.source_group) && !p.parent_source_key);
  const held = currentHeldAttempts(index).filter((row) => row.status !== 'not_run_cap').map((row) => ({
    ...row, point: row.key, held: true, file: row.file || `held:${row.key}`,
    source_group: row.source_group || 'local_s6_cold', label: `${row.key} · ${row.status === 'not_run_cap' ? 'not run (cap)' : 'no fitted output'}`,
  }));
  return fits.concat(held).sort((a, b) => currentAttemptGroup(a) - currentAttemptGroup(b)
    || String(a.point).localeCompare(String(b.point)));
}

export function currentSelectionLabel(index) {
  if (index.validation_panel) {
    const panel = index.validation_panel;
    return `${panel.source_cases} source cases · ${panel.submitted_deliveries} submitted deliveries · ${panel.source_context_views} source context views`;
  }
  const rows = currentAttempts(index);
  if (!index.automatic_s6) return `${rows.length} selected attempt${rows.length === 1 ? '' : 's'}`;
  const automatic = rows.filter((row) => row.source_group === 'automatic_shared_s6');
  const keys = new Set(automatic.map((row) => row.point));
  const parents = (index.automatic_s6.original_parents || []).filter((row) =>
    keys.has(row.key) || (row.children || []).some((key) => keys.has(key)));
  const source = parents.length ? `${parents.length} source attempt${parents.length === 1 ? '' : 's'} · ` : '';
  const partial = rows.filter((row) => row.source_group === 'opened50_partial_s6').length;
  const opened = rows.filter((row) => row.source_group === 'opened50_current_s6').length;
  const extraGroups = new Set((index.source_groups || []).filter((group) => group.current).map((group) => group.id));
  const extra = rows.filter((row) => extraGroups.has(row.source_group)).length;
  const labeled = rows.length - automatic.length - partial - opened - extra;
  return `${labeled ? `${labeled} curated labeled deliveries · ` : ''}${opened ? `${opened} opened50 current deliveries · ` : ''}${partial ? `${partial} opened50 partial reconstructions · ` : ''}${extra ? `${extra} controlled-input views · ` : ''}${labeled || partial || opened || extra ? 'automatic: ' : ''}${source}${automatic.length} delivery segment${automatic.length === 1 ? '' : 's'}`;
}

export function nativeContextSummary(doc) {
  const context = doc.review_context;
  if (!context || !doc.frame_range) return '';
  const [lo, hi] = doc.frame_range;
  if (['local_s6_observations_only', 'automatic_observations_only'].includes(context.kind)) {
    return `Original native pictures: f${lo}–${hi}. No fitted 3D output. Play, scrub or use the arrow keys to inspect the video.`;
  }
  if (context.kind === 'local_s6_observation_scope') {
    return `Original native pictures: f${lo}–${hi}. Modeled support: f${context.score_start_frame.toFixed(1)}–${context.score_end_frame.toFixed(1)}. `
      + 'Competitive ending remains unknown; pictures outside modeled support are original video only.';
  }
  if (context.observed_live_continuation) {
    return `Original native pictures: f${lo}–${hi}. Live ball shown through f${context.display_end_frame.toFixed(1)}. `
      + `The first in-bounds bounce at f${context.score_end_frame.toFixed(1)} is the current scoring boundary, not the end of live play.`;
  }
  return `Original native pictures: f${lo}–${hi}. Competitive scope: f${context.score_start_frame.toFixed(1)}–${context.score_end_frame.toFixed(1)}. `
    + 'Available pictures before and after that scope are context; no reconstructed ball is implied outside it.';
}

// Right-handed court -> Three.js mapping. Court +z is up and a broadcast camera stands behind
// the near baseline looking along +y. Negating court x compensates for the y/z axis swap; without
// it the mapping has determinant -1 and mirrors every left/right landmark.
export function courtToWorld(x, y, z) { return [-x, z, y]; }
export function courtVectorToWorld(x, y, z) { return [-x, z, y]; }

export const TRUE_BALL_RADIUS_M = 0.033;
export const FLIGHT_BALL_RADIUS_M = 0.09;

// The enlarged in-flight marker is useful at broadcast scale, but it must not
// intersect the court at impact. Near the plane use the physical ball radius
// and keep its centre at least one radius above the floor.
export function ballPresentation(heightM) {
  const height = Number(heightM);
  if (!Number.isFinite(height)) return null;
  const onCourt = height <= TRUE_BALL_RADIUS_M;
  return {
    radiusM: onCourt ? TRUE_BALL_RADIUS_M : FLIGHT_BALL_RADIUS_M,
    centreHeightM: onCourt ? Math.max(height, TRUE_BALL_RADIUS_M) : height,
  };
}

// Split the per-frame ball state into contiguous runs (no missing frames inside a run).
// A gap (frame delta > 1) becomes a run boundary so the viewer can draw the flight as a
// solid tube per run and render the missing span honestly (dashed bridge), never smoothed.
export function splitRuns(frameNums, segmentIds = null) {
  const runs = [];
  let start = 0;
  for (let i = 1; i < frameNums.length; i++) {
    if (
      frameNums[i] - frameNums[i - 1] !== 1
      || (segmentIds && segmentIds[i] !== segmentIds[i - 1])
    ) {
      runs.push([start, i - 1]);
      start = i;
    }
  }
  if (frameNums.length) runs.push([start, frameNums.length - 1]);
  return runs;
}

// The gaps between runs, as {fromIdx,toIdx} index pairs (end of one run -> start of next).
export function gapBridges(frameNums, segmentIds = null) {
  const runs = splitRuns(frameNums, segmentIds);
  const bridges = [];
  for (let i = 1; i < runs.length; i++) {
    bridges.push({ fromIdx: runs[i - 1][1], toIdx: runs[i][0] });
  }
  return bridges;
}

// min/max of a per-frame array for a chosen metric, ignoring null/NaN.
export function metricRange(values) {
  let lo = Infinity, hi = -Infinity;
  for (const v of values) {
    if (v === null || v === undefined || Number.isNaN(v)) continue;
    if (v < lo) lo = v;
    if (v > hi) hi = v;
  }
  if (!Number.isFinite(lo)) return { lo: 0, hi: 1 };
  if (lo === hi) hi = lo + 1;
  return { lo, hi };
}

// Turbo-ish perceptual colormap: t in [0,1] -> {r,g,b} in [0,1]. Blue->cyan->green->yellow->red.
export function ramp(t) {
  t = clamp(t, 0, 1);
  const stops = [
    [0.19, 0.07, 0.23], [0.27, 0.31, 0.79], [0.10, 0.65, 0.86],
    [0.22, 0.82, 0.44], [0.86, 0.86, 0.20], [0.90, 0.35, 0.12], [0.68, 0.07, 0.09],
  ];
  const n = stops.length - 1;
  const s = t * n;
  const i = Math.min(n - 1, Math.floor(s));
  const f = s - i;
  return {
    r: lerp(stops[i][0], stops[i + 1][0], f),
    g: lerp(stops[i][1], stops[i + 1][1], f),
    b: lerp(stops[i][2], stops[i + 1][2], f),
  };
}

// Local maxima of ball height within each contiguous run (flight apexes), returned as
// {index, z, x, y}. A point is an apex when its z exceeds both neighbours (strict) by a small
// margin, so noise doesn't spawn dozens of labels.
export function findApexes(frames, minZ = 0.5, margin = 0.02) {
  const { frame, x, y, z } = frames;
  const runs = splitRuns(frame);
  const out = [];
  for (const [a, b] of runs) {
    for (let i = a + 1; i < b; i++) {
      const zc = z[i];
      if (zc === null || zc < minZ) continue;
      if (z[i - 1] === null || z[i + 1] === null) continue;
      if (zc >= z[i - 1] + margin && zc >= z[i + 1] + margin) {
        out.push({ index: i, z: zc, x: x[i], y: y[i] });
      }
    }
  }
  return out;
}

// Where the ball trajectory crosses the net plane (y == netY): linear-interpolate the height
// z at the crossing. Returns [{x, z, t, aboveNet}] — the net-clearance validation affordance.
export function netCrossings(frames, netY, netHeightAt) {
  const { frame, x, y, z, t } = frames;
  const runs = splitRuns(frame);
  const out = [];
  for (const [a, b] of runs) {
    for (let i = a; i < b; i++) {
      const y0 = y[i], y1 = y[i + 1];
      if (y0 === null || y1 === null || z[i] === null || z[i + 1] === null) continue;
      if ((y0 - netY) === 0 || (y0 - netY) * (y1 - netY) < 0) {
        const f = (y0 - netY) === 0 ? 0 : (netY - y0) / (y1 - y0);
        const zc = lerp(z[i], z[i + 1], f);
        const xc = lerp(x[i], x[i + 1], f);
        const tc = lerp(t[i], t[i + 1], f);
        const need = netHeightAt ? netHeightAt(xc) : 0.914;
        out.push({ x: xc, z: zc, t: tc, aboveNet: zc >= need, clearance: zc - need });
      }
    }
  }
  return out;
}

// Net height at court-x by linear interpolation from posts (1.07 at x=0 and x=W) to center
// (0.914 at x=W/2) — a good-enough tape model for clearance checks.
export function makeNetHeightFn(width, centerH, postH) {
  const half = width / 2;
  return function (x) {
    const d = Math.abs(x - half) / half; // 0 center .. 1 post
    return lerp(centerH, postH, clamp(d, 0, 1));
  };
}

export function singlesNetGeometry(courtWidth, singlesInset, postOffset = 0.914) {
  const xMin = singlesInset - postOffset;
  const xMax = courtWidth - singlesInset + postOffset;
  return { xMin, xMax, width: xMax - xMin };
}

export function makeSinglesNetHeightFn(
  courtWidth,
  singlesInset,
  centerH,
  postH,
  postOffset = 0.914,
) {
  const geometry = singlesNetGeometry(courtWidth, singlesInset, postOffset);
  const insideNet = makeNetHeightFn(geometry.width, centerH, postH);
  return function (x) {
    if (x < geometry.xMin - 1e-9 || x > geometry.xMax + 1e-9) return 0;
    return insideNet(x - geometry.xMin);
  };
}

// Pick the per-frame index at (or just before) a given time t, for the scrubber/ball marker.
export function frameIndexAtTime(times, tSec) {
  if (!times.length) return -1;
  if (tSec <= times[0]) return 0;
  if (tSec >= times[times.length - 1]) return times.length - 1;
  // binary search for last index with times[i] <= tSec
  let lo = 0, hi = times.length - 1;
  while (lo < hi) {
    const mid = (lo + hi + 1) >> 1;
    if (times[mid] <= tSec) lo = mid; else hi = mid - 1;
  }
  return lo;
}

// Interpolated ball position at time t (only within a contiguous run; returns null in a gap
// so the moving marker honestly disappears over missing spans).
export function ballAtTime(frames, tSec) {
  const { frame, x, y, z, t } = frames;
  if (!frame.length) return null;
  const i = frameIndexAtTime(t, tSec);
  if (i < 0) return null;
  if (i === frame.length - 1) return { x: x[i], y: y[i], z: z[i], idx: i };
  // only interpolate across a contiguous (gap-free) step
  if (frame[i + 1] - frame[i] === 1 && x[i] !== null && x[i + 1] !== null) {
    const span = t[i + 1] - t[i];
    const f = span > 0 ? clamp((tSec - t[i]) / span, 0, 1) : 0;
    return {
      x: lerp(x[i], x[i + 1], f), y: lerp(y[i], y[i + 1], f),
      z: lerp(z[i], z[i + 1], f), idx: i,
    };
  }
  // inside/at a gap edge: show the last observed sample, flagged
  return { x: x[i], y: y[i], z: z[i], idx: i, gap: true };
}

// The full time window for the scrubber: union of ball frames, contacts, bounces, players.
export function timeBounds(doc) {
  if (Number.isFinite(doc.review_context?.view_start_t)
      && Number.isFinite(doc.review_context?.view_end_t)) {
    return { lo: doc.review_context.view_start_t, hi: doc.review_context.view_end_t };
  }
  let lo = Infinity, hi = -Infinity;
  const push = (v) => { if (v !== null && v !== undefined && !Number.isNaN(v)) { if (v < lo) lo = v; if (v > hi) hi = v; } };
  for (const t of doc.frames.t) push(t);
  for (const c of doc.contacts) push(c.t);
  for (const b of doc.bounces) push(b.t);
  for (const side of ['near', 'far']) for (const t of doc.players[side].t) push(t);
  if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
  return { lo, hi };
}

// Place declared label/fitted event ticks on the same axis as the native scrubber.
// Keeping both sources at their fractional native epochs makes a one-frame offset visible.
export function timelineTicks(doc, bounds = timeBounds(doc)) {
  const span = Math.max(1e-9, bounds.hi - bounds.lo);
  return (doc?.timeline_ticks || [])
    .filter((tick) => Number.isFinite(tick.t))
    .map((tick) => ({
      ...tick,
      fraction: clamp((tick.t - bounds.lo) / span, 0, 1),
    }));
}

// Derive contact-to-contact (or contact-to-terminal) review units from a point document.
// The 3D research metric is per shot, so a point is only a container for these intervals.
export function deriveShots(doc) {
  if (!doc) return [];
  const contacts = (doc.contacts || [])
    .filter((contact) => contact && contact.fit !== false && Number.isFinite(contact.t))
    .slice()
    .sort((left, right) => left.t - right.t);
  const shots = [];
  const add = (start, end, terminal) => {
    if (!start || !Number.isFinite(start.t) || !Number.isFinite(end.t) || end.t <= start.t) return;
    const index = shots.length;
    shots.push({
      shot_index: index,
      shot_id: `shot_${String(index).padStart(3, '0')}`,
      start_t: start.t,
      end_t: end.t,
      start_frame: Number.isFinite(start.frame) ? start.frame : start.t * doc.fps,
      end_frame: Number.isFinite(end.frame) ? end.frame : end.t * doc.fps,
      start_side: start.side || 'unknown',
      end_side: end.side || (terminal ? 'terminal' : 'unknown'),
      phase: start.phase || 'rally',
      terminal,
      bounce_count: (doc.bounces || []).filter(
        (bounce) => Number.isFinite(bounce.t) && bounce.t > start.t && bounce.t < end.t,
      ).length,
    });
  };
  for (let index = 0; index + 1 < contacts.length; index++) {
    add(contacts[index], contacts[index + 1], false);
  }
  if (contacts.length) {
    const last = contacts[contacts.length - 1];
    const bounds = timeBounds(doc);
    const exposure = doc.fps ? 1 / doc.fps : 0.04;
    const hasOutgoingFlight = last.speed_out != null || contacts.length === 1;
    if (hasOutgoingFlight && bounds.hi > last.t + exposure) {
      add(last, { t: bounds.hi, frame: bounds.hi * doc.fps, side: 'terminal' }, true);
    }
  }
  return shots;
}

function sliceParallelSeries(series, startT, endT) {
  if (!series || !Array.isArray(series.t)) return series;
  const indices = [];
  for (let index = 0; index < series.t.length; index++) {
    if (series.t[index] >= startT && series.t[index] <= endT) indices.push(index);
  }
  const output = {};
  for (const [key, value] of Object.entries(series)) {
    output[key] = Array.isArray(value) && value.length === series.t.length
      ? indices.map((index) => value[index])
      : value;
  }
  return output;
}

// Return a point document clipped to one review shot. This keeps all synchronized series in
// lockstep and prevents future or previous flights from influencing the visual verdict.
export function sliceDocumentToShot(source, shot) {
  if (!source || !shot) return source;
  const startT = shot.start_t;
  const endT = shot.end_t;
  const contextSeconds = Number.isFinite(shot.context_seconds) ? shot.context_seconds : 0.5;
  const pointBounds = timeBounds(source);
  const viewStartT = Math.max(pointBounds.lo, startT - contextSeconds);
  const viewEndT = Math.min(pointBounds.hi, endT + contextSeconds);
  const within = (row) => Number.isFinite(row.t) && row.t >= startT && row.t <= endT;
  const withinContext = (row) => Number.isFinite(row.t) && row.t >= viewStartT && row.t <= viewEndT;
  const frames = sliceParallelSeries(source.frames, startT, endT);
  const contacts = (source.contacts || []).filter(within);
  const bounces = (source.bounces || []).filter(within);
  const players = {
    ...(source.players || {}),
    near: sliceParallelSeries(source.players?.near || { t: [] }, viewStartT, viewEndT),
    far: sliceParallelSeries(source.players?.far || { t: [] }, viewStartT, viewEndT),
  };
  const skeletons = {
    near: (source.skeletons?.near || []).filter(withinContext),
    far: (source.skeletons?.far || []).filter(withinContext),
  };
  return {
    ...source,
    force_follow: true,
    review_shot: { ...shot },
    review_context: {
      score_start_t: startT,
      score_end_t: endT,
      view_start_t: viewStartT,
      view_end_t: viewEndT,
    },
    frames,
    contacts,
    bounces,
    players,
    skeletons,
    frame_range: [Math.round(viewStartT * source.fps), Math.round(viewEndT * source.fps)],
    counts: {
      ...(source.counts || {}),
      frames: frames.t.length,
      contacts: contacts.length,
      contacts_fit: contacts.filter((contact) => contact.fit).length,
      bounces: bounces.length,
      bounces_in_court: bounces.filter((bounce) => bounce.in_court).length,
      skeleton_frames: {
        near: skeletons.near.length,
        far: skeletons.far.length,
      },
    },
  };
}

export function reviewPhase(doc, tSec) {
  const context = doc?.review_context;
  if (!context) return 'scored';
  // Exported exposure epochs are rounded to microseconds; equality stays inclusive.
  const epsilon = ['local_s6_competitive', 'local_s6_observation_scope'].includes(context.kind) ? 1e-6 : 0;
  if (tSec < context.score_start_t - epsilon) return 'pre';
  if (tSec > (context.display_end_t ?? context.score_end_t) + epsilon) return 'post';
  return 'scored';
}

// Only opt-in local S6 exports alter the historical point-viewer presentation.
export function trajectoryRuns(doc) {
  const f = doc.frames;
  if (doc.review_context?.kind !== 'local_s6_competitive') return splitRuns(f.frame, f.segment);
  const roles = f.frame.map((_, i) => `${f.segment?.[i]}:${reviewPhase(doc, f.t[i])}`);
  return splitRuns(f.frame, roles);
}

export function reviewCaption(doc, tSec) {
  if (['local_s6_observations_only', 'automatic_observations_only'].includes(doc.review_context?.kind)) return 'ORIGINAL VIDEO · NO FIT';
  if (doc.review_context?.kind === 'local_s6_observation_scope') {
    const context = doc.review_context;
    if (Number.isFinite(context.modeled_horizon_frame)) {
      const phase = reviewPhase(doc, tSec);
      if (phase === 'pre') return 'ORIGINAL VIDEO · BEFORE MODELED SCOPE';
      if (phase === 'post') return 'ORIGINAL VIDEO · AFTER MODELED SCOPE';
      return `MODELED THROUGH f${context.modeled_horizon_frame} · ENDING UNKNOWN`;
    }
    return 'OBSERVED SCOPE · ENDING UNKNOWN';
  }
  const phase = reviewPhase(doc, tSec);
  if (doc.review_context?.kind === 'local_s6_competitive') {
    if (phase === 'scored' && doc.review_context.observed_live_continuation
        && tSec > doc.review_context.score_end_t + 1e-6) return 'OBSERVED LIVE CONTINUATION';
    return phase === 'pre' ? 'PRE-ATTEMPT CONTEXT'
      : phase === 'post' ? 'PASSIVE CONTEXT' : 'COMPETITIVE ATTEMPT';
  }
  return phase.toUpperCase().replace('SCORED', 'SCORED SHOT');
}

// Player court position at time t for one side (step-hold to nearest earlier sample).
export function playerAtTime(side, tSec) {
  const { t, x, y } = side;
  if (!t.length) return null;
  const i = frameIndexAtTime(t, tSec);
  if (i < 0) return null;
  const decorate = (value) => ({
    ...value,
    index: i,
    frame: side.frame?.[i] ?? null,
    pose_status: side.pose_status?.[i] ?? null,
    track_id: side.track_id?.[i] ?? null,
  });
  // Interpolate only across a short supported span. A long gap is an explicit
  // missing player state, never a held/stale court position.
  if (i < t.length - 1 && t[i + 1] - t[i] <= 0.4 && x[i] !== null && x[i + 1] !== null) {
    const span = t[i + 1] - t[i];
    const f = span > 0 ? clamp((tSec - t[i]) / span, 0, 1) : 0;
    return decorate({ x: lerp(x[i], x[i + 1], f), y: lerp(y[i], y[i + 1], f) });
  }
  if (x[i] === null || Math.abs(tSec - t[i]) > 1e-6) return null;
  return decorate({ x: x[i], y: y[i] });
}

// Interpolate a metric 3D skeleton only across adjacent, nearby pose samples. Joints missing from
// either endpoint are omitted rather than held, so an uncertain wrist never freezes in space.
export function skeletonAtTime(rows, tSec, maximumGap = 0.12) {
  if (!rows || !rows.length) return null;
  const times = rows.map((row) => row.t);
  const index = frameIndexAtTime(times, tSec);
  if (index < 0) return null;
  const first = rows[index];
  if (index === rows.length - 1) {
    return Math.abs(tSec - first.t) <= 1e-6 ? first : null;
  }
  const second = rows[index + 1];
  const span = second.t - first.t;
  const nonNativeGap = Number.isFinite(first.frame) && Number.isFinite(second.frame)
    && second.frame - first.frame !== 1;
  if (span <= 0 || span > maximumGap || nonNativeGap || tSec < first.t) {
    return Math.abs(tSec - first.t) <= 1e-6 ? first : null;
  }
  const fraction = clamp((tSec - first.t) / span, 0, 1);
  const joints = {};
  for (const [name, firstJoint] of Object.entries(first.joints || {})) {
    const secondJoint = (second.joints || {})[name];
    if (!secondJoint) continue;
    joints[name] = [
      lerp(firstJoint[0], secondJoint[0], fraction),
      lerp(firstJoint[1], secondJoint[1], fraction),
      lerp(firstJoint[2], secondJoint[2], fraction),
      Math.min(firstJoint[3], secondJoint[3]),
    ];
  }
  return Object.keys(joints).length ? { t: tSec, joints } : null;
}

// First time (s) at which the ball is actually observed. The scrubber spans the whole point
// (players + contacts + bounces can start before the ball track), so pressing PLAY at
// timeBounds.lo can sit in a leading span with NO moving ball — which reads as "play does
// nothing". Returns the earliest ball-frame time, or null when there are no ball samples.
export function firstBallTime(doc) {
  const t = doc && doc.frames && doc.frames.t;
  const x = doc && doc.frames && doc.frames.x;
  if (!t || !t.length) return null;
  for (let i = 0; i < t.length; i++) {
    if (x[i] !== null && x[i] !== undefined && t[i] !== null && t[i] !== undefined) return t[i];
  }
  return null;
}

// Map a playhead time (s) to a broadcast frame file number. Frame numbers in the artifacts
// are LOCAL clip indices equal to the f_%04d.jpg file numbers, and t == frame/fps, so the
// map is round(t*fps), clamped to >= 1. Returns null when fps is missing.
export function frameNumberAtTime(tSec, fps) {
  if (!fps || !Number.isFinite(fps) || fps <= 0) return null;
  const n = Math.round(tSec * fps);
  return n < 1 ? 1 : n;
}

// Exact native epochs are authoritative when supplied. Fractional frame time is
// interpolation, never a synthesized video picture. Legacy clocks stay unchanged.
export function nativeFrameAtTime(doc, t) {
  const clock = doc && doc.native_timebase;
  if (clock && clock.mapping === 'piecewise_linear_original_native_pts') {
    const times = clock.native_times_seconds;
    const frames = clock.native_frames;
    let lo = 0, hi = times.length - 1;
    if (t <= times[lo]) return frames[lo];
    if (t >= times[hi]) return frames[hi];
    while (hi - lo > 1) {
      const mid = Math.floor((lo + hi) / 2);
      if (times[mid] <= t) lo = mid; else hi = mid;
    }
    return t - times[lo] < times[hi] - t ? frames[lo] : frames[hi];
  }
  if (clock && Number.isFinite(clock.origin_frame)
      && Number.isFinite(clock.origin_t) && Number.isFinite(doc.fps)) {
    return Math.max(1, Math.round(clock.origin_frame + (t - clock.origin_t) * doc.fps));
  }
  return frameNumberAtTime(t, doc && doc.fps);
}

export function nativeTimeForFrame(doc, frame) {
  const clock = doc && doc.native_timebase;
  if (clock && clock.mapping === 'piecewise_linear_original_native_pts') {
    const times = clock.native_times_seconds;
    const frames = clock.native_frames;
    if (frame <= frames[0]) return times[0];
    if (frame >= frames[frames.length - 1]) return times[times.length - 1];
    let lo = 0, hi = frames.length - 1;
    while (hi - lo > 1) {
      const mid = Math.floor((lo + hi) / 2);
      if (frames[mid] <= frame) lo = mid; else hi = mid;
    }
    return lerp(times[lo], times[hi], (frame - frames[lo]) / (frames[hi] - frames[lo]));
  }
  if (clock && Number.isFinite(clock.origin_frame)
      && Number.isFinite(clock.origin_t) && Number.isFinite(doc.fps)) {
    return clock.origin_t + (frame - clock.origin_frame) / doc.fps;
  }
  return frame / doc.fps;
}

// Zero-padded broadcast frame filename, e.g. (6, "f_%04d.jpg") -> "f_0006.jpg".
export function frameFileName(n, pattern) {
  const m = (pattern || "f_%04d.jpg").match(/%0(\d+)d/);
  const width = m ? parseInt(m[1], 10) : 4;
  const num = String(n).padStart(width, "0");
  return (pattern || "f_%04d.jpg").replace(/%0\d+d/, num);
}

// Root-relative URL of the broadcast frame at frame number n for a point doc, or null when
// this clip has no served frames. Path: /<match>/<frames_dir>/<clip>/f_%04d.jpg.
export function frameUrl(doc, n) {
  if (!doc || !doc.frames_available || !doc.frames_dir || n == null) return null;
  const base = doc.frames_base || `/${doc.match}`;
  return `${base}/${doc.frames_dir}/${doc.clip}/${frameFileName(n, doc.frames_pattern)}`;
}

// A cheap quality score for ranking DIAGNOSTIC points best-first in the picker (lower is
// better): junction agreement in metres dominates (how well the fitted flights join at
// contacts), then fit RMS, with a penalty for a poor in-court bounce rate. Points missing
// junction data (too few fitted shots) sort last. Pure function over an index entry.
export function diagnosticScore(entry) {
  const j = (entry.junction_med == null || Number.isNaN(entry.junction_med)) ? 9.9 : entry.junction_med;
  const rms = (entry.rms_med == null || Number.isNaN(entry.rms_med)) ? 99 : entry.rms_med;
  const bounceMiss = 1 - ((entry.bounce_in_court_rate == null) ? 0 : entry.bounce_in_court_rate);
  return j + 0.02 * rms + 0.5 * bounceMiss;
}

// Order index entries for the picker: accepted (verified) first, diagnostics after, each
// group ranked best-first. Returns a NEW sorted array; does not mutate the input.
export function orderPoints(points) {
  const withIdx = points.map((p, i) => ({ p, i }));
  withIdx.sort((a, b) => {
    const aa = a.p.accepted === true, ba = b.p.accepted === true;
    if (aa !== ba) return aa ? -1 : 1;          // accepted first
    const sa = diagnosticScore(a.p), sb = diagnosticScore(b.p);
    if (aa) {                                    // both accepted: best junction first
      if (sa !== sb) return sa - sb;
    } else {                                     // both diagnostic: best junction first
      if (sa !== sb) return sa - sb;
    }
    const pa = a.p.point == null ? Infinity : a.p.point;
    const pb = b.p.point == null ? Infinity : b.p.point;
    return pa - pb;
  });
  return withIdx.map((w) => w.p);
}

// The index entry to open by default: the first ACCEPTED point (never a rejected fit), else
// the best-ranked diagnostic. `ordered` must already be orderPoints() output.
export function defaultPoint(ordered) {
  if (!ordered || !ordered.length) return null;
  const acc = ordered.find((p) => p.accepted === true);
  return acc || ordered[0];
}

// Short match tag for labels: rg2025f -> "RG", uso2025f -> "USO".
export function matchTag(match) {
  if (!match) return "";
  if (match.startsWith("rg")) return "RG";
  if (match.startsWith("uso")) return "USO";
  return match.replace(/2025f?$/, "").toUpperCase();
}

// Pull the points score out of a Stage-1 wing score_state ("pts 30-15 / games 0-0" -> "30-15").
export function scoreShort(scoreState) {
  if (!scoreState) return null;
  const m = String(scoreState).match(/pts\s+([0-9AD]+-[0-9AD]+)/i);
  return m ? m[1] : null;
}

// Human-readable picker label for a point, joining the 3D index entry with optional Stage-1
// wing score context {server, score_state}. e.g.
//   "RG · point 26 · Sinner serving 30-15 · 8 shots · 14s · verified"
// Falls back gracefully when score context is absent.
export function pointLabel(entry, ctx) {
  const parts = [`${matchTag(entry.match)} · point ${entry.point ?? entry.clip}`];
  if (ctx && ctx.server) {
    const sc = scoreShort(ctx.score_state);
    parts.push(sc ? `${ctx.server} serving ${sc}` : `${ctx.server} serving`);
  }
  const shots = entry.n_contacts_fit != null ? entry.n_contacts_fit : entry.n_contacts;
  if (shots != null) parts.push(`${shots} shots`);
  if (entry.duration_s != null) parts.push(`${Math.round(entry.duration_s)}s`);
  parts.push(entry.accepted === true ? "verified" : "diagnostic (fit rejected)");
  return parts.join(" · ");
}

// Playback-trail opacity for one element (a tube vertex, marker, or label) given its own
// time `sampleT` and the current playhead time `headT`. This is the pure playhead-time ->
// alpha window mapping the trail mode is built on:
//   * anything AHEAD of the playhead (not yet reached) is invisible (alpha 0) — the flight
//     draws only up to the ball;
//   * a recent window of `recent` seconds behind the playhead is full opacity — the live ball
//     and its immediate wake;
//   * older than that fades (over a short `fade` band, so it doesn't pop) down to a low
//     `ghost` floor so the point's accumulated shape stays visible without burying the ball.
// `full` is the opacity of the recent window (1 by default). Pure + framework-free so the
// viewer can drive per-vertex alpha and per-marker opacity from it, and so it is unit-tested.
export function trailAlpha(sampleT, headT, opts) {
  const o = opts || {};
  const recent = o.recent != null ? o.recent : 1.5;
  const ghost = o.ghost != null ? o.ghost : 0.18;
  const fade = o.fade != null ? o.fade : 0.6;
  const full = o.full != null ? o.full : 1;
  if (sampleT == null || Number.isNaN(sampleT)) return ghost;
  if (sampleT > headT) return 0;                 // ahead of the playhead: not yet revealed
  const age = headT - sampleT;
  if (age <= recent) return full;                // inside the recent (full-opacity) window
  if (age >= recent + fade) return ghost;        // fully aged to the ghost floor
  const f = (age - recent) / fade;               // linear fade across the transition band
  return full + (ghost - full) * f;
}

// Is this a "long" point — the kind where the full-point spaghetti becomes unreadable and
// follow (progressive-reveal) mode should be the default? True when the rally is either many
// shots or many seconds. Thresholds default to >8 shots OR >8 s, overridable for tests.
export function isLongPoint(doc, opts) {
  if (!doc) return false;
  const o = opts || {};
  const shotThresh = o.shots != null ? o.shots : 8;
  const secThresh = o.seconds != null ? o.seconds : 8;
  const counts = doc.counts || {};
  const shots = counts.contacts_fit != null ? counts.contacts_fit
    : (counts.contacts != null ? counts.contacts : 0);
  const tb = timeBounds(doc);
  const dur = tb.hi - tb.lo;
  return shots > shotThresh || dur > secThresh;
}

// Resolve whether follow (progressive-reveal) mode should be ON for a point. An explicit
// per-session user preference ('on' | 'off') always wins; with no preference (null/undefined)
// it auto-decides by isLongPoint — so long rallies open in follow mode, short ones in the
// classic full view, until the viewer flips the toggle.
export function shouldFollow(doc, pref, opts) {
  if (doc && doc.force_follow) return true;
  if (pref === 'on') return true;
  if (pref === 'off') return false;
  return isLongPoint(doc, opts);
}

// Classify a contact for styling: serve / rally / dead(unsupported) — the visual tiers.
export function contactClass(c) {
  if (!c.fit) return 'dead';
  if (c.phase === 'serve') return 'serve';
  return 'rally';
}

const GENERIC_FLIGHT_CAUSES = new Set([
  '', 'miss', 'held', 'execution failed', 'unlisted miss', 'not_launched', 'none',
]);

// Fresh panels 0 and A, the development sets, opened panel C, or the named error themes.
export function auditSetForPanel(panel) {
  if (panel === 'panel_0' || panel === 'panel_A') return 'fresh';
  if (panel === 'matched_fresh12_v1' || panel === 'matched_fresh36_v1') return 'development';
  if (panel === 'panel_C' || panel === 'panel_c') return 'panel_c';
  return 'other';
}

export function entryAuditSet(entry) {
  if (!entry) return 'other';
  if (entry.audit_set) return entry.audit_set;
  if (entry.source_group && entry.source_group !== 'flight_review') return 'points';
  return auditSetForPanel(entry.panel);
}

export function isErrorThemeFlight(flight) {
  if (!flight) return false;
  const cause = String(flight.cause || '').trim();
  if (!cause || GENERIC_FLIGHT_CAUSES.has(cause.toLowerCase())) return false;
  return true;
}

export function matchIdentity(entry) {
  const source = entry.source_id || String(entry.point || '').split('_')[0] || 'match';
  const tournament = entry.tournament || '';
  const players = entry.players || '';
  const title = [tournament, players].filter(Boolean).join(' · ') || source;
  return { source, tournament, players, title };
}

export function overlayUserUnits(viewWidth, viewHeight, clientWidth, clientHeight, px) {
  const width = Number(viewWidth) || 1920;
  const height = Number(viewHeight) || 1080;
  // Before layout, assume the small overlay player rather than full native pixels.
  const shownW = Number(clientWidth) > 0 ? Number(clientWidth) : Math.min(width, 520);
  const shownH = Number(clientHeight) > 0 ? Number(clientHeight) : shownW * height / width;
  const scale = Math.min(shownW / width, shownH / height) || 1;
  return Number(px) / scale;
}

export function flightEntriesForSet(entries, auditSet) {
  const rows = (entries || []).filter((entry) => entry && entry.source_group === 'flight_review');
  if (!auditSet || auditSet === 'all') return rows;
  if (auditSet === 'error_themes') {
    return rows.filter((entry) => entry.error_theme || (entry.flights || []).some(isErrorThemeFlight));
  }
  if (auditSet === 'points') return [];
  return rows.filter((entry) => entryAuditSet(entry) === auditSet);
}

export function searchFlightEntries(entries, query) {
  const needle = String(query || '').trim().toLowerCase();
  if (!needle) return entries || [];
  return (entries || []).filter((entry) => {
    const blob = [
      entry.point, entry.panel, entry.arm, entry.tournament, entry.players,
      entry.broadcast, entry.label, entry.source_id,
      ...(entry.flights || []).flatMap((flight) => [flight.id, flight.outcome, flight.cause, flight.event_type]),
    ].join(' ').toLowerCase();
    return blob.includes(needle);
  });
}

export function matchChoices(entries) {
  const by = new Map();
  for (const entry of entries || []) {
    const id = matchIdentity(entry);
    if (!by.has(id.source)) by.set(id.source, { ...id, attempts: new Set() });
    by.get(id.source).attempts.add(entry.point);
  }
  return [...by.values()].map((row) => ({
    source: row.source,
    title: row.title,
    tournament: row.tournament,
    players: row.players,
    attempts: row.attempts.size,
  })).sort((a, b) => a.title.localeCompare(b.title) || a.source.localeCompare(b.source));
}

export function attemptChoices(entries, source) {
  const rows = (entries || []).filter((entry) => matchIdentity(entry).source === source);
  const by = new Map();
  for (const entry of rows) {
    if (!by.has(entry.point)) by.set(entry.point, { point: entry.point, arms: [] });
    by.get(entry.point).arms.push(entry);
  }
  return [...by.values()].sort((a, b) => String(a.point).localeCompare(String(b.point)));
}

export function armChips(armEntries) {
  return (armEntries || []).map((entry) => {
    const counts = { matched: 0, missed: 0, extra: 0 };
    const causes = [];
    for (const flight of entry.flights || []) {
      const outcome = flight.outcome || 'missed';
      if (Object.prototype.hasOwnProperty.call(counts, outcome)) counts[outcome] += 1;
      if (flight.cause && !causes.includes(flight.cause)) causes.push(flight.cause);
    }
    return { arm: entry.arm, file: entry.file, counts, causes };
  }).sort((a, b) => String(a.arm).localeCompare(String(b.arm)));
}

export function orderedFlights(entry) {
  return ((entry && entry.flights) || []).slice().sort((a, b) =>
    Number(a.origin_frame || 0) - Number(b.origin_frame || 0)
    || String(a.id).localeCompare(String(b.id)));
}

export function reviewSequence(entries) {
  const ordered = (entries || []).slice().sort((a, b) =>
    String(a.panel || '').localeCompare(String(b.panel || ''))
    || String(a.point || '').localeCompare(String(b.point || ''))
    || String(a.arm || '').localeCompare(String(b.arm || '')));
  const sequence = [];
  for (const entry of ordered) {
    const flights = orderedFlights(entry);
    if (!flights.length) sequence.push({ entry, flight: null });
    else for (const flight of flights) sequence.push({ entry, flight });
  }
  return sequence;
}

export function stepReview(entries, file, flightId, direction) {
  const sequence = reviewSequence(entries);
  if (!sequence.length) return null;
  let index = sequence.findIndex((row) =>
    row.entry.file === file && ((row.flight && row.flight.id) || null) === (flightId || null));
  if (index < 0) index = sequence.findIndex((row) => row.entry.file === file);
  if (index < 0) index = 0;
  const step = direction < 0 ? -1 : 1;
  const next = Math.min(sequence.length - 1, Math.max(0, index + step));
  return sequence[next];
}

// dual-mode export: usable as an ES module in the browser AND grabbable by the Node VM test.
const _api = {
  clamp, lerp, supportedSpans, componentView, observedBallLegend, reconstructionStatus, resolvePointRequest, currentCohortLabel, currentAttemptGroup, currentAttempts, currentHeldAttempts, currentSelectionLabel, nativeContextSummary, courtToWorld, courtVectorToWorld,
  TRUE_BALL_RADIUS_M, FLIGHT_BALL_RADIUS_M, ballPresentation,
  splitRuns, gapBridges, metricRange, ramp, findApexes, netCrossings,
  singlesNetGeometry, makeSinglesNetHeightFn,
  makeNetHeightFn, frameIndexAtTime, ballAtTime, timeBounds, timelineTicks, deriveShots,
  sliceDocumentToShot, reviewPhase, trajectoryRuns, reviewCaption, playerAtTime, skeletonAtTime,
  contactClass, trailAlpha,
  isLongPoint, shouldFollow,
  firstBallTime, frameNumberAtTime, nativeFrameAtTime, nativeTimeForFrame, frameFileName, frameUrl, diagnosticScore, orderPoints,
  defaultPoint, matchTag, scoreShort, pointLabel, projectCamera, flightReviewChoices,
  auditSetForPanel, entryAuditSet, isErrorThemeFlight, matchIdentity, overlayUserUnits,
  flightEntriesForSet, searchFlightEntries, matchChoices, attemptChoices, armChips,
  orderedFlights, reviewSequence, stepReview,
};
if (typeof globalThis !== 'undefined') globalThis.ViewerCore = _api;
export default _api;
