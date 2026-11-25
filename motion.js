import { Euler, Quaternion, Vector3 } from 'three';

const radius = .05;
const halfWidth = .009;
const halfTrack = .119;
const hipHeight = .248;
const rad = Math.PI / 180;
const xAxis = new Vector3(1, 0, 0);
const zAxis = new Vector3(0, 0, 1);
const clamp = (value, minimum, maximum) => Math.max(minimum, Math.min(maximum, value));
const smooth = (p) => p ** 3 * (10 - 15 * p + 6 * p ** 2);

const ENV_MOTION_PARAMS = {
  'flat-ground': {
    speedScale: 1.0,
    bankScale: 1.0,
    extraCrouch: 0,
    slipRatio: 1.0,
    sinkOffset: 0,
    slopeAdapt: false,
    stepAdapt: false,
  },
  'hilly-terrain': {
    speedScale: 0.88,
    bankScale: 0.85,
    extraCrouch: 0.004,
    slipRatio: 1.04,
    sinkOffset: 0,
    slopeAdapt: true,
    stepAdapt: false,
  },
  'obstacle-park': {
    speedScale: 0.92,
    bankScale: 0.95,
    extraCrouch: 0.002,
    slipRatio: 1.0,
    sinkOffset: 0,
    slopeAdapt: false,
    stepAdapt: true,
  },
  'sandy-beach': {
    speedScale: 0.68,
    bankScale: 0.55,
    extraCrouch: 0.013,
    slipRatio: 1.18,
    sinkOffset: -0.007,
    slopeAdapt: true,
    stepAdapt: false,
  },
  'urban-sidewalk': {
    speedScale: 0.90,
    bankScale: 0.85,
    extraCrouch: 0.002,
    slipRatio: 1.0,
    sinkOffset: 0,
    slopeAdapt: false,
    stepAdapt: true,
  },
};

export function makeArc(start, x, z, config = {}) {
  const dx = x - start.x, dz = z - start.z;
  const chord = Math.hypot(dx, dz);
  if (chord < .004) return null;
  const speedScale = config.speedScale ?? 1.0;
  const bankScale = config.bankScale ?? 1.0;
  const extraCrouch = config.extraCrouch ?? 0;
  const slipRatio = config.slipRatio ?? 1.0;

  const forward = dx * Math.sin(start.yaw) + dz * Math.cos(start.yaw);
  const side = dx * Math.cos(start.yaw) - dz * Math.sin(start.yaw);
  const direction = forward < 0 ? -1 : 1;
  const turn = 2 * Math.atan2(direction * side, Math.abs(forward));
  const length = Math.abs(turn) < 1e-8 ? chord : chord * turn / (2 * Math.sin(turn / 2));
  const curvature = turn / length;
  const baseDuration = Math.max(1.4, length / .28, Math.abs(turn) / .86);
  const duration = baseDuration / speedScale;

  function at(fraction) {
    const yawChange = turn * fraction, halfAngle = yawChange / 2;
    const sinc = Math.abs(halfAngle) < 1e-8 ? 1 : Math.sin(halfAngle) / halfAngle;
    const distance = direction * length * fraction;
    return {
      x: start.x + distance * sinc * Math.sin(start.yaw + halfAngle),
      z: start.z + distance * sinc * Math.cos(start.yaw + halfAngle),
      yaw: start.yaw + yawChange,
      left: start.left + ((distance * slipRatio) + halfTrack * yawChange) / radius,
      right: start.right + ((distance * slipRatio) - halfTrack * yawChange) / radius,
    };
  }

  return {
    at, duration, direction, length,
    sample(time) {
      const p = clamp(time / duration, 0, 1);
      const travel = smooth(p);
      const speed = (length / duration) * 30 * p ** 2 * (1 - p) ** 2;
      const acceleration = (length / duration ** 2) * 60 * p * (1 - p) * (1 - 2 * p);
      const engagement = Math.sin(Math.PI * p) ** 2;
      const bank = -clamp(direction * curvature * speed ** 2 * .48 * bankScale, -.115, .115);
      return {
        ...at(travel),
        bodyPitch: clamp(direction * acceleration * .22, -.13, .13),
        bank,
        crouch: .011 * engagement + extraCrouch * engagement,
        stance: engagement,
      };
    },
  };
}

export function makeEnvironmentMotion(start, waypoints, envManager, envId = 'flat-ground') {
  const cfg = ENV_MOTION_PARAMS[envId] || ENV_MOTION_PARAMS['flat-ground'];
  const list = Array.isArray(waypoints) && waypoints.length ? waypoints : [];
  if (list.length === 0) return null;

  // Single arc on flat ground preserves original motion 1:1
  if (list.length === 1 && envId === 'flat-ground') {
    return makeArc(start, list[0].x, list[0].z);
  }

  // Build sequential arc segments through waypoints
  const segments = [];
  let cursor = { ...start };
  for (const wp of list) {
    const seg = makeArc(cursor, wp.x, wp.z, cfg);
    if (!seg) continue;
    segments.push(seg);
    cursor = seg.at(1);
  }
  if (segments.length === 0) return null;

  const totalDuration = segments.reduce((acc, s) => acc + s.duration, 0);
  const totalLength = segments.reduce((acc, s) => acc + s.length, 0);
  const segStarts = [];
  let tAccum = 0;
  for (const s of segments) {
    segStarts.push(tAccum);
    tAccum += s.duration;
  }

  function at(fraction) {
    const f = clamp(fraction, 0, 1);
    if (segments.length === 1) return segments[0].at(f);
    const targetLen = f * totalLength;
    let lenAcc = 0;
    for (let i = 0; i < segments.length; i++) {
      const s = segments[i];
      if (lenAcc + s.length >= targetLen || i === segments.length - 1) {
        const localF = s.length > 1e-6 ? clamp((targetLen - lenAcc) / s.length, 0, 1) : 1;
        return s.at(localF);
      }
      lenAcc += s.length;
    }
    return segments[segments.length - 1].at(1);
  }

  function sampleBase(time) {
    const t = clamp(time, 0, totalDuration);
    if (segments.length === 1) {
      return { ...segments[0].sample(t), direction: segments[0].direction };
    }
    for (let i = 0; i < segments.length; i++) {
      const startT = segStarts[i];
      const s = segments[i];
      if (t <= startT + s.duration || i === segments.length - 1) {
        const localT = clamp(t - startT, 0, s.duration);
        const pose = s.sample(localT);
        // Keep stance engaged across internal waypoint transitions
        const overallP = clamp(t / totalDuration, 0, 1);
        const overallStance = Math.sin(Math.PI * overallP) ** 2;
        return {
          ...pose,
          stance: Math.max(pose.stance, overallStance * 0.85),
          direction: s.direction,
        };
      }
    }
    return { ...segments[segments.length - 1].sample(segments[segments.length - 1].duration), direction: 1 };
  }

  // If no terrain manager or flat ground, return directly without terrain profile overhead
  if (!envManager || envId === 'flat-ground') {
    return {
      at,
      duration: totalDuration,
      direction: segments[0].direction,
      waypoints: list,
      sample: sampleBase,
    };
  }

  // Pre-sample terrain height, wheel contacts, normal slope, and step transitions
  // along the trajectory so both live playback and timeline scrubbing are smooth and deterministic.
  const N = 65;
  const includeSteppable = envId === 'urban-sidewalk' || envId === 'obstacle-park';
  const rawSamples = [];
  for (let i = 0; i < N; i++) {
    const t = (i / (N - 1)) * totalDuration;
    const base = sampleBase(t);
    const cosY = Math.cos(base.yaw);
    const sinY = Math.sin(base.yaw);
    const lx = base.x - halfTrack * cosY;
    const lz = base.z + halfTrack * sinY;
    const rx = base.x + halfTrack * cosY;
    const rz = base.z - halfTrack * sinY;

    const cTerrain = envManager.sampleTerrain(base.x, base.z, { includeSteppable });
    const lTerrain = envManager.sampleTerrain(lx, lz, { includeSteppable });
    const rTerrain = envManager.sampleTerrain(rx, rz, { includeSteppable });
    rawSamples.push({ base, cTerrain, lTerrain, rTerrain });
  }

  // Smooth elevation & normal across neighboring samples to eliminate sudden orientation jumps
  const profile = rawSamples.map((_, idx) => {
    const weights = cfg.slopeAdapt ? [1, 2, 3, 2, 1] : [1, 2, 3, 2, 1];
    const halfW = 2;
    let wSum = 0;
    let avgH = 0;
    let avgLH = 0;
    let avgRH = 0;
    let nx = 0;
    let ny = 0;
    let nz = 0;
    let stepDelta = 0;

    for (let k = -halfW; k <= halfW; k++) {
      const sIdx = clamp(idx + k, 0, N - 1);
      const w = weights[k + halfW];
      const item = rawSamples[sIdx];
      avgH += item.cTerrain.height * w;
      avgLH += item.lTerrain.height * w;
      avgRH += item.rTerrain.height * w;
      nx += item.cTerrain.normal.x * w;
      ny += item.cTerrain.normal.y * w;
      nz += item.cTerrain.normal.z * w;
      wSum += w;
    }

    // Detect nearby step edge for Urban Sidewalk / Obstacle Park step adaptation
    if (cfg.stepAdapt) {
      for (let k = -3; k <= 3; k++) {
        const aIdx = clamp(idx + k - 1, 0, N - 1);
        const bIdx = clamp(idx + k + 1, 0, N - 1);
        const diff = Math.abs(rawSamples[bIdx].cTerrain.height - rawSamples[aIdx].cTerrain.height);
        if (diff > stepDelta) stepDelta = diff;
      }
    }

    const nLen = Math.hypot(nx, ny, nz) || 1;
    const cur = rawSamples[idx];
    const wheelAvg = (avgLH + avgRH) / (2 * wSum);
    const groundY = cfg.stepAdapt ? avgH / wSum : wheelAvg;

    return {
      groundY: groundY + cfg.sinkOffset,
      leftY: avgLH / wSum,
      rightY: avgRH / wSum,
      nx: nx / nLen,
      ny: ny / nLen,
      nz: nz / nLen,
      stepActivity: cfg.stepAdapt ? clamp((stepDelta - 0.015) / 0.075, 0, 1) : 0,
      slopeDeg: cur.cTerrain.slopeDeg,
      walkable: cur.cTerrain.walkable,
    };
  });

  function interpolateProfile(t) {
    const u = clamp(t / totalDuration, 0, 1) * (N - 1);
    const i0 = Math.floor(u);
    const i1 = Math.min(N - 1, i0 + 1);
    const a = u - i0;
    const p0 = profile[i0];
    const p1 = profile[i1];
    const lerp = (v0, v1) => v0 + (v1 - v0) * a;
    return {
      groundY: lerp(p0.groundY, p1.groundY),
      leftY: lerp(p0.leftY, p1.leftY),
      rightY: lerp(p0.rightY, p1.rightY),
      nx: lerp(p0.nx, p1.nx),
      ny: lerp(p0.ny, p1.ny),
      nz: lerp(p0.nz, p1.nz),
      stepActivity: lerp(p0.stepActivity, p1.stepActivity),
    };
  }

  return {
    at,
    duration: totalDuration,
    direction: segments[0].direction,
    waypoints: list,
    sample(time) {
      const base = sampleBase(time);
      const prof = interpolateProfile(time);
      const sinY = Math.sin(base.yaw);
      const cosY = Math.cos(base.yaw);

      let terrainPitch = 0;
      let terrainRoll = 0;
      let clearanceLeft = 0;
      let clearanceRight = 0;
      let stepCrouch = 0;

      if (cfg.slopeAdapt) {
        const fwdDot = clamp(prof.nx * sinY + prof.nz * cosY, -0.45, 0.45);
        const rightDot = clamp(prof.nx * cosY - prof.nz * sinY, -0.45, 0.45);
        terrainPitch = clamp(Math.asin(fwdDot) * 0.65, -0.24, 0.24);
        terrainRoll = clamp(-Math.asin(rightDot) * 0.42, -0.16, 0.16);
        const wheelMid = (prof.leftY + prof.rightY) * 0.5;
        clearanceLeft = clamp(prof.leftY - wheelMid, -0.055, 0.055);
        clearanceRight = clamp(prof.rightY - wheelMid, -0.055, 0.055);
      }

      if (cfg.stepAdapt && prof.stepActivity > 0.01) {
        const lift = 0.042 * prof.stepActivity;
        stepCrouch = 0.014 * prof.stepActivity;
        clearanceLeft += lift;
        clearanceRight += lift;
      }

      return {
        ...base,
        groundY: prof.groundY,
        bodyPitch: clamp(base.bodyPitch + terrainPitch, -0.28, 0.28),
        bank: clamp(base.bank + terrainRoll, -0.20, 0.20),
        crouch: base.crouch + stepCrouch,
        clearanceLeft,
        clearanceRight,
        stepActivity: prof.stepActivity,
        slipRatio: cfg.slipRatio,
      };
    },
  };
}

export function makeRig(model, ground) {
  const legs = ['L', 'R'].map((side, index) => {
    const roll = model.getObjectByName(side + '_Hip_roll');
    const hip = model.getObjectByName(side + '_Hip_pitch');
    const knee = model.getObjectByName(side + '_Knee_pitch');
    const wheel = model.getObjectByName(side + '_Wheel_spin');
    return {roll, hip, knee, wheel, side: index === 0 ? -1 : 1};
  });
  const neutralHeight = hipHeight - ground;
  const body = new Quaternion(), orientation = new Euler();
  const wheelCenter = new Vector3(), axle = new Vector3();

  function solveLeg(leg, swing, bodyY, targetZ, clearance) {
    const {roll, hip, knee, wheel} = leg;
    const rollAngle = roll.rotation.z;
    axle.copy(xAxis).applyAxisAngle(zAxis, rollAngle).applyQuaternion(body);
    const support = radius * Math.sqrt(1 - axle.y ** 2) + halfWidth * Math.abs(axle.y);
    const base = roll.position;
    const upper = knee.position, lower = wheel.position;
    function evaluate(hipAngle, kneeAngle) {
      wheelCenter.copy(lower).applyAxisAngle(xAxis, kneeAngle).add(upper)
        .applyAxisAngle(xAxis, hipAngle).add(hip.position)
        .applyAxisAngle(zAxis, rollAngle).add(base);
      wheelCenter.y -= hipHeight;
      wheelCenter.applyQuaternion(body);
      return [wheelCenter.y + bodyY - support - clearance, wheelCenter.z - targetZ];
    }

    const step = .0001;
    let a = swing, b = -.34;
    for (let i = 0; i < 11; i++) {
      const [y, z] = evaluate(a, b);
      if (Math.hypot(y, z) < .00006) break;
      const [ya, za] = evaluate(a + step, b);
      const [yb, zb] = evaluate(a, b + step);
      const A = (ya - y) / step, B = (yb - y) / step;
      const C = (za - z) / step, D = (zb - z) / step;
      const det = A * D - B * C;
      if (Math.abs(det) < 1e-8) break;
      a = clamp(a + clamp((-D * y + B * z) / det, -.24, .24), -47 * rad, 47 * rad);
      b = clamp(b + clamp((C * y - A * z) / det, -.24, .24), -51 * rad, .10);
    }
    hip.rotation.x = a;
    knee.rotation.x = b;
    return evaluate(a, b)[0];
  }

  return (posture, p) => {
    const {
      crouch = 0,
      bank = 0,
      bodyPitch = 0,
      stance = 0,
      clearanceLeft = 0,
      clearanceRight = 0,
    } = p;
    body.setFromEuler(orientation.set(bodyPitch, 0, bank, 'ZXY'));
    posture.quaternion.copy(body);
    const shift = Math.min(.018, Math.abs(bank) * .052);
    const height = neutralHeight - crouch - shift;
    posture.position.set(-.15 * bank, height, 0);

    const needsTerrainPose =
      Math.abs(clearanceLeft) > .001 ||
      Math.abs(clearanceRight) > .001 ||
      Math.abs(bodyPitch) > .002 ||
      Math.abs(bank) > .002 ||
      Math.abs(crouch) > .001;

    let error = 0;
    for (const leg of legs) {
      const {side, roll, hip, knee, wheel} = leg;
      const rollEngage = Math.max(stance, needsTerrainPose ? .25 : 0);
      roll.rotation.z = side * .075 * rollEngage - .64 * bank;
      if (stance < .000001 && !needsTerrainPose) {
        hip.rotation.x = knee.rotation.x = 0;
      } else {
        // The wheel centers stay almost aligned while the torso banks into the curve.
        const targetZ = .005 + side * (-bank) * .068;
        const clearance = side < 0 ? clearanceLeft : clearanceRight;
        const supportError = solveLeg(leg, -bodyPitch + .22, height, targetZ, clearance);
        error = Math.max(error, Math.abs(supportError));
      }
      wheel.rotation.x = side < 0 ? p.left : p.right;
    }
    return {kneeLeft: legs[0].knee.rotation.x, kneeRight: legs[1].knee.rotation.x, bank, error};
  };
}
