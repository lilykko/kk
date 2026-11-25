import * as T from 'three';

export const DEFAULT_CRITIC_WEIGHTS = {
  groundContact: 0.24,
  collisionAvoidance: 0.26,
  jointValidity: 0.12,
  motionContinuity: 0.12,
  balanceStability: 0.14,
  terrainAdaptation: 0.12,
};

export const APPROX_JOINT_LIMITS = {
  hipPitchMinDeg: -47,
  hipPitchMaxDeg: 47,
  kneePitchMinDeg: -51,
  kneePitchMaxDeg: 6,
  hipRollMaxDeg: 22,
  maxJointRateDegPerSec: 160,
};

const RAD_TO_DEG = 180 / Math.PI;
const WHEEL_RADIUS = 0.05;
const WHEEL_HALF_WIDTH = 0.009;

const _vA = new T.Vector3();
const _vB = new T.Vector3();
const _vTorso = new T.Vector3();
const _vKneeL = new T.Vector3();
const _vKneeR = new T.Vector3();
const _vWheelL = new T.Vector3();
const _vWheelR = new T.Vector3();
const _vUp = new T.Vector3();
const _qPosture = new T.Quaternion();
const _qWheelL = new T.Quaternion();
const _qWheelR = new T.Quaternion();
const _axleL = new T.Vector3();
const _axleR = new T.Vector3();
const _xAxis = new T.Vector3(1, 0, 0);
const _yAxis = new T.Vector3(0, 1, 0);

function clamp(v, min, max) {
  return Math.max(min, Math.min(max, v));
}

function sphereIntersectsBox3(center, radius, box) {
  const cx = clamp(center.x, box.min.x, box.max.x);
  const cy = clamp(center.y, box.min.y, box.max.y);
  const cz = clamp(center.z, box.min.z, box.max.z);
  const dx = center.x - cx;
  const dy = center.y - cy;
  const dz = center.z - cz;
  const distSq = dx * dx + dy * dy + dz * dz;
  if (distSq >= radius * radius) return 0;
  return radius - Math.sqrt(distSq);
}

export function statusFromScore(score) {
  if (score >= 90) return { key: 'highly_plausible', en: 'Highly Plausible', zh: '高度可信' };
  if (score >= 75) return { key: 'mostly_plausible', en: 'Mostly Plausible', zh: '基本合理' };
  if (score >= 50) return { key: 'needs_improvement', en: 'Needs Improvement', zh: '有待改进' };
  return { key: 'physically_inconsistent', en: 'Physically Inconsistent', zh: '物理不一致' };
}

function pushIssue(issues, candidate) {
  // Merge nearby issues of the same type within 0.35s so timelines stay clean and actionable
  const existing = issues.find(
    (item) =>
      item.code === candidate.code &&
      item.bodyPart === candidate.bodyPart &&
      Math.abs(item.endTime - candidate.time) <= 0.35,
  );
  if (existing) {
    existing.endTime = Math.max(existing.endTime, candidate.time);
    if (candidate.magnitude > existing.magnitude) {
      existing.magnitude = candidate.magnitude;
      existing.peakTime = candidate.time;
      existing.position = candidate.position;
      existing.explanation = candidate.explanation;
    }
    const rank = { Critical: 4, High: 3, Moderate: 2, Low: 1 };
    if (rank[candidate.severity] > rank[existing.severity]) {
      existing.severity = candidate.severity;
    }
    return;
  }
  issues.push({
    id: `issue-${issues.length + 1}`,
    code: candidate.code,
    title: candidate.title,
    severity: candidate.severity,
    category: candidate.category,
    time: candidate.time,
    peakTime: candidate.time,
    endTime: candidate.time,
    bodyPart: candidate.bodyPart,
    targetName: candidate.targetName || null,
    position: candidate.position,
    magnitude: candidate.magnitude,
    explanation: candidate.explanation,
    recommendation: candidate.recommendation,
  });
}

function getRobotJointNodes(robot) {
  return {
    rollL: robot.getObjectByName('L_Hip_roll'),
    hipL: robot.getObjectByName('L_Hip_pitch'),
    kneeL: robot.getObjectByName('L_Knee_pitch'),
    wheelL: robot.getObjectByName('L_Wheel_spin'),
    rollR: robot.getObjectByName('R_Hip_roll'),
    hipR: robot.getObjectByName('R_Hip_pitch'),
    kneeR: robot.getObjectByName('R_Knee_pitch'),
    wheelR: robot.getObjectByName('R_Wheel_spin'),
  };
}

/**
 * Evaluate a single robot state in-place (read-only on scene; updates matrixWorld of robot).
 */
export function inspectFrameState({
  pose,
  time,
  robot,
  posture,
  joints,
  envManager,
  envId,
}) {
  robot.updateMatrixWorld(true);
  const includeSteppable = envId === 'urban-sidewalk' || envId === 'obstacle-park';

  // Approximate torso center (0, 0.32, 0) in robot model local frame -> posture local (0, 0.072, 0)
  _vTorso.set(0, 0.072, 0).applyMatrix4(posture.matrixWorld);
  posture.getWorldQuaternion(_qPosture);
  _vUp.copy(_yAxis).applyQuaternion(_qPosture).normalize();

  if (joints.kneeL) joints.kneeL.getWorldPosition(_vKneeL);
  else _vKneeL.copy(_vTorso);
  if (joints.kneeR) joints.kneeR.getWorldPosition(_vKneeR);
  else _vKneeR.copy(_vTorso);

  if (joints.wheelL) {
    joints.wheelL.getWorldPosition(_vWheelL);
    joints.wheelL.getWorldQuaternion(_qWheelL);
    _axleL.copy(_xAxis).applyQuaternion(_qWheelL).normalize();
  } else {
    _vWheelL.set(pose.x, robot.position.y + WHEEL_RADIUS, pose.z);
    _axleL.set(1, 0, 0);
  }

  if (joints.wheelR) {
    joints.wheelR.getWorldPosition(_vWheelR);
    joints.wheelR.getWorldQuaternion(_qWheelR);
    _axleR.copy(_xAxis).applyQuaternion(_qWheelR).normalize();
  } else {
    _vWheelR.set(pose.x, robot.position.y + WHEEL_RADIUS, pose.z);
    _axleR.set(1, 0, 0);
  }

  const supportL = WHEEL_RADIUS * Math.sqrt(Math.max(0, 1 - _axleL.y ** 2)) + WHEEL_HALF_WIDTH * Math.abs(_axleL.y);
  const supportR = WHEEL_RADIUS * Math.sqrt(Math.max(0, 1 - _axleR.y ** 2)) + WHEEL_HALF_WIDTH * Math.abs(_axleR.y);
  const wheelBottomL = _vWheelL.y - supportL;
  const wheelBottomR = _vWheelR.y - supportR;

  const groundC = envManager
    ? envManager.sampleTerrain(pose.x, pose.z, { includeSteppable })
    : { hit: true, height: 0, normal: { x: 0, y: 1, z: 0 }, slopeDeg: 0, walkable: true };
  const groundL = envManager
    ? envManager.sampleTerrain(_vWheelL.x, _vWheelL.z, { includeSteppable })
    : { hit: true, height: 0, normal: { x: 0, y: 1, z: 0 }, slopeDeg: 0, walkable: true };
  const groundR = envManager
    ? envManager.sampleTerrain(_vWheelR.x, _vWheelR.z, { includeSteppable })
    : { hit: true, height: 0, normal: { x: 0, y: 1, z: 0 }, slopeDeg: 0, walkable: true };

  const clearanceL = wheelBottomL - groundL.height;
  const clearanceR = wheelBottomR - groundR.height;

  // Approximate geometric stability metrics (CoM projection proxy vs 2-wheel support segment)
  const midX = (_vWheelL.x + _vWheelR.x) * 0.5;
  const midZ = (_vWheelL.z + _vWheelR.z) * 0.5;
  const dxMid = _vTorso.x - midX;
  const dzMid = _vTorso.z - midZ;
  const fwdX = Math.sin(pose.yaw);
  const fwdZ = Math.cos(pose.yaw);
  const rightX = Math.cos(pose.yaw);
  const rightZ = -Math.sin(pose.yaw);
  const comSagittalOffset = dxMid * fwdX + dzMid * fwdZ;
  const comLateralOffset = dxMid * rightX + dzMid * rightZ;
  const bodyTiltDeg = Math.acos(clamp(_vUp.y, -1, 1)) * RAD_TO_DEG;

  // Obstacles (Obstacle_* only, Decor_* excluded)
  let minObstacleClearance = Infinity;
  const collisions = [];
  if (envManager?.obstacles?.length) {
    const bodyProxies = [
      { label: 'Torso', pos: _vTorso, radius: 0.095 },
      { label: 'Left Leg', pos: _vKneeL, radius: 0.036 },
      { label: 'Right Leg', pos: _vKneeR, radius: 0.036 },
      { label: 'Left Wheel', pos: _vWheelL, radius: 0.048, isWheel: true },
      { label: 'Right Wheel', pos: _vWheelR, radius: 0.048, isWheel: true },
    ];
    for (const obs of envManager.obstacles) {
      for (const proxy of bodyProxies) {
        // Steppable surfaces (StepStones / Raised_Slab) support wheels on top without counting as side collision
        if (obs.steppable && proxy.isWheel && proxy.pos.y - WHEEL_RADIUS >= obs.maxY - 0.032) {
          continue;
        }
        const pen = sphereIntersectsBox3(proxy.pos, proxy.radius, obs.box);
        if (pen > 0.004) {
          collisions.push({
            bodyPart: proxy.label,
            obstacleName: obs.name,
            penetration: pen,
            position: { x: proxy.pos.x, y: proxy.pos.y, z: proxy.pos.z },
          });
        }
        const cx = clamp(proxy.pos.x, obs.minX, obs.maxX);
        const cy = clamp(proxy.pos.y, obs.minY, obs.maxY);
        const cz = clamp(proxy.pos.z, obs.minZ, obs.maxZ);
        const d = Math.hypot(proxy.pos.x - cx, proxy.pos.y - cy, proxy.pos.z - cz) - proxy.radius;
        if (d < minObstacleClearance) minObstacleClearance = d;
      }
    }
  }

  // Body & knee terrain penetration
  const torsoBottomY = _vTorso.y - 0.095;
  const kneeClearanceL = _vKneeL.y - 0.028 - groundL.height;
  const kneeClearanceR = _vKneeR.y - 0.028 - groundR.height;

  const hipPitchL = (joints.hipL?.rotation.x ?? 0) * RAD_TO_DEG;
  const hipPitchR = (joints.hipR?.rotation.x ?? 0) * RAD_TO_DEG;
  const kneePitchL = (joints.kneeL?.rotation.x ?? 0) * RAD_TO_DEG;
  const kneePitchR = (joints.kneeR?.rotation.x ?? 0) * RAD_TO_DEG;
  const hipRollL = (joints.rollL?.rotation.z ?? 0) * RAD_TO_DEG;
  const hipRollR = (joints.rollR?.rotation.z ?? 0) * RAD_TO_DEG;

  return {
    time,
    pose: { ...pose },
    groundC,
    groundL,
    groundR,
    clearanceL,
    clearanceR,
    wheelL: { x: _vWheelL.x, y: _vWheelL.y, z: _vWheelL.z, bottomY: wheelBottomL },
    wheelR: { x: _vWheelR.x, y: _vWheelR.y, z: _vWheelR.z, bottomY: wheelBottomR },
    torso: { x: _vTorso.x, y: _vTorso.y, z: _vTorso.z, bottomY: torsoBottomY },
    kneeClearanceL,
    kneeClearanceR,
    comSagittalOffset,
    comLateralOffset,
    bodyTiltDeg,
    bodyPitchDeg: (pose.bodyPitch || 0) * RAD_TO_DEG,
    bankDeg: (pose.bank || 0) * RAD_TO_DEG,
    hipPitchL,
    hipPitchR,
    kneePitchL,
    kneePitchR,
    hipRollL,
    hipRollR,
    collisions,
    minObstacleClearance,
  };
}

/**
 * Full-trajectory local physical validation engine (Phase 1 & 2).
 */
export function evaluateMotionSequence({
  motion,
  robot,
  posture,
  poseRig,
  envManager,
  envId,
  envMeta,
  weights = DEFAULT_CRITIC_WEIGHTS,
}) {
  if (!motion || !robot || !posture || !poseRig) {
    return null;
  }

  const savedRobotPos = robot.position.clone();
  const savedRobotRotY = robot.rotation.y;
  const joints = getRobotJointNodes(robot);

  const duration = Math.max(0.05, motion.duration || 1);
  const sampleCount = clamp(Math.round(duration * 22), 36, 96);
  const dt = duration / (sampleCount - 1);
  const frames = [];

  try {
    const includeSteppable = envId === 'urban-sidewalk' || envId === 'obstacle-park';
    for (let i = 0; i < sampleCount; i++) {
      const t = (i / (sampleCount - 1)) * duration;
      const p = motion.sample(t);
      const terrain = envManager
        ? envManager.sampleTerrain(p.x, p.z, { includeSteppable })
        : { height: 0 };
      const bodyY = typeof p.groundY === 'number' ? p.groundY : terrain.height;
      robot.position.set(p.x, bodyY, p.z);
      robot.rotation.y = p.yaw;
      poseRig(posture, p);
      frames.push(
        inspectFrameState({
          pose: p,
          time: t,
          robot,
          posture,
          joints,
          envManager,
          envId,
        }),
      );
    }
  } finally {
    robot.position.copy(savedRobotPos);
    robot.rotation.y = savedRobotRotY;
    robot.updateMatrixWorld(true);
  }

  const issues = [];
  const stats = {
    sampleCount,
    duration,
    maxPenetrationM: 0,
    maxFloatGapM: 0,
    maxObstaclePenM: 0,
    minObstacleDistM: Infinity,
    maxSlopeDeg: 0,
    avgSlopeDeg: 0,
    maxSpeedMps: 0,
    maxAccelMps2: 0,
    maxPitchDeg: 0,
    maxBankDeg: 0,
    maxTiltDeg: 0,
    maxJointRateDegPerSec: 0,
    maxSlipRatioDelta: 0,
    collisionCount: 0,
    unwalkableFrames: 0,
  };

  let groundPenalty = 0;
  let collisionPenalty = 0;
  let jointPenalty = 0;
  let continuityPenalty = 0;
  let balancePenalty = 0;
  let terrainPenalty = 0;

  let prevSpeed = 0;
  let prevFrame = null;
  let slopeSum = 0;

  const expectedSandSink = envId === 'sandy-beach' ? 0.007 : 0;

  for (let i = 0; i < frames.length; i++) {
    const f = frames[i];
    slopeSum += f.groundC.slopeDeg;
    if (f.groundC.slopeDeg > stats.maxSlopeDeg) stats.maxSlopeDeg = f.groundC.slopeDeg;
    if (!f.groundC.walkable) stats.unwalkableFrames += 1;
    if (f.minObstacleClearance < stats.minObstacleDistM) {
      stats.minObstacleDistM = f.minObstacleClearance;
    }

    // 1. Ground Contact checks
    const stepAllowance = (f.pose.stepActivity || 0) * 0.05;
    for (const [side, clr, wPos, gSample] of [
      ['Left Wheel', f.clearanceL, f.wheelL, f.groundL],
      ['Right Wheel', f.clearanceR, f.wheelR, f.groundR],
    ]) {
      const pen = -(clr + expectedSandSink);
      if (pen > stats.maxPenetrationM) stats.maxPenetrationM = pen;
      if (pen > 0.012) {
        const severe = pen > 0.032;
        groundPenalty += severe ? 16 : 6;
        pushIssue(issues, {
          code: 'wheel-penetration',
          title: `${side === 'Left Wheel' ? '左轮' : '右轮'}陷入地面`,
          severity: severe ? 'Critical' : 'High',
          category: 'Ground Contact',
          time: f.time,
          bodyPart: side,
          position: { x: wPos.x, y: gSample.height, z: wPos.z },
          magnitude: pen,
          explanation: `${side === 'Left Wheel' ? '左' : '右'}轮底面低于实际地形高度 ${(pen * 100).toFixed(1)} cm。`,
          recommendation: '提高对应侧腿部离地补偿或减小下坡/侧倾压缩量以保持轮面贴地。',
        });
      }

      const gap = clr - stepAllowance;
      if (gap > stats.maxFloatGapM) stats.maxFloatGapM = gap;
      if (gap > 0.022) {
        const highGap = gap > 0.045;
        groundPenalty += highGap ? 10 : 4;
        pushIssue(issues, {
          code: 'wheel-floating',
          title: `${side === 'Left Wheel' ? '左轮' : '右轮'}悬空脱离地面`,
          severity: highGap ? 'High' : 'Moderate',
          category: 'Ground Contact',
          time: f.time,
          bodyPart: side,
          position: { x: wPos.x, y: wPos.y, z: wPos.z },
          magnitude: gap,
          explanation: `${side === 'Left Wheel' ? '左' : '右'}轮距离下方地形高出 ${(gap * 100).toFixed(1)} cm，存在悬空失真。`,
          recommendation: '增大下伸腿长或平滑高度过渡曲线，避免地形凹陷处单轮悬空。',
        });
      }
    }

    // Terrain clipping for torso / knees
    if (f.torso.bottomY < f.groundC.height + 0.03) {
      collisionPenalty += 22;
      pushIssue(issues, {
        code: 'torso-ground-clip',
        title: '机身底部触地/穿模',
        severity: 'Critical',
        category: 'Collision Avoidance',
        time: f.time,
        bodyPart: 'Torso',
        position: { x: f.torso.x, y: f.groundC.height, z: f.torso.z },
        magnitude: f.groundC.height + 0.03 - f.torso.bottomY,
        explanation: '机身下沿逼近或穿透地面网格，说明下蹲或地形起伏超出安全净空。',
        recommendation: '抬高机身基准高度并限制剧烈下蹲幅度。',
      });
    }
    for (const [label, kClr, wPos] of [
      ['Left Leg', f.kneeClearanceL, f.wheelL],
      ['Right Leg', f.kneeClearanceR, f.wheelR],
    ]) {
      if (kClr < 0) {
        collisionPenalty += 14;
        pushIssue(issues, {
          code: 'leg-ground-clip',
          title: `${label === 'Left Leg' ? '左腿' : '右腿'}关节擦地`,
          severity: 'High',
          category: 'Collision Avoidance',
          time: f.time,
          bodyPart: label,
          position: { x: wPos.x, y: f.groundC.height + 0.02, z: wPos.z },
          magnitude: -kClr,
          explanation: '膝关节近似碰撞体低于地形高度。',
          recommendation: '增大膝关节弯曲抬升高度以保护下肢连杆。',
        });
      }
    }

    // 2. Obstacle Collisions
    for (const col of f.collisions) {
      stats.collisionCount += 1;
      if (col.penetration > stats.maxObstaclePenM) stats.maxObstaclePenM = col.penetration;
      const isCritical = col.penetration > 0.022 || col.bodyPart === 'Torso';
      collisionPenalty += isCritical ? 24 : 12;
      pushIssue(issues, {
        code: 'obstacle-collision',
        title: `${col.bodyPart} 与障碍物发生碰撞`,
        severity: isCritical ? 'Critical' : 'High',
        category: 'Collision Avoidance',
        time: f.time,
        bodyPart: col.bodyPart,
        targetName: col.obstacleName,
        position: col.position,
        magnitude: col.penetration,
        explanation: `${col.bodyPart} 近似包络与 ${col.obstacleName} 发生 ${(col.penetration * 100).toFixed(1)} cm 交叠。`,
        recommendation: `调整路径规划安全边距，使轨迹绕开 ${col.obstacleName} 或提升跨步高度。`,
      });
    }

    // 3. Joint Validity
    for (const [label, hipDeg, kneeDeg, rollDeg] of [
      ['Left Leg', f.hipPitchL, f.kneePitchL, f.hipRollL],
      ['Right Leg', f.hipPitchR, f.kneePitchR, f.hipRollR],
    ]) {
      if (
        hipDeg < APPROX_JOINT_LIMITS.hipPitchMinDeg + 0.5 ||
        hipDeg > APPROX_JOINT_LIMITS.hipPitchMaxDeg - 0.5 ||
        kneeDeg < APPROX_JOINT_LIMITS.kneePitchMinDeg + 0.5 ||
        kneeDeg > APPROX_JOINT_LIMITS.kneePitchMaxDeg - 0.5
      ) {
        jointPenalty += 5;
        pushIssue(issues, {
          code: 'joint-saturation',
          title: `${label === 'Left Leg' ? '左腿' : '右腿'}关节接近近似极限`,
          severity: 'Moderate',
          category: 'Joint Validity',
          time: f.time,
          bodyPart: label,
          position: { x: f.pose.x, y: f.torso.y - 0.08, z: f.pose.z },
          magnitude: Math.max(Math.abs(hipDeg), Math.abs(kneeDeg)),
          explanation: `髋俯仰 (${hipDeg.toFixed(1)}°) 或膝俯仰 (${kneeDeg.toFixed(1)}°) 触及原型近似行程边界。`,
          recommendation: '减小单侧腿长补偿差或放缓加减速俯仰角度。',
        });
      }
      if (Math.abs(rollDeg) > APPROX_JOINT_LIMITS.hipRollMaxDeg) {
        jointPenalty += 6;
        pushIssue(issues, {
          code: 'hip-roll-excess',
          title: `${label === 'Left Leg' ? '左髋' : '右髋'}侧摆角过大`,
          severity: 'Moderate',
          category: 'Joint Validity',
          time: f.time,
          bodyPart: label,
          position: { x: f.pose.x, y: f.torso.y - 0.05, z: f.pose.z },
          magnitude: Math.abs(rollDeg),
          explanation: `髋侧摆角达到 ${rollDeg.toFixed(1)}°，超出常规稳定范围。`,
          recommendation: '降低弯道侧倾增益或放慢急转弯速度。',
        });
      }
    }

    // 4. Continuity & Joint Rates
    if (prevFrame) {
      const dist = Math.hypot(
        f.pose.x - prevFrame.pose.x,
        (f.pose.groundY || 0) - (prevFrame.pose.groundY || 0),
        f.pose.z - prevFrame.pose.z,
      );
      const speed = dist / dt;
      const accel = Math.abs(speed - prevSpeed) / dt;
      if (speed > stats.maxSpeedMps) stats.maxSpeedMps = speed;
      if (accel > stats.maxAccelMps2) stats.maxAccelMps2 = accel;

      const yawRate = Math.abs(Math.atan2(Math.sin(f.pose.yaw - prevFrame.pose.yaw), Math.cos(f.pose.yaw - prevFrame.pose.yaw))) / dt * RAD_TO_DEG;
      const pitchRate = Math.abs(f.bodyPitchDeg - prevFrame.bodyPitchDeg) / dt;
      const bankRate = Math.abs(f.bankDeg - prevFrame.bankDeg) / dt;

      if (dist > 0.14 || speed > 1.15) {
        continuityPenalty += 18;
        pushIssue(issues, {
          code: 'position-jump',
          title: '轨迹出现瞬时位移跳变',
          severity: 'Critical',
          category: 'Motion Continuity',
          time: f.time,
          bodyPart: 'Torso',
          position: { x: f.pose.x, y: f.groundC.height + 0.12, z: f.pose.z },
          magnitude: speed,
          explanation: `相邻帧瞬时速度达 ${speed.toFixed(2)} m/s，可能存在非连续路径跳变。`,
          recommendation: '在多段航点衔接处加入平滑过渡，避免位置阶跃。',
        });
      }

      if (pitchRate > 95 || bankRate > 95 || yawRate > 150) {
        continuityPenalty += 6;
        pushIssue(issues, {
          code: 'orientation-snap',
          title: '机身姿态角变化过快',
          severity: 'Moderate',
          category: 'Motion Continuity',
          time: f.time,
          bodyPart: 'Torso',
          position: { x: f.torso.x, y: f.torso.y, z: f.torso.z },
          magnitude: Math.max(pitchRate, bankRate, yawRate),
          explanation: `机身角速度峰值约 ${Math.max(pitchRate, bankRate, yawRate).toFixed(0)}°/s，姿态过渡偏急。`,
          recommendation: '扩大地形法向与航向平滑窗口，降低坡度突变处的姿态响应速率。',
        });
      }

      const maxJRate = Math.max(
        Math.abs(f.hipPitchL - prevFrame.hipPitchL) / dt,
        Math.abs(f.hipPitchR - prevFrame.hipPitchR) / dt,
        Math.abs(f.kneePitchL - prevFrame.kneePitchL) / dt,
        Math.abs(f.kneePitchR - prevFrame.kneePitchR) / dt,
      );
      if (maxJRate > stats.maxJointRateDegPerSec) stats.maxJointRateDegPerSec = maxJRate;
      if (maxJRate > APPROX_JOINT_LIMITS.maxJointRateDegPerSec) {
        jointPenalty += 6;
        pushIssue(issues, {
          code: 'joint-velocity-spike',
          title: '腿部关节角速度突变',
          severity: 'Moderate',
          category: 'Joint Validity',
          time: f.time,
          bodyPart: 'Left Leg',
          position: { x: f.pose.x, y: f.torso.y - 0.08, z: f.pose.z },
          magnitude: maxJRate,
          explanation: `腿部关节瞬时角速度达到 ${maxJRate.toFixed(0)}°/s，存在急剧屈伸。`,
          recommendation: '平滑上下台阶或转弯过渡段的腿长目标变化率。',
        });
      }

      // Wheel slip estimate (compare center ground arc step vs wheel spin angle * radius)
      const dSpinAvg = ((f.pose.left - prevFrame.pose.left) + (f.pose.right - prevFrame.pose.right)) * 0.5;
      const wheelArc = Math.abs(dSpinAvg * WHEEL_RADIUS);
      const groundStep = Math.hypot(f.pose.x - prevFrame.pose.x, f.pose.z - prevFrame.pose.z);
      if (groundStep > 0.004) {
        const slipDiff = Math.abs(wheelArc - groundStep) / groundStep;
        if (slipDiff > stats.maxSlipRatioDelta) stats.maxSlipRatioDelta = slipDiff;
        if (envId !== 'sandy-beach' && slipDiff > 0.28) {
          groundPenalty += 3;
          pushIssue(issues, {
            code: 'wheel-slip',
            title: '双轮滚动与位移存在轻微打滑偏差',
            severity: 'Low',
            category: 'Ground Contact',
            time: f.time,
            bodyPart: 'Left Wheel',
            position: { x: f.pose.x, y: f.groundC.height + 0.02, z: f.pose.z },
            magnitude: slipDiff,
            explanation: `车轮滚转弧长与地面实际位移偏差约 ${(slipDiff * 100).toFixed(0)}%。`,
            recommendation: '同步车轮旋转积分与地形斜坡投影弧长以减少视觉滑移。',
          });
        }
      }

      prevSpeed = speed;
    }

    // 5. Balance & Stability (Approximate)
    if (Math.abs(f.bodyPitchDeg) > stats.maxPitchDeg) stats.maxPitchDeg = Math.abs(f.bodyPitchDeg);
    if (Math.abs(f.bankDeg) > stats.maxBankDeg) stats.maxBankDeg = Math.abs(f.bankDeg);
    if (f.bodyTiltDeg > stats.maxTiltDeg) stats.maxTiltDeg = f.bodyTiltDeg;

    if (f.bodyTiltDeg > 17 || Math.abs(f.comSagittalOffset) > 0.085) {
      const highTilt = f.bodyTiltDeg > 23;
      balancePenalty += highTilt ? 10 : 5;
      pushIssue(issues, {
        code: 'stability-tilt',
        title: '机身倾角接近稳定边界（近似估计）',
        severity: highTilt ? 'High' : 'Moderate',
        category: 'Balance and Stability',
        time: f.time,
        bodyPart: 'Torso',
        position: { x: f.torso.x, y: f.torso.y, z: f.torso.z },
        magnitude: f.bodyTiltDeg,
        explanation: `近似质心前向偏移 ${(f.comSagittalOffset * 100).toFixed(1)} cm，机身总倾角 ${f.bodyTiltDeg.toFixed(1)}°。`,
        recommendation: '降低坡面/弯道中的加速度和侧倾幅度，保持机身近似质心处于双轮支撑区间附近。',
      });
    }

    // 6. Terrain Adaptation checks
    if (!f.groundC.walkable) {
      terrainPenalty += 8;
      pushIssue(issues, {
        code: 'steep-unwalkable',
        title: '路径经过高坡度难行区域',
        severity: 'High',
        category: 'Terrain Adaptation',
        time: f.time,
        bodyPart: 'Torso',
        position: { x: f.pose.x, y: f.groundC.height + 0.05, z: f.pose.z },
        magnitude: f.groundC.slopeDeg,
        explanation: `局部地形坡度达到 ${f.groundC.slopeDeg.toFixed(1)}°，超出建议可通行阈值 (38°)。`,
        recommendation: '绕行陡坡并改用等高线缓坡路径。',
      });
    } else if (envId === 'hilly-terrain' && f.groundC.slopeDeg > 16 && stats.maxSpeedMps > 0.36) {
      terrainPenalty += 4;
      pushIssue(issues, {
        code: 'hill-overspeed',
        title: '山地坡面行进速度偏快',
        severity: 'Low',
        category: 'Terrain Adaptation',
        time: f.time,
        bodyPart: 'Torso',
        position: { x: f.pose.x, y: f.groundC.height + 0.05, z: f.pose.z },
        magnitude: stats.maxSpeedMps,
        explanation: `在 ${f.groundC.slopeDeg.toFixed(1)}° 斜坡上瞬时速度达 ${stats.maxSpeedMps.toFixed(2)} m/s。`,
        recommendation: '在起伏山地适度下调最大速度以增强地形贴合度。',
      });
    } else if (envId === 'sandy-beach' && Math.abs(f.bankDeg) > 10) {
      terrainPenalty += 5;
      pushIssue(issues, {
        code: 'sand-bank-large',
        title: '沙地转弯侧倾偏大',
        severity: 'Moderate',
        category: 'Terrain Adaptation',
        time: f.time,
        bodyPart: 'Torso',
        position: { x: f.torso.x, y: f.torso.y, z: f.torso.z },
        magnitude: Math.abs(f.bankDeg),
        explanation: `松软沙面模式下机身侧倾达 ${Math.abs(f.bankDeg).toFixed(1)}°，可能增加侧滑风险。`,
        recommendation: '减小沙地转弯侧倾系数并适当放宽转弯半径。',
      });
    }

    prevFrame = f;
  }

  stats.avgSlopeDeg = slopeSum / frames.length;

  // Sandy beach intentional visual slip note (informational low check if smooth)
  const categories = {
    groundContact: Math.round(clamp(100 - groundPenalty, 0, 100)),
    collisionAvoidance: Math.round(clamp(100 - collisionPenalty, 0, 100)),
    jointValidity: Math.round(clamp(100 - jointPenalty, 0, 100)),
    motionContinuity: Math.round(clamp(100 - continuityPenalty, 0, 100)),
    balanceStability: Math.round(clamp(100 - balancePenalty, 0, 100)),
    terrainAdaptation: Math.round(clamp(100 - terrainPenalty, 0, 100)),
  };

  const wSum =
    weights.groundContact +
    weights.collisionAvoidance +
    weights.jointValidity +
    weights.motionContinuity +
    weights.balanceStability +
    weights.terrainAdaptation;

  const rawWeighted =
    (categories.groundContact * weights.groundContact +
      categories.collisionAvoidance * weights.collisionAvoidance +
      categories.jointValidity * weights.jointValidity +
      categories.motionContinuity * weights.motionContinuity +
      categories.balanceStability * weights.balanceStability +
      categories.terrainAdaptation * weights.terrainAdaptation) /
    (wSum || 1);

  const hasCritical = issues.some((it) => it.severity === 'Critical');
  const hasHigh = issues.some((it) => it.severity === 'High');

  let overallScore = Math.round(rawWeighted);
  if (hasCritical) {
    overallScore = Math.min(overallScore, 48);
  } else if (hasHigh) {
    overallScore = Math.min(overallScore, 72);
  }

  const severityOrder = { Critical: 0, High: 1, Moderate: 2, Low: 3 };
  issues.sort((a, b) => severityOrder[a.severity] - severityOrder[b.severity] || a.time - b.time);

  return {
    envId,
    envLabel: envMeta ? `${envMeta.en} · ${envMeta.zh}` : envId,
    modeLabel: envMeta ? `${envMeta.modeEn} · ${envMeta.modeZh}` : '',
    overallScore,
    status: statusFromScore(overallScore),
    hasCritical,
    hasHigh,
    weights: { ...weights },
    categories,
    issues,
    stats,
    framesSummary: {
      start: { x: frames[0].pose.x, y: frames[0].groundC.height, z: frames[0].pose.z },
      end: {
        x: frames[frames.length - 1].pose.x,
        y: frames[frames.length - 1].groundC.height,
        z: frames[frames.length - 1].pose.z,
      },
      detoured: !!motion.detoured,
      avoidedNames: motion.avoidedNames || [],
    },
    limitations: [
      'GLB 模型不包含质量分布、转动惯量、地面摩擦系数及电机力矩极限，稳定度与关节限位均为几何近似指标。',
      '评分属于原型阶段运动学与几何检查指标，不代表经动力学验证的真实物理仿真结论。',
    ],
  };
}

export function generateLocalExplanation(report) {
  const { envLabel, overallScore, status, issues, stats, categories, framesSummary } = report;

  const summaryParts = [
    `在「${envLabel}」环境中，本段历时 ${stats.duration.toFixed(2)} s 的运动序列获得 ${overallScore}/100（${status.zh} · ${status.en}）。`,
  ];
  if (issues.length === 0) {
    summaryParts.push(
      `全序列共采样 ${stats.sampleCount} 帧，未检测到明显地面穿模、悬空失真或障碍物碰撞。最大局部坡度为 ${stats.maxSlopeDeg.toFixed(1)}°，最大机身倾角 ${stats.maxTiltDeg.toFixed(1)}°。`,
    );
  } else {
    const top = issues[0];
    summaryParts.push(
      `共检测到 ${issues.length} 项物理一致性问题，首要问题为 [${top.severity}] ${top.title}（约 ${top.peakTime.toFixed(2)} s，${top.bodyPart}）。`,
    );
  }

  let terrainNotes = '';
  switch (report.envId) {
    case 'flat-ground':
      terrainNotes = `平地基准测试下，双轮贴地评分 ${categories.groundContact}/100，运动连续性 ${categories.motionContinuity}/100；最大移动速度 ${stats.maxSpeedMps.toFixed(2)} m/s。`;
      break;
    case 'hilly-terrain':
      terrainNotes = `起伏山地最大坡度 ${stats.maxSlopeDeg.toFixed(1)}°（平均 ${stats.avgSlopeDeg.toFixed(1)}°），最大机身俯仰 ${stats.maxPitchDeg.toFixed(1)}°、侧倾 ${stats.maxBankDeg.toFixed(1)}°。${categories.terrainAdaptation >= 85 ? '机身姿态与左右腿差动补偿整体跟随坡面变化。' : '建议在陡坡过渡段降低速度并增大姿态平滑幅度。'}`;
      break;
    case 'obstacle-park':
      terrainNotes = `${framesSummary.detoured ? `路径规划已自动绕行障碍物（${framesSummary.avoidedNames.join(', ') || 'Obstacle_*'}）。` : '当前轨迹未触发绕行。'}最小障碍净距 ${Number.isFinite(stats.minObstacleDistM) ? (stats.minObstacleDistM * 100).toFixed(1) + ' cm' : '无障碍'}，已忽略 Decor_* 装饰物。`;
      break;
    case 'sandy-beach':
      terrainNotes = `松软沙面模式下最大速度限制在 ${stats.maxSpeedMps.toFixed(2)} m/s，最大侧倾 ${stats.maxBankDeg.toFixed(1)}°，包含简化的轮地视觉滑移与微下沉效果（非真实颗粒沙土物理）。`;
      break;
    case 'urban-sidewalk':
      terrainNotes = `城市人行道环境中，最小障碍（含路缘石）净距 ${Number.isFinite(stats.minObstacleDistM) ? (stats.minObstacleDistM * 100).toFixed(1) + ' cm' : '无障碍'}；台阶适应评分 ${categories.terrainAdaptation}/100。`;
      break;
    default:
      terrainNotes = `地形适应评分 ${categories.terrainAdaptation}/100。`;
  }

  const recommendations = [];
  const seenRec = new Set();
  for (const issue of issues) {
    if (!seenRec.has(issue.recommendation)) {
      seenRec.add(issue.recommendation);
      recommendations.push(`[${issue.severity}] ${issue.recommendation}`);
    }
    if (recommendations.length >= 5) break;
  }
  if (recommendations.length === 0) {
    if (report.envId === 'hilly-terrain') {
      recommendations.push('当前动作在山地起伏下贴地良好；若向更陡斜坡移动，建议缩短单次弧线跨度并预留腿部伸展余量。');
    } else if (report.envId === 'obstacle-park') {
      recommendations.push('障碍避让有效；如需登上低矮踏脚石，可将目标点直接设定在踏脚石顶面中心以获得稳定支撑。');
    } else if (report.envId === 'sandy-beach') {
      recommendations.push('松软沙面下姿态平稳；建议继续保持低侧倾与大半径转弯以减少视觉侧滑风险。');
    } else if (report.envId === 'urban-sidewalk') {
      recommendations.push('人行道与路缘净空安全；靠近高出路面台阶时保持垂直正向切入有助于双腿同步抬升。');
    } else {
      recommendations.push('平地基准轨迹连续且贴地稳定，可切换至起伏山地或障碍公园进一步对比环境适应能力。');
    }
  }

  return {
    source: 'local-rules',
    sourceLabel: 'Local Explanation · 本地规则分析（未连接外部 AI 服务）',
    summary: summaryParts.join(' '),
    terrainNotes,
    recommendations,
    uncertainties: report.limitations,
  };
}

export async function requestAiInterpretation(report) {
  const payload = {
    envId: report.envId,
    envLabel: report.envLabel,
    modeLabel: report.modeLabel,
    overallScore: report.overallScore,
    status: report.status,
    categories: report.categories,
    stats: {
      ...report.stats,
      minObstacleDistM: Number.isFinite(report.stats.minObstacleDistM)
        ? Number(report.stats.minObstacleDistM.toFixed(3))
        : null,
    },
    framesSummary: report.framesSummary,
    issues: report.issues.slice(0, 10).map((it) => ({
      title: it.title,
      severity: it.severity,
      category: it.category,
      timeRange: `${it.time.toFixed(2)}s - ${it.endTime.toFixed(2)}s`,
      bodyPart: it.bodyPart,
      targetName: it.targetName,
      explanation: it.explanation,
    })),
    limitations: report.limitations,
  };

  try {
    const controller = new AbortController();
    const timer = setTimeout(() => controller.abort(), 9000);
    const res = await fetch('./api/critic', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload),
      signal: controller.signal,
    });
    clearTimeout(timer);

    if (res.ok) {
      const data = await res.json();
      if (data?.available && data?.analysis) {
        return {
          source: 'ai-model',
          sourceLabel: `AI Interpretation · 已连接模型 (${data.model || 'gemini-4-hydrogen'})`,
          summary: data.analysis.summary || '',
          terrainNotes: data.analysis.terrainNotes || '',
          recommendations: Array.isArray(data.analysis.recommendations) ? data.analysis.recommendations : [],
          uncertainties: Array.isArray(data.analysis.uncertainties) && data.analysis.uncertainties.length
            ? data.analysis.uncertainties
            : report.limitations,
        };
      }
    }
  } catch {
    // Expected on static hosting (e.g., GitHub Pages) or when backend AI is unavailable
  }

  return generateLocalExplanation(report);
}
