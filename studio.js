import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

import { makeArc, makeEnvironmentMotion, makeRig } from './motion.js';
import { ENVIRONMENTS, EnvironmentManager } from './environment.js';
import {
  evaluateMotionSequence,
  inspectFrameState,
  requestAiInterpretation,
} from './critic.js';

const $ = (id) => document.getElementById(id);
const host = $('scene'), status = $('status');
const LIMIT = 1.35;
const zero = () => ({x:0,z:0,yaw:0,left:0,right:0});
const clamp = (v,a,b) => Math.min(b, Math.max(a,v));
const angle = (v) => Math.atan2(Math.sin(v),Math.cos(v));
const deg = (v) => (v*180/Math.PI).toFixed(0)+'°';
const position = (x,z) => '('+x.toFixed(2)+', '+z.toFixed(2)+') m';

const SEVERITY_COLORS = {
  Critical: 0xdc2626,
  High: 0xea580c,
  Moderate: 0xca8a04,
  Low: 0x125ac5,
};

const CATEGORY_LABELS = [
  ['groundContact', 'Ground Contact · 地面接触'],
  ['collisionAvoidance', 'Collision Avoidance · 碰撞避让'],
  ['jointValidity', 'Joint Validity · 关节合理性'],
  ['motionContinuity', 'Motion Continuity · 运动连续性'],
  ['balanceStability', 'Balance & Stability · 稳定度(近似)'],
  ['terrainAdaptation', 'Terrain Adaptation · 地形适应'],
];

let scene,camera,renderer,controls,robot,posture,poseRig,marker,route;
let basePlane,gridHelper,originRing,envManager,obstacleVisuals,waypointGroup;
let issueVisualGroup, partHighlightMesh = null;
let pose=zero(), motion=null, playing=false, elapsed=0, previous=0, frame=0;
let switchingEnv = false;
let currentReport = null, selectedIssueId = null, analyzing = false;
let liveMonitorEnabled = false, lastLiveCheckAt = 0;

const floor=new T.Plane(new T.Vector3(0,1,0),0);
const ray=new T.Raycaster();
const UP_VEC=new T.Vector3(0,1,0);
const TMP_NORMAL=new T.Vector3();

function thumbSvg(kind){
  switch(kind){
    case 'flat':
      return '<svg class="env-thumb" viewBox="0 0 34 24" aria-hidden="true"><rect width="34" height="24" fill="#dce5eb"/><path d="M0 16h34v8H0z" fill="#b9c6d1"/><path d="M4 18h26M8 21h18" stroke="#7f93a3" stroke-width="1"/></svg>';
    case 'hill':
      return '<svg class="env-thumb" viewBox="0 0 34 24" aria-hidden="true"><rect width="34" height="24" fill="#d9e4df"/><path d="M0 18Q8 10 17 15t17-4v13H0z" fill="#6d9170"/><path d="M0 20Q10 15 20 18t14-3v9H0z" fill="#527455"/></svg>';
    case 'park':
      return '<svg class="env-thumb" viewBox="0 0 34 24" aria-hidden="true"><rect width="34" height="24" fill="#dce7df"/><rect y="16" width="34" height="8" fill="#719c6b"/><rect x="6" y="13" width="6" height="4" fill="#909aa3"/><rect x="14" y="12" width="6" height="5" fill="#7a848d"/><rect x="23" y="10" width="6" height="7" fill="#9d6d48"/></svg>';
    case 'beach':
      return '<svg class="env-thumb" viewBox="0 0 34 24" aria-hidden="true"><rect width="34" height="24" fill="#dce9ef"/><path d="M0 15q8-3 17 0t17 0v9H0z" fill="#dfc99a"/><circle cx="26" cy="14" r="3" fill="#8b8378"/></svg>';
    case 'sidewalk':
      return '<svg class="env-thumb" viewBox="0 0 34 24" aria-hidden="true"><rect width="34" height="24" fill="#dfe4e8"/><rect y="16" width="34" height="8" fill="#b4bcc2"/><rect x="2" y="13" width="5" height="5" fill="#7e8891"/><rect x="27" y="13" width="5" height="5" fill="#7e8891"/><rect x="12" y="14" width="10" height="3" fill="#8f9aa3"/></svg>';
    default:
      return '';
  }
}

function buildEnvironmentSelector(){
  const list = $('env-list');
  if (!list) return;
  list.innerHTML = '';
  for (const env of ENVIRONMENTS) {
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = 'env-btn' + (env.id === 'flat-ground' ? ' active' : '');
    btn.dataset.envId = env.id;
    btn.setAttribute('role', 'radio');
    btn.setAttribute('aria-checked', env.id === 'flat-ground' ? 'true' : 'false');
    btn.innerHTML = `${thumbSvg(env.icon)}<span class="env-text"><span class="env-name">${env.en}</span><span class="env-sub">${env.zh} · ${env.modeZh}</span></span>`;
    btn.addEventListener('click', () => selectEnvironment(env.id));
    list.appendChild(btn);
  }
}

function setSelectorActive(envId, disabled){
  const list = $('env-list');
  if (!list) return;
  for (const btn of list.querySelectorAll('.env-btn')) {
    const active = btn.dataset.envId === envId;
    btn.classList.toggle('active', active);
    btn.setAttribute('aria-checked', active ? 'true' : 'false');
    btn.disabled = !!disabled;
  }
}

function clearObstacleVisuals(){
  if (!obstacleVisuals) return;
  while (obstacleVisuals.children.length) {
    const child = obstacleVisuals.children[0];
    obstacleVisuals.remove(child);
    if (child.geometry) child.geometry.dispose();
    if (child.material) child.material.dispose();
  }
}

function rebuildObstacleVisuals(){
  clearObstacleVisuals();
  if (!envManager || !envManager.obstacles.length) return;
  for (const obs of envManager.obstacles) {
    const boxGeom = new T.BoxGeometry(
      Math.max(0.04, obs.size.x + 0.02),
      Math.max(0.04, obs.size.y + 0.02),
      Math.max(0.04, obs.size.z + 0.02),
    );
    const edges = new T.EdgesGeometry(boxGeom);
    boxGeom.dispose();
    const mat = new T.LineBasicMaterial({
      color: obs.steppable ? 0x2563eb : 0xd97706,
      transparent: true,
      opacity: 0.25,
      depthWrite: false,
    });
    const wire = new T.LineSegments(edges, mat);
    wire.position.copy(obs.center);
    wire.userData.obstacleName = obs.name;
    wire.userData.steppable = obs.steppable;
    wire.visible = false;
    obstacleVisuals.add(wire);
  }
}

function updateObstacleVisuals(robotX, robotZ){
  if (!obstacleVisuals) return;
  const avoided = new Set(motion?.avoidedNames || []);
  const selectedIssue = currentReport?.issues?.find((it) => it.id === selectedIssueId) || null;
  for (const wire of obstacleVisuals.children) {
    const name = wire.userData.obstacleName;
    const obs = envManager?.obstacles.find((o) => o.name === name);
    if (!obs) continue;
    const dx = Math.max(obs.minX - robotX, 0, robotX - obs.maxX);
    const dz = Math.max(obs.minZ - robotZ, 0, robotZ - obs.maxZ);
    const dist = Math.hypot(dx, dz);
    const isAvoided = avoided.has(name);
    const isIssueTarget = selectedIssue?.targetName === name;
    const isNear = dist < 0.48;
    wire.visible = isIssueTarget || isAvoided || isNear;
    if (isIssueTarget) {
      wire.material.color.setHex(SEVERITY_COLORS[selectedIssue.severity] ?? 0xdc2626);
      wire.material.opacity = 1.0;
    } else if (isAvoided) {
      wire.material.color.setHex(0xdc2626);
      wire.material.opacity = 0.9;
    } else if (isNear) {
      wire.material.color.setHex(obs.steppable ? 0x125ac5 : 0xd97706);
      wire.material.opacity = 0.72;
    }
  }
}

function clearWaypoints(){
  if (!waypointGroup) return;
  while (waypointGroup.children.length) {
    const c = waypointGroup.children[0];
    waypointGroup.remove(c);
    if (c.geometry) c.geometry.dispose();
    if (c.material) c.material.dispose();
  }
}

function clearPartHighlight(){
  if (partHighlightMesh) {
    if (partHighlightMesh.parent) partHighlightMesh.parent.remove(partHighlightMesh);
    if (partHighlightMesh.geometry) partHighlightMesh.geometry.dispose();
    if (partHighlightMesh.material) partHighlightMesh.material.dispose();
    partHighlightMesh = null;
  }
}

function clearIssueVisuals(){
  selectedIssueId = null;
  clearPartHighlight();
  if (issueVisualGroup) {
    while (issueVisualGroup.children.length) {
      const c = issueVisualGroup.children[0];
      issueVisualGroup.remove(c);
      if (c.geometry) c.geometry.dispose();
      if (c.material) c.material.dispose();
    }
    issueVisualGroup.visible = false;
  }
  if ($('clear-issue')) $('clear-issue').hidden = true;
  for (const el of document.querySelectorAll('.issue-card.active, .t-marker.active')) {
    el.classList.remove('active');
  }
}

function highlightRobotBodyPart(bodyPart, colorHex){
  clearPartHighlight();
  if (!robot || !bodyPart) return;
  let targetNode = null;
  let boxSize = [0.16, 0.21, 0.21];
  let offset = [0, 0.08, 0];

  if (bodyPart === 'Left Wheel') {
    targetNode = robot.getObjectByName('L_Wheel_spin');
    boxSize = [0.05, 0.115, 0.115];
    offset = [0, 0, 0];
  } else if (bodyPart === 'Right Wheel') {
    targetNode = robot.getObjectByName('R_Wheel_spin');
    boxSize = [0.05, 0.115, 0.115];
    offset = [0, 0, 0];
  } else if (bodyPart === 'Left Leg') {
    targetNode = robot.getObjectByName('L_Knee_pitch');
    boxSize = [0.075, 0.14, 0.075];
    offset = [0, -0.02, 0];
  } else if (bodyPart === 'Right Leg') {
    targetNode = robot.getObjectByName('R_Knee_pitch');
    boxSize = [0.075, 0.14, 0.075];
    offset = [0, -0.02, 0];
  } else {
    targetNode = posture;
    boxSize = [0.17, 0.22, 0.22];
    offset = [0, 0.08, 0];
  }

  if (!targetNode) return;
  const geom = new T.BoxGeometry(boxSize[0], boxSize[1], boxSize[2]);
  const edges = new T.EdgesGeometry(geom);
  geom.dispose();
  partHighlightMesh = new T.LineSegments(
    edges,
    new T.LineBasicMaterial({ color: colorHex, depthTest: false, transparent: true, opacity: 0.95 }),
  );
  partHighlightMesh.position.set(offset[0], offset[1], offset[2]);
  partHighlightMesh.renderOrder = 20;
  targetNode.add(partHighlightMesh);
}

function selectIssue(issue){
  if (!issue) return;
  clearIssueVisuals();
  selectedIssueId = issue.id;
  if ($('clear-issue')) $('clear-issue').hidden = false;

  // Seek timeline to the issue peak timestamp
  if (motion) {
    stop();
    elapsed = clamp(issue.peakTime, 0, motion.duration);
    sample(elapsed);
  }

  const colorHex = SEVERITY_COLORS[issue.severity] ?? 0x125ac5;
  highlightRobotBodyPart(issue.bodyPart, colorHex);

  if (issueVisualGroup && issue.position) {
    const ring = new T.Mesh(
      new T.RingGeometry(0.055, 0.074, 32),
      new T.MeshBasicMaterial({ color: colorHex, side: T.DoubleSide, depthWrite: false }),
    );
    ring.rotation.x = -Math.PI / 2;
    ring.position.set(issue.position.x, issue.position.y + 0.008, issue.position.z);

    const stem = new T.Line(
      new T.BufferGeometry().setFromPoints([
        new T.Vector3(issue.position.x, issue.position.y + 0.008, issue.position.z),
        new T.Vector3(issue.position.x, issue.position.y + 0.24, issue.position.z),
      ]),
      new T.LineBasicMaterial({ color: colorHex }),
    );
    const tip = new T.Mesh(
      new T.SphereGeometry(0.018, 12, 10),
      new T.MeshBasicMaterial({ color: colorHex }),
    );
    tip.position.set(issue.position.x, issue.position.y + 0.24, issue.position.z);

    issueVisualGroup.add(ring, stem, tip);
    issueVisualGroup.visible = true;
  }

  for (const el of document.querySelectorAll(`[data-issue-id="${issue.id}"]`)) {
    el.classList.add('active');
    if (el.classList.contains('issue-card')) {
      el.scrollIntoView({ block: 'nearest', behavior: 'smooth' });
    }
  }
  updateObstacleVisuals(pose.x, pose.z);
  render();
}

function resetCriticPanel(){
  currentReport = null;
  clearIssueVisuals();
  if ($('timeline-markers')) $('timeline-markers').innerHTML = '';
  if ($('critic-score')) $('critic-score').textContent = '—';
  if ($('critic-status')) {
    $('critic-status').textContent = '等待分析运动序列';
    $('critic-status').className = 'status-tag';
  }
  if ($('critic-meta')) $('critic-meta').textContent = '请先生成一段轨迹并点击「Analyze Current Motion」';
  if ($('critic-env')) $('critic-env').textContent = envManager?.currentMeta ? `${envManager.currentMeta.en} · ${envManager.currentMeta.zh}` : '—';
  if ($('critic-duration')) $('critic-duration').textContent = '—';
  if ($('critic-issue-count')) $('critic-issue-count').textContent = '—';
  if ($('critic-categories')) $('critic-categories').innerHTML = '<p class="critic-empty">尚未运行物理验证。</p>';
  if ($('critic-issues')) $('critic-issues').innerHTML = '<p class="critic-empty">暂无检测报告。</p>';
  if ($('critic-source')) {
    $('critic-source').textContent = '未生成';
    $('critic-source').className = 'source-badge';
  }
  if ($('critic-ai')) $('critic-ai').innerHTML = '<p class="critic-empty">点击「Analyze Current Motion」后，将结合本地计算指标生成问题解析、地形建议与模型不确定性说明。</p>';
}

function renderTimelineMarkers(report){
  const hostEl = $('timeline-markers');
  if (!hostEl) return;
  hostEl.innerHTML = '';
  if (!report || !report.issues.length || !report.stats.duration) return;
  const dur = report.stats.duration;
  for (const issue of report.issues) {
    const pct = clamp((issue.peakTime / dur) * 100, 0, 100);
    const btn = document.createElement('button');
    btn.type = 'button';
    btn.className = `t-marker sev-${issue.severity}`;
    btn.dataset.issueId = issue.id;
    btn.style.left = `${pct.toFixed(2)}%`;
    btn.title = `[${issue.severity}] ${issue.title} (${issue.peakTime.toFixed(2)} s)`;
    btn.setAttribute('aria-label', btn.title);
    btn.addEventListener('click', (e) => {
      e.stopPropagation();
      selectIssue(issue);
    });
    hostEl.appendChild(btn);
  }
}

function renderCriticReport(report, aiResult){
  $('critic-score').textContent = String(report.overallScore);
  const tag = $('critic-status');
  tag.textContent = `${report.status.en} · ${report.status.zh}`;
  tag.className = `status-tag ${report.status.key}`;

  $('critic-meta').textContent = report.hasCritical
    ? '检测到严重物理违规（总分已受上限约束），请点击下方问题定位检查。'
    : report.hasHigh
      ? '存在高优先级物理偏差，建议依据下方改进建议调整。'
      : '本地几何与运动学指标整体合理（原型近似评估）。';

  $('critic-env').textContent = report.envLabel;
  $('critic-duration').textContent = `${report.stats.duration.toFixed(2)} s (${report.stats.sampleCount} 帧)`;
  $('critic-issue-count').textContent = `${report.issues.length} 项`;

  const catHost = $('critic-categories');
  catHost.innerHTML = '';
  for (const [key, label] of CATEGORY_LABELS) {
    const val = report.categories[key] ?? 0;
    const item = document.createElement('div');
    item.className = 'cat-item';
    const row = document.createElement('div');
    row.className = 'cat-row';
    const nameEl = document.createElement('span');
    nameEl.textContent = label;
    const valEl = document.createElement('strong');
    valEl.textContent = `${val}/100`;
    row.append(nameEl, valEl);

    const bar = document.createElement('div');
    bar.className = 'cat-bar';
    const fill = document.createElement('div');
    fill.className = 'cat-fill' + (val < 55 ? ' low' : val < 80 ? ' mid' : '');
    fill.style.width = `${val}%`;
    bar.appendChild(fill);

    item.append(row, bar);
    catHost.appendChild(item);
  }

  const issueHost = $('critic-issues');
  issueHost.innerHTML = '';
  if (report.issues.length === 0) {
    const ok = document.createElement('p');
    ok.className = 'critic-empty';
    ok.textContent = '未检测到地面穿透、悬空、障碍碰撞或显著连续性异常。';
    issueHost.appendChild(ok);
  } else {
    for (const issue of report.issues) {
      const card = document.createElement('button');
      card.type = 'button';
      card.className = 'issue-card';
      card.dataset.issueId = issue.id;

      const top = document.createElement('div');
      top.className = 'issue-top';
      const titleEl = document.createElement('span');
      titleEl.className = 'issue-title';
      titleEl.textContent = issue.title;
      const sevEl = document.createElement('span');
      sevEl.className = `sev-pill sev-${issue.severity}`;
      sevEl.textContent = issue.severity;
      top.append(titleEl, sevEl);

      const metaEl = document.createElement('div');
      metaEl.className = 'issue-meta';
      const timeRange = Math.abs(issue.endTime - issue.time) > 0.05
        ? `${issue.time.toFixed(2)}–${issue.endTime.toFixed(2)} s`
        : `${issue.peakTime.toFixed(2)} s`;
      metaEl.textContent = `时间: ${timeRange} · 部位: ${issue.bodyPart}${issue.targetName ? ` · 对象: ${issue.targetName}` : ''}`;

      const descEl = document.createElement('div');
      descEl.className = 'issue-desc';
      descEl.textContent = issue.explanation;

      const fixEl = document.createElement('div');
      fixEl.className = 'issue-fix';
      fixEl.textContent = `建议: ${issue.recommendation}`;

      card.append(top, metaEl, descEl, fixEl);
      card.addEventListener('click', () => selectIssue(issue));
      issueHost.appendChild(card);
    }
  }

  renderTimelineMarkers(report);

  const srcBadge = $('critic-source');
  const aiHost = $('critic-ai');
  if (!aiResult) {
    srcBadge.textContent = '解析生成中…';
    srcBadge.className = 'source-badge';
    aiHost.innerHTML = '<p class="critic-empty">正在基于本地验证指标生成解释与建议…</p>';
    return;
  }

  srcBadge.textContent = aiResult.sourceLabel;
  srcBadge.className = aiResult.source === 'ai-model' ? 'source-badge ai' : 'source-badge';
  aiHost.innerHTML = '';

  const addBlock = (title, content, isList = false) => {
    const blk = document.createElement('div');
    blk.className = 'ai-block';
    const h = document.createElement('h4');
    h.textContent = title;
    blk.appendChild(h);
    if (isList) {
      const ul = document.createElement('ul');
      for (const line of content) {
        const li = document.createElement('li');
        li.textContent = line;
        ul.appendChild(li);
      }
      blk.appendChild(ul);
    } else {
      const p = document.createElement('p');
      p.textContent = content;
      blk.appendChild(p);
    }
    aiHost.appendChild(blk);
  };

  addBlock('总体运动评估 / Summary', aiResult.summary);
  addBlock('地形适应分析 / Terrain Behavior', aiResult.terrainNotes);
  addBlock('具体改进建议 / Recommendations', aiResult.recommendations, true);
  addBlock('数据局限与不确定性 / Limitations', aiResult.uncertainties, true);
}

async function analyzeCurrentMotion(){
  if (!motion || !robot || !posture || !poseRig || analyzing) return;
  analyzing = true;
  const btn = $('analyze-motion');
  if (btn) {
    btn.disabled = true;
    btn.textContent = 'Analyzing Motion… · 正在分析运动…';
  }
  clearIssueVisuals();

  try {
    const meta = envManager?.currentMeta || ENVIRONMENTS[0];
    const report = evaluateMotionSequence({
      motion,
      robot,
      posture,
      poseRig,
      envManager,
      envId: meta.id,
      envMeta: meta,
    });
    if (!report) return;
    currentReport = report;
    renderCriticReport(report, null);

    const aiResult = await requestAiInterpretation(report);
    if (currentReport === report) {
      renderCriticReport(report, aiResult);
    }
  } finally {
    analyzing = false;
    if (btn) {
      btn.disabled = !motion;
      btn.textContent = 'Analyze Current Motion · 分析当前运动';
    }
    render();
  }
}

function runLiveCheck(p, time){
  if (!liveMonitorEnabled || !robot || !posture) return;
  const now = performance.now();
  if (now - lastLiveCheckAt < 160) return;
  lastLiveCheckAt = now;

  const envId = envManager?.currentMeta?.id || 'flat-ground';
  const joints = {
    rollL: robot.getObjectByName('L_Hip_roll'),
    hipL: robot.getObjectByName('L_Hip_pitch'),
    kneeL: robot.getObjectByName('L_Knee_pitch'),
    wheelL: robot.getObjectByName('L_Wheel_spin'),
    rollR: robot.getObjectByName('R_Hip_roll'),
    hipR: robot.getObjectByName('R_Hip_pitch'),
    kneeR: robot.getObjectByName('R_Knee_pitch'),
    wheelR: robot.getObjectByName('R_Wheel_spin'),
  };
  const snap = inspectFrameState({
    pose: p,
    time,
    robot,
    posture,
    joints,
    envManager,
    envId,
  });

  const pill = $('live-status');
  if (!pill) return;
  const expectedSink = envId === 'sandy-beach' ? 0.007 : 0;
  const pen = Math.max(-(snap.clearanceL + expectedSink), -(snap.clearanceR + expectedSink));
  const stepAllowance = (p.stepActivity || 0) * 0.05;
  const gap = Math.max(snap.clearanceL - stepAllowance, snap.clearanceR - stepAllowance);

  if (snap.collisions.length > 0) {
    pill.textContent = `实时监测: ${snap.collisions[0].obstacleName} 碰撞风险`;
    pill.className = 'live-pill warn';
  } else if (pen > 0.015) {
    pill.textContent = `实时监测: 轮面穿模 ${(pen * 100).toFixed(1)} cm`;
    pill.className = 'live-pill warn';
  } else if (gap > 0.025) {
    pill.textContent = `实时监测: 轮面悬空 ${(gap * 100).toFixed(1)} cm`;
    pill.className = 'live-pill warn';
  } else if (!snap.groundC.walkable) {
    pill.textContent = `实时监测: 陡坡 ${snap.groundC.slopeDeg.toFixed(0)}°`;
    pill.className = 'live-pill warn';
  } else {
    pill.textContent = `实时监测: 正常 (${snap.groundC.slopeDeg.toFixed(1)}°)`;
    pill.className = 'live-pill';
  }
}

async function selectEnvironment(envId){
  if (!envManager || switchingEnv) return;
  if (envManager.currentMeta && envManager.currentMeta.id === envId && envManager.root) return;
  switchingEnv = true;
  setSelectorActive(envId, true);
  stop();
  const prevStatus = status.textContent;
  status.textContent = '切换环境中';
  try {
    const loaded = await envManager.load(envId);
    if (!loaded) return;
    const { meta } = loaded;
    if (basePlane) basePlane.visible = meta.id === 'flat-ground';
    if (gridHelper) gridHelper.visible = !!meta.showGrid;
    if (originRing) originRing.visible = meta.id === 'flat-ground';
    rebuildObstacleVisuals();
    if (robot) reset();
    else render();
  } catch (err) {
    console.error('Environment switch failed:', err);
    status.textContent = prevStatus || '环境加载失败';
  } finally {
    switchingEnv = false;
    setSelectorActive(envManager.currentMeta.id, false);
  }
}

function render(){renderer.render(scene,camera);}
function setView(){
  camera.position.set(.98,1.035,1.34);
  controls.target.set(.02,.145,.17);
  controls.update();
  render();
}
function resize(){
  const w=host.clientWidth,h=host.clientHeight;
  if(!w || !h) return;
  camera.aspect=w/h; camera.fov=camera.aspect<1.12 ? 54 : 43;
  camera.updateProjectionMatrix();
  renderer.setSize(w,h);
  render();
}
function update(p,time,length,label){
  pose=p;
  const meta = envManager?.currentMeta || ENVIRONMENTS[0];
  const envId = meta.id;
  const includeSteppable = envId === 'urban-sidewalk' || envId === 'obstacle-park';
  const terrain = envManager ? envManager.sampleTerrain(p.x, p.z, { includeSteppable }) : { height: 0, slopeDeg: 0, walkable: true };
  const bodyY = typeof p.groundY === 'number' ? p.groundY : terrain.height;
  robot.position.set(p.x, bodyY, p.z);
  robot.rotation.y=p.yaw;
  const joints = poseRig(posture,p);

  const nearest = envManager ? envManager.getNearestObstacle(p.x, p.z) : { obstacle: null, distance: Infinity };
  updateObstacleVisuals(p.x, p.z);

  $('env-name').textContent = `${meta.en} · ${meta.zh}`;
  const modeLabel = motion?.detoured && envId === 'obstacle-park'
    ? `${meta.modeZh} (已绕行)`
    : `${meta.modeEn} · ${meta.modeZh}`;
  $('adapt-mode').textContent = modeLabel;
  $('ground-height').textContent = bodyY.toFixed(3) + ' m';
  $('local-slope').textContent = terrain.slopeDeg.toFixed(1) + '°';
  $('nearest-obstacle').textContent = Number.isFinite(nearest.distance)
    ? nearest.distance.toFixed(2) + ' m'
    : '无障碍';

  if ($('hud-mode')) $('hud-mode').textContent = `${meta.modeZh}`;
  if ($('hud-slope-val')) $('hud-slope-val').textContent = terrain.slopeDeg.toFixed(1) + '°';
  if ($('slope-bar')) {
    const tiltDeg = clamp(terrain.slopeDeg, 0, 35);
    const radTilt = (tiltDeg * Math.PI) / 180;
    const dx = Math.cos(radTilt) * 7;
    const dy = Math.sin(radTilt) * 7;
    $('slope-bar').setAttribute('x1', (9 - dx).toFixed(2));
    $('slope-bar').setAttribute('y1', (6 + dy).toFixed(2));
    $('slope-bar').setAttribute('x2', (9 + dx).toFixed(2));
    $('slope-bar').setAttribute('y2', (6 - dy).toFixed(2));
  }

  $('knees').textContent = deg(joints.kneeLeft)+' / '+deg(joints.kneeRight);
  $('bank').textContent = deg(joints.bank);
  $('pos').textContent=position(p.x,p.z);
  $('yaw').textContent=deg(angle(p.yaw));
  $('wheels').textContent=deg(p.left)+' / '+deg(p.right);
  const contactText = !terrain.walkable
    ? '坡度较大 · 注意稳定'
    : envId === 'sandy-beach'
      ? '松软沙面 · 减载稳态'
      : p.stepActivity > 0.15
        ? '台阶抬腿适应中'
        : '双轮贴地 · 视觉';
  $('contact').textContent=contactText;
  $('time').value=length ? time/length : 0;
  $('clock').textContent=time.toFixed(2)+' / '+length.toFixed(2)+' s';
  if(status.textContent!==label) status.textContent=label;
  $('play').textContent=playing?'暂停':(length && time>=length?'重播':'播放');
  if (playing) runLiveCheck(p, time);
  render();
}
function sample(t){
  if(!motion) return;
  const {duration, direction} = motion;
  const label = t >= duration ? '已到达目标' : !playing ? '已暂停' :
    motion.detoured ? '正在避障绕行' :
    direction < 0 ? '正在倒车入弯' : '正在滚动';
  update(motion.sample(t), t, duration, label);
}
function tick(now){
  frame=0;
  if(!playing)return;
  if(previous)elapsed=Math.min(motion.duration,elapsed+Math.min(.08,(now-previous)/1000));
  previous=now;
  if(elapsed>=motion.duration) playing=false;
  sample(elapsed);
  if(playing)frame=requestAnimationFrame(tick);
}
function stop(){
  playing=false;previous=0;
  if(frame)cancelAnimationFrame(frame);
  frame=0;
}
function play(){
  if(!motion||playing)return;
  if(elapsed>=motion.duration)elapsed=0;
  playing=true;previous=0;sample(elapsed);frame=requestAnimationFrame(tick);
}
function plan(x,z){
  if(!robot)return;
  x=clamp(x,-LIMIT,LIMIT); z=clamp(z,-LIMIT,LIMIT);
  const envId = envManager?.currentMeta?.id || 'flat-ground';
  const planned = envManager && envId !== 'flat-ground'
    ? envManager.planWaypoints(pose, x, z, makeArc, LIMIT)
    : { waypoints: [{ x, z }], targetX: x, targetZ: z, adjusted: false, detoured: false, avoidedNames: [] };
  if (!planned) return;
  const tx = planned.targetX;
  const tz = planned.targetZ;
  $('x').value=tx.toFixed(2); $('z').value=tz.toFixed(2);
  const next = makeEnvironmentMotion(pose, planned.waypoints, envManager, envId);
  if (!next) return;
  next.detoured = planned.detoured;
  next.adjusted = planned.adjusted;
  next.avoidedNames = planned.avoidedNames;
  stop();
  motion = next;
  resetCriticPanel();
  if ($('analyze-motion')) $('analyze-motion').disabled = false;

  const includeSteppable = envId === 'urban-sidewalk' || envId === 'obstacle-park';
  const targetSample = envManager ? envManager.sampleTerrain(tx, tz, { includeSteppable }) : { height: 0, normal: { x: 0, y: 1, z: 0 } };
  marker.visible = route.visible = true;
  marker.position.set(tx, targetSample.height + .006, tz);
  TMP_NORMAL.set(targetSample.normal.x, targetSample.normal.y, targetSample.normal.z).normalize();
  marker.quaternion.setFromUnitVectors(UP_VEC, TMP_NORMAL);

  route.material.color.setHex(planned.detoured ? 0x0e7a56 : 0x125ac5);
  const vertices = route.geometry.getAttribute('position');
  for (let i = 0; i < vertices.count; i++) {
    const point = next.at(i / (vertices.count - 1));
    const s = envManager ? envManager.sampleTerrain(point.x, point.z, { includeSteppable }) : { height: 0 };
    vertices.setXYZ(i, point.x, s.height + .012, point.z);
  }
  vertices.needsUpdate = true;
  route.computeLineDistances();

  clearWaypoints();
  if (planned.waypoints.length > 1) {
    for (let i = 0; i < planned.waypoints.length - 1; i++) {
      const wp = planned.waypoints[i];
      const ws = envManager ? envManager.sampleTerrain(wp.x, wp.z, { includeSteppable }) : { height: 0 };
      const dot = new T.Mesh(
        new T.SphereGeometry(.016, 12, 10),
        new T.MeshBasicMaterial({ color: 0x0e7a56 })
      );
      dot.position.set(wp.x, ws.height + .018, wp.z);
      waypointGroup.add(dot);
    }
  }

  elapsed=0;$('play').disabled=$('time').disabled=false;play();
}
function reset(){
  if(!robot)return;
  stop();motion=null;elapsed=0;
  route.visible=marker.visible=false;
  clearWaypoints();
  resetCriticPanel();
  if ($('analyze-motion')) $('analyze-motion').disabled = true;
  $('time').disabled=$('play').disabled=true;
  $('x').value='0.45';$('z').value='0.65';
  update(zero(),0,0,'已就绪');
}
function init(){
  scene=new T.Scene();scene.background=new T.Color(0xdce5eb);scene.fog=new T.Fog(0xdce5eb,3.7,7);
  camera=new T.PerspectiveCamera(43,1,.025,14);
  renderer=new T.WebGLRenderer({antialias:true,powerPreference:'low-power'});
  renderer.setPixelRatio(Math.min(devicePixelRatio,1.5));
  renderer.outputColorSpace=T.SRGBColorSpace;
  host.append(renderer.domElement);
  controls=new OrbitControls(camera,renderer.domElement);
  controls.enablePan=false;controls.maxPolarAngle=Math.PI*.49;controls.minPolarAngle=.16;
  controls.minDistance=.55;controls.maxDistance=4;
  controls.addEventListener('change',render);
  scene.add(new T.HemisphereLight(0xf7faff,0x7d8696,2.35));
  const key=new T.DirectionalLight(0xfff5e7,2);key.position.set(2,4,1);scene.add(key);
  basePlane=new T.Mesh(new T.PlaneGeometry(24,24),new T.MeshBasicMaterial({color:0xdce5eb}));
  basePlane.rotation.x=-Math.PI/2;basePlane.position.y=-.004;scene.add(basePlane);
  gridHelper=new T.GridHelper(2.7,18,0x8599a9,0xa3b5c3);
  gridHelper.position.y=.0015;
  scene.add(gridHelper);
  originRing=new T.Mesh(new T.RingGeometry(.031,.036,36),new T.MeshBasicMaterial({color:0x748c9d}));
  originRing.rotation.x=-Math.PI/2;originRing.position.y=.002;scene.add(originRing);
  marker=new T.Group();
  const accents=[[new T.RingGeometry(.040,.049,40),0x175fc6],[new T.CircleGeometry(.012,20),0x175fc6]];
  for(const [g,color] of accents){
    const m=new T.Mesh(g,new T.MeshBasicMaterial({color,depthWrite:false}));m.rotation.x=-Math.PI/2;marker.add(m);
  }
  const normalStem = new T.Line(
    new T.BufferGeometry().setFromPoints([new T.Vector3(0,0,0), new T.Vector3(0,.055,0)]),
    new T.LineBasicMaterial({color:0x175fc6})
  );
  marker.add(normalStem);
  route=new T.Line(new T.BufferGeometry().setAttribute('position', new T.BufferAttribute(new Float32Array(65 * 3), 3)),new T.LineDashedMaterial({color:0x125ac5,dashSize:.034,gapSize:.024}));
  route.frustumCulled=false;route.visible=marker.visible=false;
  obstacleVisuals = new T.Group();
  waypointGroup = new T.Group();
  issueVisualGroup = new T.Group();
  issueVisualGroup.visible = false;
  scene.add(route,marker,obstacleVisuals,waypointGroup,issueVisualGroup);
  envManager = new EnvironmentManager(scene);
  buildEnvironmentSelector();
  new ResizeObserver(resize).observe(host);
  setView();resize();

  let down=null;
  host.addEventListener('pointerdown',e=>{down=e.isPrimary&&e.button===0?{x:e.clientX,y:e.clientY,moved:false,id:e.pointerId}:null;});
  host.addEventListener('pointermove',e=>{if(down&&Math.hypot(e.clientX-down.x,e.clientY-down.y)>5)down.moved=true;});
  host.addEventListener('pointercancel',()=>{down=null;});
  host.addEventListener('pointerup',e=>{
    const start=down;down=null;
    if(!start||start.id!==e.pointerId||start.moved||Math.hypot(e.clientX-start.x,e.clientY-start.y)>5)return;
    const b=host.getBoundingClientRect(), ndc=new T.Vector2((e.clientX-b.left)/b.width*2-1,1-(e.clientY-b.top)/b.height*2);
    ray.setFromCamera(ndc,camera);
    const picked = envManager ? envManager.pickGroundPoint(ray, { includeSteppable: true }) : null;
    const point = picked ? picked.point : ray.ray.intersectPlane(floor,new T.Vector3());
    if(point && Math.abs(point.x)<=LIMIT && Math.abs(point.z)<=LIMIT)
      plan(Math.round(point.x*100)/100,Math.round(point.z*100)/100);
  });
}
function fail(e){
  console.error(e);status.textContent='预览无法启动';
  $('failure').hidden=false;
  $('failure').textContent='无法加载三维场景。请检查浏览器的图形加速，或关闭页面后用项目启动文件重新打开。';
}
try{
  init();
  Promise.all([
    new Promise((resolve, reject) => new GLTFLoader().load('./assets/Roadrunner-IROS.glb', resolve, undefined, reject)),
    envManager.load('flat-ground'),
  ]).then(([asset, loadedEnv])=>{
    try{
      const model=asset.scene;
      const left=model.getObjectByName('L_Wheel_spin'), right=model.getObjectByName('R_Wheel_spin');
      if(!left||!right) throw new Error('Wheel nodes missing');
      model.updateMatrixWorld(true);
      const ground=Math.min(new T.Box3().setFromObject(left,true).min.y,new T.Box3().setFromObject(right,true).min.y);
      robot=new T.Group();
      posture = new T.Group();
      model.position.y -= .248;
      posture.add(model);
      robot.add(posture);
      poseRig = makeRig(model, ground);
      const shade=document.createElement('canvas'); shade.width=shade.height=64;
      const ctx=shade.getContext('2d'), gradient=ctx.createRadialGradient(32,32,0,32,32,31);
      gradient.addColorStop(0,'rgba(27,41,54,.30)'); gradient.addColorStop(.55,'rgba(27,41,54,.13)'); gradient.addColorStop(1,'rgba(27,41,54,0)');
      ctx.fillStyle=gradient;ctx.fillRect(0,0,64,64);
      const shadow=new T.Mesh(new T.PlaneGeometry(.43,.31),new T.MeshBasicMaterial({map:new T.CanvasTexture(shade),transparent:true,depthWrite:false}));
      shadow.rotation.x=-Math.PI/2;shadow.position.y=.003;robot.add(shadow);
      scene.add(robot);
      if (loadedEnv?.meta) {
        basePlane.visible = loadedEnv.meta.id === 'flat-ground';
        gridHelper.visible = !!loadedEnv.meta.showGrid;
        originRing.visible = loadedEnv.meta.id === 'flat-ground';
        rebuildObstacleVisuals();
      }
      $('go').disabled=$('reset').disabled=false;
      reset();
    }catch(e){fail(e);}
  }).catch(fail);
}catch(e){fail(e);}
$('target').addEventListener('submit',e=>{e.preventDefault();plan($('x').valueAsNumber,$('z').valueAsNumber);});
$('reset').addEventListener('click',reset);
$('view').addEventListener('click',()=>{if(controls)setView();});
$('play').addEventListener('click',()=>{if(playing){stop();sample(elapsed);}else play();});
$('time').addEventListener('input',()=>{if(!motion)return;const v=Number($('time').value);stop();elapsed=v*motion.duration;sample(elapsed);});
$('analyze-motion')?.addEventListener('click', analyzeCurrentMotion);
$('clear-issue')?.addEventListener('click', () => { clearIssueVisuals(); updateObstacleVisuals(pose.x, pose.z); render(); });
$('live-monitor')?.addEventListener('change', (e) => {
  liveMonitorEnabled = !!e.target.checked;
  const pill = $('live-status');
  if (pill) {
    pill.hidden = !liveMonitorEnabled;
    pill.textContent = '实时监测待机';
    pill.className = 'live-pill';
  }
});
