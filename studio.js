import * as T from 'three';
import { OrbitControls } from 'three/addons/controls/OrbitControls.js';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

import { makeArc, makeEnvironmentMotion, makeRig } from './motion.js';
import { ENVIRONMENTS, EnvironmentManager } from './environment.js';

const $ = (id) => document.getElementById(id);
const host = $('scene'), status = $('status');
const LIMIT = 1.35;
const zero = () => ({x:0,z:0,yaw:0,left:0,right:0});
const clamp = (v,a,b) => Math.min(b, Math.max(a,v));
const angle = (v) => Math.atan2(Math.sin(v),Math.cos(v));
const deg = (v) => (v*180/Math.PI).toFixed(0)+'°';
const position = (x,z) => '('+x.toFixed(2)+', '+z.toFixed(2)+') m';

let scene,camera,renderer,controls,robot,posture,poseRig,marker,route;
let basePlane,gridHelper,originRing,envManager,obstacleVisuals,waypointGroup;
let pose=zero(), motion=null, playing=false, elapsed=0, previous=0, frame=0;
let switchingEnv = false;
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
  for (const wire of obstacleVisuals.children) {
    const name = wire.userData.obstacleName;
    const obs = envManager?.obstacles.find((o) => o.name === name);
    if (!obs) continue;
    const dx = Math.max(obs.minX - robotX, 0, robotX - obs.maxX);
    const dz = Math.max(obs.minZ - robotZ, 0, robotZ - obs.maxZ);
    const dist = Math.hypot(dx, dz);
    const isAvoided = avoided.has(name);
    const isNear = dist < 0.48;
    wire.visible = isAvoided || isNear;
    if (isAvoided) {
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
  scene.add(route,marker,obstacleVisuals,waypointGroup);
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
