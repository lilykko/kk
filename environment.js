import * as T from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';

export const ENVIRONMENTS = [
  {
    id: 'flat-ground',
    file: 'flat-ground.glb',
    en: 'Flat Ground',
    zh: '平整地面',
    modeEn: 'Standard Mode',
    modeZh: '标准模式',
    bgColor: 0xdce5eb,
    fogNear: 3.7,
    fogFar: 7.0,
    showGrid: true,
    icon: 'flat',
  },
  {
    id: 'hilly-terrain',
    file: 'hilly-terrain.glb',
    en: 'Hilly Terrain',
    zh: '起伏山地',
    modeEn: 'Slope Adaptation',
    modeZh: '坡度自适应',
    bgColor: 0xd9e4df,
    fogNear: 4.0,
    fogFar: 7.8,
    showGrid: false,
    icon: 'hill',
  },
  {
    id: 'obstacle-park',
    file: 'obstacle-park.glb',
    en: 'Obstacle Park',
    zh: '障碍公园',
    modeEn: 'Obstacle Avoidance',
    modeZh: '避障规划',
    bgColor: 0xdce7df,
    fogNear: 4.0,
    fogFar: 7.8,
    showGrid: false,
    icon: 'park',
  },
  {
    id: 'sandy-beach',
    file: 'sandy-beach.glb',
    en: 'Sandy Beach',
    zh: '沙滩',
    modeEn: 'Soft Terrain Mode',
    modeZh: '松软地形模式',
    bgColor: 0xe6e5db,
    fogNear: 4.0,
    fogFar: 7.8,
    showGrid: false,
    icon: 'beach',
  },
  {
    id: 'urban-sidewalk',
    file: 'urban-sidewalk.glb',
    en: 'Urban Sidewalk',
    zh: '城市人行道',
    modeEn: 'Step Adaptation',
    modeZh: '台阶适应模式',
    bgColor: 0xdfe4e8,
    fogNear: 4.0,
    fogFar: 7.8,
    showGrid: false,
    icon: 'sidewalk',
  },
];

const envLoader = new GLTFLoader();
let manifestCache = null;

export async function loadEnvironmentManifest() {
  if (manifestCache) return manifestCache;
  try {
    const res = await fetch('./environment-manifest.json');
    if (res.ok) {
      manifestCache = await res.json();
      return manifestCache;
    }
  } catch (err) {
    console.warn('Could not load environment-manifest.json, using defaults:', err);
  }
  return null;
}

function loadGltfAsync(url) {
  return new Promise((resolve, reject) => {
    envLoader.load(url, resolve, undefined, reject);
  });
}

async function loadEnvironmentAsset(file) {
  // Support both nested ./environments/environments/<file> and ./environments/<file>
  // using relative paths compatible with GitHub Pages subpaths.
  const candidates = [
    `./environments/environments/${file}`,
    `./environments/${file}`,
  ];
  let lastError = null;
  for (const url of candidates) {
    try {
      const gltf = await loadGltfAsync(url);
      return { gltf, url };
    } catch (err) {
      lastError = err;
    }
  }
  throw lastError || new Error(`Failed to load environment asset: ${file}`);
}

export function disposeObjectTree(root) {
  if (!root) return;
  root.traverse((obj) => {
    if (obj.geometry) {
      obj.geometry.dispose();
    }
    if (obj.material) {
      const materials = Array.isArray(obj.material) ? obj.material : [obj.material];
      for (const mat of materials) {
        if (!mat) continue;
        const texKeys = [
          'map',
          'normalMap',
          'roughnessMap',
          'metalnessMap',
          'aoMap',
          'emissiveMap',
          'bumpMap',
          'alphaMap',
          'envMap',
        ];
        for (const key of texKeys) {
          if (mat[key] && typeof mat[key].dispose === 'function') {
            mat[key].dispose();
          }
        }
        mat.dispose();
      }
    }
  });
}

const UP = new T.Vector3(0, 1, 0);
const DOWN = new T.Vector3(0, -1, 0);
const MAX_WALKABLE_SLOPE_DEG = 38;
const ROBOT_CLEARANCE = 0.19;

function pointToBoxDistance2D(x, z, box) {
  const dx = Math.max(box.minX - x, 0, x - box.maxX);
  const dz = Math.max(box.minZ - z, 0, z - box.maxZ);
  return Math.hypot(dx, dz);
}

function pushOutOfBox2D(x, z, box, margin) {
  const minX = box.minX - margin;
  const maxX = box.maxX + margin;
  const minZ = box.minZ - margin;
  const maxZ = box.maxZ + margin;
  if (x <= minX || x >= maxX || z <= minZ || z >= maxZ) {
    return { x, z, pushed: false };
  }
  const dLeft = x - minX;
  const dRight = maxX - x;
  const dBack = z - minZ;
  const dFront = maxZ - z;
  const minD = Math.min(dLeft, dRight, dBack, dFront);
  if (minD === dLeft) return { x: minX, z, pushed: true };
  if (minD === dRight) return { x: maxX, z, pushed: true };
  if (minD === dBack) return { x, z: minZ, pushed: true };
  return { x, z: maxZ, pushed: true };
}

export class EnvironmentManager {
  constructor(scene) {
    this.scene = scene;
    this.currentMeta = ENVIRONMENTS[0];
    this.root = null;
    this.groundMesh = null;
    this.obstacles = [];
    this.decorObjects = [];
    this.steppableMeshes = [];
    this.loadToken = 0;

    this._ray = new T.Raycaster();
    this._origin = new T.Vector3();
    this._normal = new T.Vector3();
    this._normalMatrix = new T.Matrix3();
  }

  async load(envId) {
    const meta = ENVIRONMENTS.find((e) => e.id === envId) || ENVIRONMENTS[0];
    const token = ++this.loadToken;

    await loadEnvironmentManifest();
    const { gltf, url } = await loadEnvironmentAsset(meta.file);

    if (token !== this.loadToken) {
      disposeObjectTree(gltf.scene);
      return null;
    }

    this.clearCurrent();

    const group = gltf.scene;
    group.name = `env-${meta.id}`;
    group.updateMatrixWorld(true);

    let groundMesh = null;
    const obstacles = [];
    const decorObjects = [];
    const steppableMeshes = [];

    group.traverse((obj) => {
      if (obj.isMesh) {
        obj.receiveShadow = true;
        if (obj.material) {
          obj.material.needsUpdate = true;
        }
      }
      const name = obj.name || '';
      if (name === 'Ground') {
        groundMesh = obj;
      } else if (name.startsWith('Obstacle_')) {
        const box = new T.Box3().setFromObject(obj, true);
        const center = new T.Vector3();
        const size = new T.Vector3();
        box.getCenter(center);
        box.getSize(size);
        const isStepStone = name.startsWith('Obstacle_StepStone_');
        const isRaisedSlab = name === 'Obstacle_Raised_Slab';
        const steppable = isStepStone || isRaisedSlab;
        if (steppable) {
          steppableMeshes.push(obj);
        }
        obstacles.push({
          name,
          object: obj,
          box,
          center,
          size,
          minX: box.min.x,
          maxX: box.max.x,
          minZ: box.min.z,
          maxZ: box.max.z,
          minY: box.min.y,
          maxY: box.max.y,
          steppable,
        });
      } else if (name.startsWith('Decor_')) {
        decorObjects.push(obj);
      }
    });

    this.root = group;
    this.currentMeta = meta;
    this.groundMesh = groundMesh;
    this.obstacles = obstacles;
    this.decorObjects = decorObjects;
    this.steppableMeshes = steppableMeshes;

    this.scene.add(group);
    this.scene.background = new T.Color(meta.bgColor);
    this.scene.fog = new T.Fog(meta.bgColor, meta.fogNear, meta.fogFar);

    return {
      meta,
      url,
      groundMesh,
      obstacles,
      decorObjects,
    };
  }

  _extractNormal(hit) {
    if (hit.face && hit.object) {
      this._normalMatrix.getNormalMatrix(hit.object.matrixWorld);
      this._normal.copy(hit.face.normal).applyMatrix3(this._normalMatrix).normalize();
      if (this._normal.y < 0) this._normal.negate();
    } else {
      this._normal.copy(UP);
    }
    return this._normal;
  }

  sampleTerrain(x, z, { includeSteppable = false } = {}) {
    const targets = [];
    if (this.groundMesh) targets.push(this.groundMesh);
    if (includeSteppable && this.steppableMeshes.length > 0) {
      targets.push(...this.steppableMeshes);
    }
    if (targets.length === 0) {
      return {
        hit: false,
        height: 0,
        normal: { x: 0, y: 1, z: 0 },
        slopeRad: 0,
        slopeDeg: 0,
        walkable: true,
        surfaceName: 'Plane',
      };
    }

    this._origin.set(x, 6.0, z);
    this._ray.set(this._origin, DOWN);
    this._ray.far = 14.0;
    const hits = this._ray.intersectObjects(targets, true);
    if (!hits.length) {
      return {
        hit: false,
        height: 0,
        normal: { x: 0, y: 1, z: 0 },
        slopeRad: 0,
        slopeDeg: 0,
        walkable: false,
        surfaceName: 'None',
      };
    }

    const topHit = hits[0];
    const normal = this._extractNormal(topHit);
    const cosTheta = Math.max(-1, Math.min(1, normal.y));
    const slopeRad = Math.acos(cosTheta);
    const slopeDeg = (slopeRad * 180) / Math.PI;
    const walkable = slopeDeg <= MAX_WALKABLE_SLOPE_DEG;

    return {
      hit: true,
      height: topHit.point.y,
      normal: { x: normal.x, y: normal.y, z: normal.z },
      slopeRad,
      slopeDeg,
      walkable,
      surfaceName: topHit.object?.name || 'Ground',
    };
  }

  pickGroundPoint(raycaster, { includeSteppable = true } = {}) {
    const targets = [];
    if (this.groundMesh) targets.push(this.groundMesh);
    if (includeSteppable && this.steppableMeshes.length > 0) {
      targets.push(...this.steppableMeshes);
    }
    if (targets.length === 0) return null;
    const hits = raycaster.intersectObjects(targets, true);
    if (!hits.length) return null;
    const hit = hits[0];
    const normal = this._extractNormal(hit).clone();
    return {
      point: hit.point.clone(),
      normal,
      objectName: hit.object?.name || 'Ground',
    };
  }

  getNearestObstacle(x, z) {
    if (!this.obstacles.length) {
      return { obstacle: null, distance: Infinity };
    }
    let nearest = null;
    let minDist = Infinity;
    for (const obs of this.obstacles) {
      const d = pointToBoxDistance2D(x, z, obs);
      if (d < minDist) {
        minDist = d;
        nearest = obs;
      }
    }
    return { obstacle: nearest, distance: minDist };
  }

  getBlockingObstacles({ allowSteppableName = null, allowAllSteppable = false } = {}) {
    return this.obstacles.filter((obs) => {
      if (allowAllSteppable && obs.steppable) return false;
      if (allowSteppableName && obs.name === allowSteppableName) return false;
      return true;
    });
  }

  findSteppableAt(x, z) {
    for (const obs of this.obstacles) {
      if (!obs.steppable) continue;
      if (x >= obs.minX && x <= obs.maxX && z >= obs.minZ && z <= obs.maxZ) {
        return obs;
      }
    }
    return null;
  }

  isPointBlocked(x, z, clearance = ROBOT_CLEARANCE, options = {}) {
    const blockers = this.getBlockingObstacles(options);
    for (const obs of blockers) {
      if (pointToBoxDistance2D(x, z, obs) < clearance) {
        return obs;
      }
    }
    return null;
  }

  adjustTargetForObstacles(x, z, limit = 1.35) {
    const targetStep = this.findSteppableAt(x, z);
    // In urban-sidewalk, Raised_Slab is walkable; in obstacle-park, StepStones are walkable when targeted
    const allowSteppableName = targetStep ? targetStep.name : null;
    const allowAllSteppable = this.currentMeta.id === 'urban-sidewalk';
    const blockers = this.getBlockingObstacles({ allowSteppableName, allowAllSteppable });

    let adjX = x;
    let adjZ = z;
    let adjusted = false;
    let blockedBy = null;

    for (let pass = 0; pass < 3; pass++) {
      let moved = false;
      for (const obs of blockers) {
        if (pointToBoxDistance2D(adjX, adjZ, obs) < ROBOT_CLEARANCE) {
          const out = pushOutOfBox2D(adjX, adjZ, obs, ROBOT_CLEARANCE + 0.02);
          if (out.pushed) {
            adjX = Math.max(-limit, Math.min(limit, out.x));
            adjZ = Math.max(-limit, Math.min(limit, out.z));
            adjusted = true;
            blockedBy = obs.name;
            moved = true;
          }
        }
      }
      if (!moved) break;
    }

    return {
      x: Math.round(adjX * 100) / 100,
      z: Math.round(adjZ * 100) / 100,
      adjusted,
      blockedBy,
      allowSteppableName,
      allowAllSteppable,
    };
  }

  isSegmentClear(x1, z1, x2, z2, clearance, options) {
    const dist = Math.hypot(x2 - x1, z2 - z1);
    const steps = Math.max(8, Math.ceil(dist / 0.06));
    for (let i = 1; i <= steps; i++) {
      const t = i / steps;
      const sx = x1 + (x2 - x1) * t;
      const sz = z1 + (z2 - z1) * t;
      if (this.isPointBlocked(sx, sz, clearance, options)) {
        return false;
      }
    }
    return true;
  }

  planWaypoints(start, targetX, targetZ, makeArcFn, limit = 1.35) {
    const targetInfo = this.adjustTargetForObstacles(targetX, targetZ, limit);
    const finalX = targetInfo.x;
    const finalZ = targetInfo.z;
    const filterOpts = {
      allowSteppableName: targetInfo.allowSteppableName,
      allowAllSteppable: targetInfo.allowAllSteppable,
    };

    const directArc = makeArcFn(start, finalX, finalZ);
    if (!directArc) return null;

    const hitObstacles = new Set();
    if (targetInfo.blockedBy) hitObstacles.add(targetInfo.blockedBy);

    // Check if direct arc collides with any blocking obstacle
    const checkSamples = 28;
    for (let i = 1; i <= checkSamples; i++) {
      const pt = directArc.at(i / checkSamples);
      const hit = this.isPointBlocked(pt.x, pt.z, ROBOT_CLEARANCE * 0.92, filterOpts);
      if (hit) hitObstacles.add(hit.name);
    }

    if (hitObstacles.size === 0 || this.obstacles.length === 0) {
      return {
        waypoints: [{ x: finalX, z: finalZ }],
        targetX: finalX,
        targetZ: finalZ,
        adjusted: targetInfo.adjusted,
        detoured: false,
        avoidedNames: Array.from(hitObstacles),
      };
    }

    // Lightweight 2D A* grid search over [-limit, limit]
    const step = 0.1;
    const halfN = Math.round(limit / step);
    const size = halfN * 2 + 1;
    const toIdx = (v) => Math.max(0, Math.min(size - 1, Math.round(v / step) + halfN));
    const toWorld = (idx) => (idx - halfN) * step;

    const sx = toIdx(start.x);
    const sz = toIdx(start.z);
    const gx = toIdx(finalX);
    const gz = toIdx(finalZ);

    const gridBlocked = new Uint8Array(size * size);
    for (let ix = 0; ix < size; ix++) {
      for (let iz = 0; iz < size; iz++) {
        if ((ix === sx && iz === sz) || (ix === gx && iz === gz)) continue;
        const wx = toWorld(ix);
        const wz = toWorld(iz);
        if (this.isPointBlocked(wx, wz, ROBOT_CLEARANCE * 0.95, filterOpts)) {
          gridBlocked[ix * size + iz] = 1;
        }
      }
    }

    const total = size * size;
    const gScore = new Float32Array(total).fill(Infinity);
    const fScore = new Float32Array(total).fill(Infinity);
    const parent = new Int32Array(total).fill(-1);
    const closed = new Uint8Array(total);
    const open = [];

    const startKey = sx * size + sz;
    const goalKey = gx * size + gz;
    gScore[startKey] = 0;
    fScore[startKey] = Math.hypot(gx - sx, gz - sz);
    open.push(startKey);

    const dirs = [
      [1, 0, 1],
      [-1, 0, 1],
      [0, 1, 1],
      [0, -1, 1],
      [1, 1, Math.SQRT2],
      [1, -1, Math.SQRT2],
      [-1, 1, Math.SQRT2],
      [-1, -1, Math.SQRT2],
    ];

    let found = false;
    while (open.length > 0) {
      let bestPos = 0;
      for (let i = 1; i < open.length; i++) {
        if (fScore[open[i]] < fScore[open[bestPos]]) bestPos = i;
      }
      const cur = open.splice(bestPos, 1)[0];
      if (cur === goalKey) {
        found = true;
        break;
      }
      closed[cur] = 1;
      const cx = Math.floor(cur / size);
      const cz = cur % size;

      for (const [dx, dz, cost] of dirs) {
        const nx = cx + dx;
        const nz = cz + dz;
        if (nx < 0 || nx >= size || nz < 0 || nz >= size) continue;
        const nKey = nx * size + nz;
        if (closed[nKey] || gridBlocked[nKey]) continue;
        // Prevent diagonal corner cutting
        if (dx !== 0 && dz !== 0) {
          if (gridBlocked[(cx + dx) * size + cz] || gridBlocked[cx * size + (cz + dz)]) {
            continue;
          }
        }
        const tentative = gScore[cur] + cost;
        if (tentative < gScore[nKey]) {
          parent[nKey] = cur;
          gScore[nKey] = tentative;
          fScore[nKey] = tentative + Math.hypot(gx - nx, gz - nz);
          if (!open.includes(nKey)) open.push(nKey);
        }
      }
    }

    if (!found) {
      return {
        waypoints: [{ x: finalX, z: finalZ }],
        targetX: finalX,
        targetZ: finalZ,
        adjusted: targetInfo.adjusted,
        detoured: false,
        avoidedNames: Array.from(hitObstacles),
      };
    }

    // Reconstruct path and prune with line-of-sight smoothing
    const raw = [];
    let trace = goalKey;
    while (trace !== -1 && trace !== startKey) {
      const tx = Math.floor(trace / size);
      const tz = trace % size;
      raw.push({ x: toWorld(tx), z: toWorld(tz) });
      trace = parent[trace];
    }
    raw.reverse();
    if (raw.length > 0) {
      raw[raw.length - 1] = { x: finalX, z: finalZ };
    }

    const full = [{ x: start.x, z: start.z }, ...raw];
    const pruned = [];
    let anchor = 0;
    while (anchor < full.length - 1) {
      let furthest = anchor + 1;
      for (let j = full.length - 1; j > anchor + 1; j--) {
        if (
          this.isSegmentClear(
            full[anchor].x,
            full[anchor].z,
            full[j].x,
            full[j].z,
            ROBOT_CLEARANCE * 0.9,
            filterOpts,
          )
        ) {
          furthest = j;
          break;
        }
      }
      pruned.push(full[furthest]);
      anchor = furthest;
    }

    return {
      waypoints: pruned.length ? pruned : [{ x: finalX, z: finalZ }],
      targetX: finalX,
      targetZ: finalZ,
      adjusted: targetInfo.adjusted,
      detoured: pruned.length > 1 || targetInfo.adjusted,
      avoidedNames: Array.from(hitObstacles),
    };
  }

  clearCurrent() {
    if (this.root) {
      this.scene.remove(this.root);
      disposeObjectTree(this.root);
      this.root = null;
    }
    this.groundMesh = null;
    this.obstacles = [];
    this.decorObjects = [];
    this.steppableMeshes = [];
  }
}
