import { cpSync, existsSync, mkdirSync, rmSync, statSync } from 'node:fs';
import path from 'node:path';
import { execFileSync } from 'node:child_process';
import { fileURLToPath } from 'node:url';

const rootDir = path.resolve(path.dirname(fileURLToPath(import.meta.url)), '..');
const distDir = path.join(rootDir, 'dist');

const jsFilesToCheck = [
  'server.js',
  'studio.js',
  'motion.js',
  'environment.js',
  'critic.js',
  'scripts/build.mjs',
];

for (const relPath of jsFilesToCheck) {
  execFileSync(process.execPath, ['--check', path.join(rootDir, relPath)], {
    stdio: 'inherit',
  });
}

rmSync(distDir, { recursive: true, force: true });
mkdirSync(distDir, { recursive: true });

const entriesToCopy = [
  '.nojekyll',
  'index.html',
  'studio.css',
  'studio.js',
  'motion.js',
  'environment.js',
  'critic.js',
  'environment-manifest.json',
  'assets',
  'environments',
  'vendor',
];

for (const entry of entriesToCopy) {
  const src = path.join(rootDir, entry);
  const dest = path.join(distDir, entry);
  if (!existsSync(src)) {
    throw new Error(`Missing required static entry: ${entry}`);
  }
  cpSync(src, dest, { recursive: true });
}

const requiredBuildArtifacts = [
  'index.html',
  '.nojekyll',
  'studio.css',
  'studio.js',
  'motion.js',
  'environment.js',
  'critic.js',
  'environment-manifest.json',
  'assets/Roadrunner-IROS.glb',
  'vendor/three.module.min.js',
  'vendor/three.core.min.js',
  'vendor/addons/loaders/GLTFLoader.js',
  'vendor/addons/controls/OrbitControls.js',
  'environments/environments/flat-ground.glb',
  'environments/environments/hilly-terrain.glb',
  'environments/environments/obstacle-park.glb',
  'environments/environments/sandy-beach.glb',
  'environments/environments/urban-sidewalk.glb',
];

for (const relPath of requiredBuildArtifacts) {
  const fullPath = path.join(distDir, relPath);
  if (!existsSync(fullPath) || (relPath !== '.nojekyll' && statSync(fullPath).size === 0)) {
    throw new Error(`Production build verification failed for: dist/${relPath}`);
  }
}

console.log(`Production build verified in dist/ (${requiredBuildArtifacts.length} required artifacts present).`);
