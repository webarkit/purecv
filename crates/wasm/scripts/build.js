// Multi-build script for purecv WASM (std, simd, parallel, simd+parallel)
// Output: pkg/dist-std, pkg/dist-simd, pkg/dist-parallel, pkg/dist-simd-parallel
// Typings are copied in each dist

const { execSync } = require('child_process');
const fs = require('fs');
const path = require('path');

const ROOT = path.resolve(__dirname, '..');
const PKG = path.join(ROOT, 'pkg');

const builds = [
  { name: 'dist-std', features: '' },
  { name: 'dist-simd', features: '--features simd' },
  { name: 'dist-parallel', features: '--features parallel' },
  { name: 'dist-simd-parallel', features: '--features "simd parallel"' },
];

function cleanDist(name) {
  const dist = path.join(PKG, name);
  if (fs.existsSync(dist)) fs.rmSync(dist, { recursive: true, force: true });
}

function moveBuild(name) {
  const dist = path.join(PKG, name);
  fs.mkdirSync(dist, { recursive: true });
  // Move all generated files except .gitkeep
  fs.readdirSync(PKG).forEach(f => {
    if (f === name || f === '.gitkeep') return;
    const src = path.join(PKG, f);
    const dest = path.join(dist, f);
    if (fs.lstatSync(src).isFile()) fs.renameSync(src, dest);
  });
}

function buildAll() {
  builds.forEach(({ name, features }) => {
    console.log(`\n=== Building ${name} ===`);
    cleanDist(name);
    execSync(`wasm-pack build --release --target bundler --scope webarkit ${features}`, { cwd: ROOT, stdio: 'inherit' });
    moveBuild(name);
  });
}

buildAll();
