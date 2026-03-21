/**
 * main.js – PureCV WASM butterfly demo
 *
 * Loads butterfly.jpg, feeds pixel data through purecv WASM, and renders
 * each processing result on its own <canvas> card.
 *
 * API surface (from crates/wasm/src/lib.rs):
 *   PureCvMatrixU8.fromData(rows, cols, ch, Uint8Array)
 *   PureCvMatrixF32.fromData(rows, cols, ch, Float32Array)
 *   convertU8ToF32(matU8)  → PureCvMatrixF32
 *   cvtColor(matU8, code)  → PureCvMatrixU8   (takes integer constant)
 *   blur(matF32, kw, kh, border)
 *   gaussianBlur(matF32, kw, kh, σ1, σ2, border)
 *   canny(matF32, lo, hi, aperture, l2)        → PureCvMatrixU8
 *   sobel(matF32, dx, dy, k, scale, delta, border) → PureCvMatrixF32
 *   laplacian(matF32, k, scale, delta, border)     → PureCvMatrixF32
 *   threshold(matF32, thresh, max, type)       → ThresholdResult
 *   ThresholdResult.threshVal / .getMatrix()
 *
 *   Constants (functions returning i32):
 *   COLOR_RGB2GRAY()  = 1
 *   BORDER_REFLECT_101() = 4
 *   THRESH_BINARY()   = 0
 */

// ────────────────────────────────────────────────────────────────────────────
// Logging helpers
// ────────────────────────────────────────────────────────────────────────────

const logEl = document.getElementById('log');
const statusEl = document.getElementById('status');

function log(msg, cls = '') {
    const p = document.createElement('p');
    if (cls) p.className = cls;
    p.textContent = `[${new Date().toLocaleTimeString()}] ${msg}`;
    logEl.appendChild(p);
    logEl.scrollTop = logEl.scrollHeight;
}

function setStatus(msg, cls = '') {
    statusEl.textContent = msg;
    statusEl.className = cls;
}

// ────────────────────────────────────────────────────────────────────────────
// Canvas rendering
// ────────────────────────────────────────────────────────────────────────────

/** Render a 1-channel u8 array to a canvas as greyscale RGBA. */
function drawGray(canvasId, u8, w, h) {
    const canvas = document.getElementById(canvasId);
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
        const v = u8[i];
        img.data[i * 4] = v;
        img.data[i * 4 + 1] = v;
        img.data[i * 4 + 2] = v;
        img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
}

/** Render a 3-channel u8 RGB array to a canvas. */
function drawRGB(canvasId, u8, w, h) {
    const canvas = document.getElementById(canvasId);
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    const img = ctx.createImageData(w, h);
    for (let i = 0; i < w * h; i++) {
        img.data[i * 4] = u8[i * 3];
        img.data[i * 4 + 1] = u8[i * 3 + 1];
        img.data[i * 4 + 2] = u8[i * 3 + 2];
        img.data[i * 4 + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
}

/** Render an ImageData (RGBA from canvas) directly. */
function drawImageData(canvasId, imageData) {
    const canvas = document.getElementById(canvasId);
    canvas.width = imageData.width;
    canvas.height = imageData.height;
    canvas.getContext('2d').putImageData(imageData, 0, 0);
}

/** Normalize a Float32Array (abs values) to a Uint8Array [0, 255]. */
function normalizeF32(f32) {
    let min = Infinity, max = -Infinity;
    for (const v of f32) {
        const a = Math.abs(v);
        if (a < min) min = a;
        if (a > max) max = a;
    }
    const range = max === min ? 1 : max - min;
    const out = new Uint8Array(f32.length);
    for (let i = 0; i < f32.length; i++) {
        out[i] = Math.round(((Math.abs(f32[i]) - min) / range) * 255);
    }
    return out;
}

// ────────────────────────────────────────────────────────────────────────────
// WASM bootstrap
// ────────────────────────────────────────────────────────────────────────────

async function loadWasm() {
    const module = await import(`../pkg/dist-std/purecv_wasm.js`);
    await module.default();       // initialise wasm memory
    module.init_purecv();
    try { module.init_panic_hook(); } catch (_) { /* already set */ }
    log(`purecv v${module.get_version()} ready`, 'ok');
    return module;
}

// ────────────────────────────────────────────────────────────────────────────
// Image loader (uses a hidden canvas to read pixel data)
// ────────────────────────────────────────────────────────────────────────────

function loadImagePixels(src) {
    return new Promise((resolve, reject) => {
        const img = new Image();
        img.crossOrigin = 'anonymous';
        img.onload = () => {
            const offscreen = document.createElement('canvas');
            offscreen.width = img.naturalWidth;
            offscreen.height = img.naturalHeight;
            const ctx = offscreen.getContext('2d');
            ctx.drawImage(img, 0, 0);
            const rgba = ctx.getImageData(0, 0, img.naturalWidth, img.naturalHeight);
            resolve({ rgba, width: img.naturalWidth, height: img.naturalHeight });
        };
        img.onerror = () => reject(new Error(`Cannot load ${src}`));
        img.src = src;
    });
}

// ────────────────────────────────────────────────────────────────────────────
// Main pipeline
// ────────────────────────────────────────────────────────────────────────────

async function run() {
    try {
        const m = await loadWasm();
        setStatus('⏳ Loading image…');

        // ── Load butterfly.jpg ────────────────────────────────────────────────
        const imgSrc = '../../../examples/data/butterfly.jpg';
        log(`Loading ${imgSrc}…`, 'info');
        const { rgba, width, height } = await loadImagePixels(imgSrc);
        log(`Loaded: ${width}×${height}`);

        // Show original RGBA frame
        drawImageData('c-original', rgba);

        // Strip alpha → packed RGB u8
        const rgbU8 = new Uint8Array(width * height * 3);
        for (let i = 0; i < width * height; i++) {
            rgbU8[i * 3] = rgba.data[i * 4];
            rgbU8[i * 3 + 1] = rgba.data[i * 4 + 1];
            rgbU8[i * 3 + 2] = rgba.data[i * 4 + 2];
        }

        // Wrap in purecv matrix
        const matRgbU8 = m.PureCvMatrixU8.fromData(height, width, 3, rgbU8);

        // ── Greyscale (u8) ───────────────────────────────────────────────────
        log('cvtColor RGB→Gray…', 'info');
        const matGrayU8 = m.cvtColor(matRgbU8, m.COLOR_RGB2GRAY());
        drawGray('c-gray', matGrayU8.data(), width, height);
        log('Greyscale ✓');

        // Convert grey to f32 for filter/edge ops
        const matGrayF32 = m.convertU8ToF32(matGrayU8);
        // Convert colour to f32 for blur ops (blur* only accept f32)
        const matRgbF32 = m.convertU8ToF32(matRgbU8);

        const BORDER = m.BORDER_REFLECT_101();

        // ── Blur 5×5 ─────────────────────────────────────────────────────────
        log('blur 5×5…', 'info');
        const matBlurF32 = m.blur(matRgbF32, 5, 5, BORDER);
        const blurU8 = m.convertF32ToU8(matBlurF32);
        drawRGB('c-blur', blurU8.data(), width, height);
        log('Blur ✓');

        // ── Gaussian Blur 5×5 σ=1.75   ──────────────────────────────────────────
        log('gaussianBlur 5×5 σ=1.75…', 'info');
        const matGaussF32 = m.gaussianBlur(matRgbF32, 5, 5, 1.75, 1.75, BORDER);
        const gaussU8 = m.convertF32ToU8(matGaussF32);
        drawRGB('c-gauss', gaussU8.data(), width, height);
        log('Gaussian blur ✓');

        // ── Canny ─────────────────────────────────────────────────────────────
        log('canny lo=50 hi=150…', 'info');
        const matCanny = m.canny(matGrayF32, 50, 150, 3, false);
        drawGray('c-canny', matCanny.data(), width, height);
        log('Canny ✓');

        // ── Sobel X ───────────────────────────────────────────────────────────
        log('sobel dx=1 dy=0 k=3…', 'info');
        const matSobelF32 = m.sobel(matGrayF32, 1, 0, 3, 1.0, 0.0, BORDER);
        drawGray('c-sobel', normalizeF32(matSobelF32.data()), width, height);
        log('Sobel ✓');

        // ── Laplacian ─────────────────────────────────────────────────────────
        log('laplacian k=3…', 'info');
        const matLapF32 = m.laplacian(matGrayF32, 3, 1.0, 0.0, BORDER);
        drawGray('c-laplacian', normalizeF32(matLapF32.data()), width, height);
        log('Laplacian ✓');

        // ── Threshold binary 127 ──────────────────────────────────────────────
        log('threshold binary thresh=127…', 'info');
        const threshResult = m.threshold(matGrayF32, 127.0, 255.0, m.THRESH_BINARY());
        const threshVal = threshResult.threshVal;  // read BEFORE getMatrix() consumes the object
        const threshMat = threshResult.getMatrix(); // moves self — threshResult is now invalid
        drawGray('c-thresh', normalizeF32(threshMat.data()), width, height);
        log(`Threshold ✓  (val=${threshVal})`);

        // Cleanup
        [matRgbU8, matGrayU8, matGrayF32, matRgbF32,
            matBlurF32, blurU8,
            matGaussF32, gaussU8,
            matCanny, matSobelF32, matLapF32, threshMat,
        ].forEach(obj => { try { obj.free(); } catch (_) { } });

        setStatus('✅ All filters applied', 'ok');
        log('Done! ✨', 'ok');

    } catch (err) {
        setStatus('❌ Error', 'err');
        log(`ERROR: ${err.message}`, 'err');
        console.error(err);
    }
}

run();