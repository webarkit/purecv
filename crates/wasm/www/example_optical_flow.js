/*
 *  example_optical_flow.js
 *  purecv
 *
 *  This file is part of purecv - WebARKit.
 *
 *  purecv is free software: you can redistribute it and/or modify
 *  it under the terms of the GNU Lesser General Public License as published by
 *  the Free Software Foundation, either version 3 of the License, or
 *  (at your option) any later version.
 *
 *  purecv is distributed in the hope that it will be useful,
 *  but WITHOUT ANY WARRANTY; without even the implied warranty of
 *  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 *  GNU Lesser General Public License for more details.
 *
 *  You should have received a copy of the GNU Lesser General Public License
 *  along with purecv.  If not, see <http://www.gnu.org/licenses/>.
 *
 *  As a special exception, the copyright holders of this library give you
 *  permission to link this library with independent modules to produce an
 *  executable, regardless of the license terms of these independent modules, and to
 *  copy and distribute the resulting executable under terms of your choice,
 *  provided that you also meet, for each linked independent module, the terms and
 *  conditions of the license of that module. An independent module is a module
 *  which is neither derived from nor based on this library. If you modify this
 *  library, you may extend this exception to your version of the library, but you
 *  are not obligated to do so. If you do not wish to do so, delete this exception
 *  statement from your version.
 *
 *  Copyright 2026 WebARKit.
 *
 *  Author(s): Walter Perdan @kalwalt https://github.com/kalwalt
 *
 */

import { initWasm } from './cv_demo_utils.js';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

let cv = null;
let stream = null;
let animId = null;

// Grayscale Mat of the previous frame.
let prevGray = null;

// Current tracked point coordinates as Float32Array [x,y,x,y,...]
let trackedPts = null;

// Track trail history: array of Float32Array snapshots for visualisation.
const trailHistory = [];
const MAX_TRAIL_LEN = 8;

// Minimum number of tracked points before re-detecting.
const MIN_POINTS = 15;

// DOM elements
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', { willReadFrequently: true });
const video = document.getElementById('video');

const sliders = {
    maxCorners: document.getElementById('max-corners'),
    quality: document.getElementById('quality'),
    distance: document.getElementById('distance'),
    levels: document.getElementById('levels'),
};
const displays = {
    maxCorners: document.getElementById('val-max-corners'),
    quality: document.getElementById('val-quality'),
    distance: document.getElementById('val-distance'),
    levels: document.getElementById('val-levels'),
};

// ---------------------------------------------------------------------------
// Webcam helpers
// ---------------------------------------------------------------------------

async function startWebcam() {
    stream = await navigator.mediaDevices.getUserMedia({
        video: { facingMode: 'environment', width: { ideal: 640 }, height: { ideal: 480 } }
    });
    video.srcObject = stream;
    video.classList.remove('hidden');
    await video.play();

    canvas.width = video.videoWidth;
    canvas.height = video.videoHeight;

    document.getElementById('btn-webcam').classList.add('hidden');
    document.getElementById('btn-stop').classList.remove('hidden');

    // Reset tracking state.
    prevGray = null;
    trackedPts = null;
    trailHistory.length = 0;

    requestAnimationFrame(loop);
}

function stopWebcam() {
    if (animId) cancelAnimationFrame(animId);
    animId = null;
    if (stream) { stream.getTracks().forEach(t => t.stop()); stream = null; }
    video.classList.add('hidden');
    document.getElementById('btn-webcam').classList.remove('hidden');
    document.getElementById('btn-stop').classList.add('hidden');

    if (prevGray) { prevGray.free(); prevGray = null; }
    trackedPts = null;
    trailHistory.length = 0;
}

// ---------------------------------------------------------------------------
// Core processing loop
// ---------------------------------------------------------------------------

let lastTs = 0;
let fpsSmooth = 0;

function loop(ts) {
    animId = requestAnimationFrame(loop);

    // FPS calculation.
    const dt = ts - lastTs;
    lastTs = ts;
    if (dt > 0) {
        const instantFps = 1000 / dt;
        fpsSmooth = fpsSmooth * 0.9 + instantFps * 0.1;
        document.getElementById('stat-fps').textContent = fpsSmooth.toFixed(0);
    }

    // Draw current video frame onto the canvas.
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Build a u8 grayscale Mat from the canvas.
    const imageData = ctx.getImageData(0, 0, canvas.width, canvas.height);
    const rgbaMat = cv.Mat.fromU8Data(canvas.height, canvas.width, 4, imageData.data);
    const currGray = cv.cvtColor(rgbaMat, cv.COLOR_RGBA2GRAY());
    rgbaMat.free();

    if (!prevGray) {
        // First frame — detect features, swap, and continue.
        prevGray = currGray;
        trackedPts = detectFeatures(prevGray);
        return;
    }

    // Read parameters.
    const maxLevel = parseInt(sliders.levels.value);

    // -------------------------------------------------------------------
    // Track points from prevGray → currGray.
    // -------------------------------------------------------------------
    let nTracked = 0;
    let nLost = 0;

    if (trackedPts && trackedPts.length >= 2) {
        const result = cv.calcOpticalFlowPyrLK(
            prevGray,
            currGray,
            trackedPts,
            /* win_w */ 15, /* win_h */ 15,
            maxLevel,
            /* max_count */ 30,
            /* epsilon */   0.01,
            cv.OPTFLOW_LK_GET_MIN_EIGENVALS(),
            /* min_eigen */ 1e-4,
        );

        // Filter: keep only tracked points (status == 1).
        const nextPts = result.nextPts;
        const status = result.status;
        const nPts = status.length;

        const keptPts = [];
        for (let i = 0; i < nPts; i++) {
            if (status[i] === 1) {
                keptPts.push(nextPts[i * 2], nextPts[i * 2 + 1]);
                nTracked++;
            } else {
                nLost++;
            }
        }
        trackedPts = new Float32Array(keptPts);

        // Save trail snapshot.
        trailHistory.push(new Float32Array(trackedPts));
        if (trailHistory.length > MAX_TRAIL_LEN) trailHistory.shift();
    }

    // -------------------------------------------------------------------
    // Re-detect if we lost too many.
    // -------------------------------------------------------------------
    if (!trackedPts || trackedPts.length / 2 < MIN_POINTS) {
        trackedPts = detectFeatures(currGray);
        trailHistory.length = 0;
    }

    // -------------------------------------------------------------------
    // Visualise.
    // -------------------------------------------------------------------

    // Draw trail segments (fade older segments).
    for (let t = 1; t < trailHistory.length; t++) {
        const prev = trailHistory[t - 1];
        const curr = trailHistory[t];
        const n = Math.min(prev.length, curr.length) / 2;
        const alpha = (t / trailHistory.length) * 0.8 + 0.2;
        ctx.globalAlpha = alpha;
        ctx.strokeStyle = `hsl(${(t * 40) % 360}, 100%, 60%)`;
        ctx.lineWidth = 2;
        for (let i = 0; i < n; i++) {
            ctx.beginPath();
            ctx.moveTo(prev[i * 2], prev[i * 2 + 1]);
            ctx.lineTo(curr[i * 2], curr[i * 2 + 1]);
            ctx.stroke();
        }
    }
    ctx.globalAlpha = 1.0;

    // Draw current feature points.
    if (trackedPts) {
        ctx.fillStyle = '#00FF00';
        for (let i = 0; i < trackedPts.length; i += 2) {
            ctx.beginPath();
            ctx.arc(trackedPts[i], trackedPts[i + 1], 3, 0, 2 * Math.PI);
            ctx.fill();
        }
    }

    // Update stats.
    const totalPts = trackedPts ? trackedPts.length / 2 : 0;
    document.getElementById('stat-tracked').textContent = nTracked;
    document.getElementById('stat-lost').textContent = nLost;
    document.getElementById('stat-total').textContent = totalPts;

    // Swap frames.
    prevGray.free();
    prevGray = currGray;
}

// ---------------------------------------------------------------------------
// Feature detection helper
// ---------------------------------------------------------------------------

function detectFeatures(grayU8) {
    const maxCorners = parseInt(sliders.maxCorners.value);
    const qualityLevel = parseFloat(sliders.quality.value);
    const minDistance = parseFloat(sliders.distance.value);

    // goodFeaturesToTrack needs f32 input.
    const grayF32 = grayU8.convertTo('f32');
    const pts = cv.goodFeaturesToTrack(grayF32, maxCorners, qualityLevel, minDistance, 3, false, 0.04);
    grayF32.free();
    return pts;  // Float32Array [x, y, x, y, ...]
}

// ---------------------------------------------------------------------------
// Event listeners
// ---------------------------------------------------------------------------

Object.keys(sliders).forEach(key => {
    sliders[key].addEventListener('input', () => {
        displays[key].innerText = sliders[key].value;
    });
});

document.getElementById('btn-webcam').onclick = async () => {
    try { await startWebcam(); } catch (e) {
        console.error('Webcam error:', e);
        alert('Could not access webcam: ' + e.message);
    }
};
document.getElementById('btn-stop').onclick = stopWebcam;

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function start() {
    try {
        cv = await initWasm();
        document.getElementById('loader').classList.add('hidden');
    } catch (err) {
        console.error("WASM Initialization failed:", err);
        document.getElementById('loader').innerHTML = `<p style="color:red">Error loading WASM: ${err.message}</p>`;
    }
}

start();
