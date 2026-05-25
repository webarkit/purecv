/*
 *  example_orb.js
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

import { initWasm, loadImage, getScaledDimensions, canvasToMat } from './cv_demo_utils.js';

let sourceImage = null;
let cv = null;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', "willReadFrequently: true");

// UI Elements
const sliders = {
    features: document.getElementById('features'),
    scale: document.getElementById('scale'),
    levels: document.getElementById('levels'),
    fastThreshold: document.getElementById('fast-threshold'),
    scoreType: document.getElementById('score-type')
};

const displays = {
    features: document.getElementById('val-features'),
    scale: document.getElementById('val-scale'),
    levels: document.getElementById('val-levels'),
    fastThreshold: document.getElementById('val-fast-threshold'),
    width: document.getElementById('metric-width'),
    height: document.getElementById('metric-height'),
    count: document.getElementById('metric-count'),
    time: document.getElementById('metric-time'),
    render: document.getElementById('metric-render')
};

// File Uploader Setup
const dropZone = document.getElementById('drop-zone');
const fileInput = document.getElementById('file-input');

dropZone.addEventListener('click', () => fileInput.click());
fileInput.addEventListener('change', handleFile);
dropZone.addEventListener('dragover', (e) => { e.preventDefault(); dropZone.style.borderColor = '#00f2fe'; });
dropZone.addEventListener('dragleave', () => { dropZone.style.borderColor = '#334155'; });
dropZone.addEventListener('drop', (e) => {
    e.preventDefault();
    dropZone.style.borderColor = '#334155';
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        handleFile();
    }
});

function handleFile() {
    const file = fileInput.files[0];
    if (file) {
        const reader = new FileReader();
        reader.onload = async (e) => {
            sourceImage = await loadImage(e.target.result);
            processImage();
        };
        reader.readAsDataURL(file);
    }
}

// Controller Listeners
sliders.features.addEventListener('input', () => { displays.features.textContent = sliders.features.value; processImage(); });
sliders.scale.addEventListener('input', () => { displays.scale.textContent = parseFloat(sliders.scale.value).toFixed(2); processImage(); });
sliders.levels.addEventListener('input', () => { displays.levels.textContent = sliders.levels.value; processImage(); });
sliders.fastThreshold.addEventListener('input', () => { displays.fastThreshold.textContent = sliders.fastThreshold.value; processImage(); });
sliders.scoreType.addEventListener('change', processImage);

async function start() {
    try {
        cv = await initWasm();
        document.getElementById('loader').classList.add('hidden');

        // Initial default image (OpenCV butterfly sample)
        sourceImage = await loadImage('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/butterfly.jpg');
        processImage();
    } catch (err) {
        console.error("WASM Initialization failed:", err);
        document.getElementById('loader').innerHTML = `<p style="color:red">Error loading WASM: ${err.message}</p>`;
    }
}

function processImage() {
    if (!sourceImage || !cv) return;

    const { width, height } = getScaledDimensions(sourceImage, 1024);
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(sourceImage, 0, 0, width, height);

    // Update source image dimensions in the UI
    displays.width.textContent = sourceImage.width;
    displays.height.textContent = sourceImage.height;

    // 1. Convert canvas to Mat
    const mat = canvasToMat(cv, canvas, ctx);

    // 2. Convert to Grayscale
    const gray = cv.cvtColor(mat, cv.COLOR_RGBA2GRAY());

    // 3. Extract parameter values
    const nfeatures = parseInt(sliders.features.value);
    const scaleFactor = parseFloat(sliders.scale.value);
    const nlevels = parseInt(sliders.levels.value);
    const edgeThreshold = 31;
    const firstLevel = 0;
    const wtaK = 2;
    const scoreType = parseInt(sliders.scoreType.value);
    const patchSize = 31;
    const fastThreshold = parseInt(sliders.fastThreshold.value);

    // 4. Initialize ORB Pipeline & Run
    const t0 = performance.now();
    
    // ORB(nfeatures, scale_factor, nlevels, edge_threshold, first_level, wta_k, score_type_val, patch_size, fast_threshold)
    const orb = new cv.ORB(
        nfeatures,
        scaleFactor,
        nlevels,
        edgeThreshold,
        firstLevel,
        wtaK,
        scoreType,
        patchSize,
        fastThreshold
    );

    const result = orb.detectAndCompute(gray);
    const t1 = performance.now();

    // 5. Draw Oriented Keypoints
    const r0 = performance.now();
    const keypoints = result.keypoints;
    const descriptors = result.descriptors;
    const count = keypoints.size();
    displays.count.textContent = count;
    
    drawOrbKeypoints(ctx, keypoints, '#ff0055', 6);
    const r1 = performance.now();

    // Update metrics displays
    displays.time.textContent = (t1 - t0).toFixed(2);
    displays.render.textContent = (r1 - r0).toFixed(2);

    // 6. Memory Cleanup
    mat.free();
    gray.free();
    orb.free();
    keypoints.free();
    descriptors.free();
}

function drawOrbKeypoints(ctx, kpts, color = '#ff0055', radius = 6) {
    const size = kpts.size();
    for (let i = 0; i < size; i++) {
        const kp = kpts.get(i);
        if (kp) {
            // Draw scale circle in rose/pink
            ctx.strokeStyle = color;
            ctx.lineWidth = 1.5;
            ctx.beginPath();
            ctx.arc(kp.x, kp.y, radius, 0, 2 * Math.PI);
            ctx.stroke();

            // Draw orientation line in premium sky blue
            if (kp.angle >= 0) {
                const angleRad = kp.angle * Math.PI / 180;
                ctx.strokeStyle = '#38bdf8';
                ctx.lineWidth = 2.0;
                ctx.beginPath();
                ctx.moveTo(kp.x, kp.y);
                ctx.lineTo(
                    kp.x + radius * Math.cos(angleRad),
                    kp.y + radius * Math.sin(angleRad)
                );
                ctx.stroke();
            }
        }
    }
}

start();
