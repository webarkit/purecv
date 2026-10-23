/*
 *  example_corners.js
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

import { initWasm, loadImage, getScaledDimensions, canvasToMat, drawPoints } from './cv_demo_utils.js';

let sourceImage = null;
let cv = null;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', "willReadFrequently: true");

// UI Elements
const sliders = {
    maxCorners: document.getElementById('max-corners'),
    quality: document.getElementById('quality'),
    distance: document.getElementById('distance')
};
const displays = {
    maxCorners: document.getElementById('val-max-corners'),
    quality: document.getElementById('val-quality'),
    distance: document.getElementById('val-distance')
};

async function start() {
    try {
        cv = await initWasm();
        document.getElementById('loader').classList.add('hidden');

        // Initial default image
        sourceImage = await loadImage('https://raw.githubusercontent.com/opencv/opencv/4.10.0/samples/data/butterfly.jpg');
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

    // 1. Convert canvas to Mat
    const mat = canvasToMat(cv, canvas, ctx);

    // 2. Convert to Grayscale (required for many imgproc ops)
    const gray = cv.cvtColor(mat, cv.COLOR_RGBA2GRAY()); 
    const grayF32 = gray.convertTo("f32");

    // 3. Detect corners
    const maxCorners = parseInt(sliders.maxCorners.value);
    const qualityLevel = parseFloat(sliders.quality.value);
    const minDistance = parseFloat(sliders.distance.value);

    // goodFeaturesToTrack(src, maxCorners, qualityLevel, minDistance, blockSize, useHarris, harrisK)
    const corners = cv.goodFeaturesToTrack(grayF32, maxCorners, qualityLevel, minDistance, 3, false, 0.04);

    // 4. Visualization
    drawPoints(ctx, corners, '#00FF00', 4);

    // Cleanup
    mat.free();
    gray.free();
    grayF32.free();
}

// Event Listeners
Object.keys(sliders).forEach(key => {
    sliders[key].addEventListener('input', () => {
        displays[key].innerText = sliders[key].value;
        processImage();
    });
});

const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');

dropZone.onclick = () => fileInput.click();
fileInput.onchange = async (e) => {
    if (e.target.files.length > 0) {
        const url = URL.createObjectURL(e.target.files[0]);
        sourceImage = await loadImage(url);
        processImage();
    }
};

start();
