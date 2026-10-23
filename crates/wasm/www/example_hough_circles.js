/*
 *  example_hough_circles.js
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

import { initWasm, loadImage, getScaledDimensions, canvasToMat, drawCircles } from './cv_demo_utils.js';

let sourceImage = null;
let cv = null;
const canvas = document.getElementById('canvas');
const ctx = canvas.getContext('2d', "willReadFrequently: true");

const sliders = {
    minDist: document.getElementById('min-dist'),
    p1: document.getElementById('p1'),
    p2: document.getElementById('p2'),
    minR: document.getElementById('min-r'),
    maxR: document.getElementById('max-r')
};

async function start() {
    try {
        cv = await initWasm();
        document.getElementById('loader').classList.add('hidden');

        sourceImage = await loadImage('https://raw.githubusercontent.com/opencv/opencv/4.10.0/samples/data/smarties.png');
        processImage();
    } catch (err) {
        console.error("WASM Initialization failed:", err);
        document.getElementById('loader').innerHTML = `<p style="color:red">Error loading WASM: ${err.message}</p>`;
    }
}

function processImage() {
    if (!sourceImage || !cv) return;

    const { width, height } = getScaledDimensions(sourceImage, 800);
    canvas.width = width;
    canvas.height = height;
    ctx.drawImage(sourceImage, 0, 0, width, height);

    const mat = canvasToMat(cv, canvas, ctx);
    
    // 1. Gray (RGBA u8 -> Gray u8)
    const gray = cv.cvtColor(mat, cv.COLOR_RGBA2GRAY()); 
    
    // 2. Conversion to f32 for medianBlur
    const grayF32 = gray.convertTo("f32");
    
    // 3. Blur (Median blur is good for circles)
    const blurredF32 = cv.medianBlur(grayF32, 5); 
    
    // 4. Conversion back to u8 for houghCircles
    const blurredU8 = blurredF32.convertTo("u8");

    // 5. Hough Circles
    const dp = 1.2;
    const minDist = parseFloat(sliders.minDist.value);
    const p1 = parseFloat(sliders.p1.value);
    const p2 = parseFloat(sliders.p2.value);
    const minR = parseInt(sliders.minR.value);
    const maxR = parseInt(sliders.maxR.value);

    const circles = cv.houghCircles(blurredU8, dp, minDist, p1, p2, minR, maxR);

    drawCircles(ctx, circles, '#FF3E3E', 3);

    mat.free();
    gray.free();
    grayF32.free();
    blurredF32.free();
    blurredU8.free();
}

// UI Listeners
Object.values(sliders).forEach(s => {
    s.oninput = () => {
        const display = document.getElementById('val-' + s.id);
        if (display) display.innerText = s.value;
        processImage();
    };
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
