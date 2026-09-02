/*
 *  example_histogram.js
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

import { initWasm, loadImage, getScaledDimensions, canvasToMat, matToCanvas } from './cv_demo_utils.js';

let sourceImage = null;
let cv = null;
const compareRow = document.getElementById('compare-row');
const histCanvas = document.getElementById('histogram-canvas');
const scoresEl = document.getElementById('scores');
const clipSlider = document.getElementById('clip-limit');
const clipDisplay = document.getElementById('val-clip');
const tileSlider = document.getElementById('tile-grid');
const tilesDisplay = document.getElementById('val-tiles');
const tilesDisplay2 = document.getElementById('val-tiles-2');

const COMPARE_METHODS = [
    { name: 'Correl', ctor: (cv) => cv.HIST_CMP_CORREL() },
    { name: 'ChiSqr', ctor: (cv) => cv.HIST_CMP_CHISQR() },
    { name: 'ChiSqrAlt', ctor: (cv) => cv.HIST_CMP_CHISQR_ALT() },
    { name: 'Intersect', ctor: (cv) => cv.HIST_CMP_INTERSECT() },
    { name: 'Bhattacharyya', ctor: (cv) => cv.HIST_CMP_BHATTACHARYYA() },
    { name: 'KL Divergence', ctor: (cv) => cv.HIST_CMP_KL_DIV() },
];

async function start() {
    try {
        cv = await initWasm();
        document.getElementById('loader').classList.add('hidden');

        sourceImage = await loadImage('https://raw.githubusercontent.com/opencv/opencv/master/samples/data/butterfly.jpg');
        processImage();
    } catch (err) {
        console.error("WASM Initialization failed:", err);
        document.getElementById('loader').innerHTML = `<p style="color:red">Error loading WASM: ${err.message}</p>`;
    }
}

function addCanvasBox(label) {
    const box = document.createElement('div');
    box.className = 'level-box';

    const canvas = document.createElement('canvas');
    const text = document.createElement('span');
    text.className = 'level-label';
    text.innerText = label;

    box.appendChild(canvas);
    box.appendChild(text);
    compareRow.appendChild(box);
    return canvas;
}

function drawHistogram(canvas, histData, color) {
    const w = canvas.clientWidth || 512;
    const h = 160;
    canvas.width = w;
    canvas.height = h;
    const ctx = canvas.getContext('2d');
    ctx.clearRect(0, 0, w, h);

    const max = Math.max(...histData, 1);
    const binWidth = w / histData.length;

    ctx.fillStyle = color;
    for (let i = 0; i < histData.length; i++) {
        const barHeight = (histData[i] / max) * (h - 4);
        ctx.fillRect(i * binWidth, h - barHeight, Math.max(binWidth, 1), barHeight);
    }
}

function processImage() {
    if (!sourceImage || !cv) return;

    compareRow.innerHTML = '';
    scoresEl.innerHTML = '';

    const { width, height } = getScaledDimensions(sourceImage, 512);
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d', "willReadFrequently: true");
    tempCtx.drawImage(sourceImage, 0, 0, width, height);

    const rgbaMat = canvasToMat(cv, tempCanvas, tempCtx);
    const gray = cv.cvtColor(rgbaMat, cv.COLOR_RGBA2GRAY());
    rgbaMat.free();

    const clipLimit = parseFloat(clipSlider.value);
    const tileGrid = parseInt(tileSlider.value);

    // Declared outside the try so a mid-way failure can still free whatever
    // was already allocated (WASM objects are not garbage-collected).
    let images = null;
    let hist = null;
    let equalized = null;
    let clahe = null;
    let claheOut = null;
    let claheImages = null;
    let claheHist = null;

    try {
        // --- calc_hist: 256-bin uniform histogram of the grayscale source ---
        images = new cv.MatVector();
        images.push(gray);
        const histSize = [256];
        const ranges = [0.0, 256.0];
        hist = cv.calcHistUniform(images, [0], undefined, histSize, ranges, false, undefined);
        drawHistogram(histCanvas, hist.dataF32(), '#43e97b');

        // --- equalize_hist: global histogram equalization ---
        equalized = cv.equalizeHist(gray);

        // --- CLAHE: contrast-limited adaptive histogram equalization ---
        clahe = new cv.Clahe(clipLimit, tileGrid, tileGrid);
        claheOut = clahe.apply(gray);

        // --- compare_hist: source vs. CLAHE-equalized histograms ---
        claheImages = new cv.MatVector();
        claheImages.push(claheOut);
        claheHist = cv.calcHistUniform(claheImages, [0], undefined, histSize, ranges, false, undefined);

        for (const method of COMPARE_METHODS) {
            const score = cv.compareHist(hist, claheHist, method.ctor(cv));
            const item = document.createElement('div');
            item.className = 'score-item';
            item.innerHTML = `<span class="name">${method.name}</span><span class="value">${score.toFixed(4)}</span>`;
            scoresEl.appendChild(item);
        }

        // --- render the three grayscale variants side by side ---
        matToCanvas(gray, addCanvasBox(`Grayscale: ${gray.cols}x${gray.rows}`));
        matToCanvas(equalized, addCanvasBox('equalize_hist'));
        matToCanvas(claheOut, addCanvasBox(`CLAHE (clip=${clipLimit}, ${tileGrid}x${tileGrid})`));
    } catch (e) {
        console.error("Histogram/CLAHE error:", e);
    } finally {
        gray.free();
        images?.free();
        hist?.free();
        equalized?.free();
        clahe?.free();
        claheOut?.free();
        claheImages?.free();
        claheHist?.free();
    }
}

clipSlider.oninput = () => {
    clipDisplay.innerText = clipSlider.value;
    processImage();
};

tileSlider.oninput = () => {
    tilesDisplay.innerText = tileSlider.value;
    tilesDisplay2.innerText = tileSlider.value;
    processImage();
};

const fileInput = document.getElementById('file-input');
const dropZone = document.getElementById('drop-zone');

async function loadFile(file) {
    if (!file) return;
    const url = URL.createObjectURL(file);
    try {
        sourceImage = await loadImage(url);
        processImage();
    } finally {
        URL.revokeObjectURL(url);
    }
}

dropZone.onclick = () => fileInput.click();
fileInput.onchange = (e) => loadFile(e.target.files[0]);

dropZone.ondragover = (e) => {
    e.preventDefault();
    dropZone.classList.add('drag-over');
};
dropZone.ondragleave = () => dropZone.classList.remove('drag-over');
dropZone.ondrop = (e) => {
    e.preventDefault();
    dropZone.classList.remove('drag-over');
    loadFile(e.dataTransfer.files[0]);
};

start();
