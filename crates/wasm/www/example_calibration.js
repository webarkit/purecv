/*
 *  example_calibration.js
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

// ---------------------------------------------------------------------------
//  State & Constants
// ---------------------------------------------------------------------------

let cv = null;
let sourceImage = null;
let currentMode = 'undistort'; // 'undistort' or 'warp'

const CANVAS_SIZE = 400;
const GRAB_RADIUS = 14;

// Draggable corners for Perspective Warp mode
let corners = [
    { x: 80, y: 80 },   // TL
    { x: 320, y: 70 },  // TR
    { x: 330, y: 330 }, // BR
    { x: 70, y: 320 }   // BL
];
let draggingIndex = -1;

// Elements
const canvasSrc = document.getElementById('canvas-src');
const ctxSrc = canvasSrc.getContext('2d', { willReadFrequently: true });
const canvasDst = document.getElementById('canvas-dst');
const ctxDst = canvasDst.getContext('2d', { willReadFrequently: true });

const tabUndistort = document.getElementById('tab-undistort');
const tabWarp = document.getElementById('tab-warp');
const panelUndistort = document.getElementById('panel-undistort');
const panelWarp = document.getElementById('panel-warp');
const labelCanvasSrc = document.getElementById('label-canvas-src');

const paramK1 = document.getElementById('param-k1');
const paramK2 = document.getElementById('param-k2');
const paramP1 = document.getElementById('param-p1');
const paramP2 = document.getElementById('param-p2');
const paramF = document.getElementById('param-f');

const valK1 = document.getElementById('val-k1');
const valK2 = document.getElementById('val-k2');
const valP1 = document.getElementById('val-p1');
const valP2 = document.getElementById('val-p2');
const valF = document.getElementById('val-f');

// ---------------------------------------------------------------------------
//  Main logic
// ---------------------------------------------------------------------------

async function start() {
    try {
        cv = await initWasm();
        document.getElementById('loader').classList.add('hidden');

        // Load the default butterfly image
        sourceImage = await loadImage('https://raw.githubusercontent.com/opencv/opencv/4.10.0/samples/data/butterfly.jpg');

        setupEventListeners();
        updateSliderLabels();
        render();
    } catch (err) {
        console.error("WASM Initialization failed:", err);
        document.getElementById('loader').innerHTML = `<p style="color:red">Error loading WASM: ${err.message}</p>`;
    }
}

function render() {
    if (!sourceImage || !cv) return;

    if (currentMode === 'undistort') {
        runUndistortion();
    } else {
        runPerspectiveWarp();
    }
}

// ---------------------------------------------------------------------------
//  Mode 1: Camera Undistortion & Remap
// ---------------------------------------------------------------------------

function runUndistortion() {
    labelCanvasSrc.textContent = "Original Image";

    // Setup source canvas with scaled image
    const { width, height } = getScaledDimensions(sourceImage, CANVAS_SIZE);
    canvasSrc.width = width;
    canvasSrc.height = height;
    ctxSrc.drawImage(sourceImage, 0, 0, width, height);

    // Convert source to PureCV Mat
    const srcMat = canvasToMat(cv, canvasSrc, ctxSrc);

    // Read parameters
    const k1 = parseFloat(paramK1.value);
    const k2 = parseFloat(paramK2.value);
    const p1 = parseFloat(paramP1.value);
    const p2 = parseFloat(paramP2.value);
    const f = parseFloat(paramF.value);

    try {
        // Construct standard camera matrix
        // [ fx,  0, cx ]
        // [  0, fy, cy ]
        // [  0,  0,  1 ]
        const cx = width / 2.0;
        const cy = height / 2.0;
        const fx = width * f;
        const fy = height * f;
        const camMat = cv.Mat.fromF64Data(3, 3, 1, new Float64Array([
            fx, 0, cx,
            0, fy, cy,
            0, 0, 1.0
        ]));

        // Distortion coefficients: [k1, k2, p1, p2, k3, k4, k5, k6]
        const distCoeffs = cv.Mat.fromF64Data(1, 5, 1, new Float64Array([
            k1, k2, p1, p2, 0.0
        ]));

        // Optional rectification (Identity)
        const rMat = cv.Mat.fromF64Data(3, 3, 1, new Float64Array([
            1.0, 0, 0,
            0, 1.0, 0,
            0, 0, 1.0
        ]));

        // New camera matrix (Same as original for simplicity)
        const newCamMat = cv.Mat.fromF64Data(3, 3, 1, new Float64Array([
            fx, 0, cx,
            0, fy, cy,
            0, 0, 1.0
        ]));

        // Generate maps
        const maps = cv.initUndistortRectifyMap(camMat, distCoeffs, rMat, newCamMat, width, height);
        const map1 = maps.map1;
        const map2 = maps.map2;

        // Apply remap (Linear interpolation, Constant border)
        const dstMat = cv.remap(srcMat, map1, map2, 1, cv.BORDER_CONSTANT(), cv.Scalar.all(0));

        // Render destination canvas
        matToCanvas(dstMat, canvasDst);

        // Deallocate WASM matrices
        camMat.free();
        distCoeffs.free();
        rMat.free();
        newCamMat.free();
        map1.free();
        map2.free();
        dstMat.free();
    } catch (e) {
        console.error("Undistortion remapping failed:", e);
    }

    srcMat.free();
}

// ---------------------------------------------------------------------------
//  Mode 2: Perspective Warping
// ---------------------------------------------------------------------------

function runPerspectiveWarp() {
    labelCanvasSrc.textContent = "Source Image (Drag green corners)";

    const { width, height } = getScaledDimensions(sourceImage, CANVAS_SIZE);
    canvasSrc.width = width;
    canvasSrc.height = height;

    // Draw the image first
    ctxSrc.drawImage(sourceImage, 0, 0, width, height);

    // Draw lines connecting the corners
    ctxSrc.strokeStyle = '#10b981';
    ctxSrc.lineWidth = 2;
    ctxSrc.beginPath();
    ctxSrc.moveTo(corners[0].x, corners[0].y);
    for (let i = 1; i < corners.length; i++) {
        ctxSrc.lineTo(corners[i].x, corners[i].y);
    }
    ctxSrc.closePath();
    ctxSrc.stroke();

    // Draw corners
    corners.forEach((p, idx) => {
        ctxSrc.fillStyle = (idx === draggingIndex) ? '#38bdf8' : '#10b981';
        ctxSrc.beginPath();
        ctxSrc.arc(p.x, p.y, 8, 0, 2 * Math.PI);
        ctxSrc.fill();
        ctxSrc.lineWidth = 2;
        ctxSrc.strokeStyle = '#ffffff';
        ctxSrc.stroke();
    });

    // Run warp perspective
    const srcMat = canvasToMat(cv, canvasSrc, ctxSrc);

    try {
        const srcPts = new cv.Point2fVector();
        const dstPts = new cv.Point2fVector();

        // Target quad is the entire output canvas
        srcPts.push(0, 0);
        srcPts.push(width, 0);
        srcPts.push(width, height);
        srcPts.push(0, height);

        // Source quad is the draggable polygon
        corners.forEach(p => dstPts.push(p.x, p.y));

        // Find homography (Source to Destination)
        const hRes = cv.findHomography(srcPts, dstPts, 0, 3.0);
        const hMat = hRes.homography;

        // Warp image onto the draggable quadrilateral
        const dstMat = cv.warpPerspective(srcMat, hMat, width, height, 1, cv.BORDER_CONSTANT(), new cv.Scalar(0, 0, 0, 0));

        matToCanvas(dstMat, canvasDst);

        srcPts.free();
        dstPts.free();
        hMat.free();
        hRes.mask.free();
        dstMat.free();
    } catch (e) {
        console.error("Warp perspective failed:", e);
    }

    srcMat.free();
}

// ---------------------------------------------------------------------------
//  Interaction & Event Listeners
// ---------------------------------------------------------------------------

function setupEventListeners() {
    // Tab Switching
    tabUndistort.onclick = () => {
        tabUndistort.classList.add('active');
        tabWarp.classList.remove('active');
        panelUndistort.classList.remove('hidden');
        panelWarp.classList.add('hidden');
        currentMode = 'undistort';
        render();
    };

    tabWarp.onclick = () => {
        tabWarp.classList.add('active');
        tabUndistort.classList.remove('active');
        panelWarp.classList.remove('hidden');
        panelUndistort.classList.add('hidden');
        currentMode = 'warp';
        render();
    };

    // Sliders
    [paramK1, paramK2, paramP1, paramP2, paramF].forEach(slider => {
        slider.oninput = () => {
            updateSliderLabels();
            render();
        };
    });

    // Resets
    document.getElementById('reset-undistort').onclick = () => {
        paramK1.value = -0.40;
        paramK2.value = 0.10;
        paramP1.value = 0.00;
        paramP2.value = 0.00;
        paramF.value = 0.95;
        updateSliderLabels();
        render();
    };

    document.getElementById('reset-warp').onclick = () => {
        const { width, height } = getScaledDimensions(sourceImage, CANVAS_SIZE);
        corners = [
            { x: width * 0.2, y: height * 0.2 },
            { x: width * 0.8, y: height * 0.18 },
            { x: width * 0.82, y: height * 0.82 },
            { x: width * 0.18, y: height * 0.8 }
        ];
        render();
    };

    // Custom Image Uploader
    const fileInput = document.getElementById('file-input');
    const dropZone = document.getElementById('drop-zone');
    dropZone.onclick = () => fileInput.click();
    fileInput.onchange = async (e) => {
        if (e.target.files.length > 0) {
            const url = URL.createObjectURL(e.target.files[0]);
            sourceImage = await loadImage(url);
            
            // Reset corners position for the new image size
            const { width, height } = getScaledDimensions(sourceImage, CANVAS_SIZE);
            corners = [
                { x: width * 0.2, y: height * 0.2 },
                { x: width * 0.8, y: height * 0.18 },
                { x: width * 0.82, y: height * 0.82 },
                { x: width * 0.18, y: height * 0.8 }
            ];
            render();
        }
    };

    // Dragging corners interaction on canvasSrc
    canvasSrc.addEventListener('pointerdown', (e) => {
        if (currentMode !== 'warp') return;
        const pos = getMousePos(e);
        for (let i = 0; i < corners.length; i++) {
            const dx = pos.x - corners[i].x;
            const dy = pos.y - corners[i].y;
            if (Math.sqrt(dx * dx + dy * dy) < GRAB_RADIUS) {
                draggingIndex = i;
                canvasSrc.setPointerCapture(e.pointerId);
                render();
                break;
            }
        }
    });

    canvasSrc.addEventListener('pointermove', (e) => {
        if (draggingIndex < 0) return;
        const pos = getMousePos(e);
        const { width, height } = getScaledDimensions(sourceImage, CANVAS_SIZE);
        corners[draggingIndex].x = Math.max(10, Math.min(width - 10, pos.x));
        corners[draggingIndex].y = Math.max(10, Math.min(height - 10, pos.y));
        render();
    });

    const stopDragging = () => {
        if (draggingIndex >= 0) {
            draggingIndex = -1;
            render();
        }
    };
    canvasSrc.addEventListener('pointerup', stopDragging);
    canvasSrc.addEventListener('pointercancel', stopDragging);
}

function getMousePos(e) {
    const rect = canvasSrc.getBoundingClientRect();
    const { width, height } = getScaledDimensions(sourceImage, CANVAS_SIZE);
    const scaleX = width / rect.width;
    const scaleY = height / rect.height;
    return {
        x: (e.clientX - rect.left) * scaleX,
        y: (e.clientY - rect.top) * scaleY,
    };
}

function updateSliderLabels() {
    valK1.textContent = parseFloat(paramK1.value).toFixed(2);
    valK2.textContent = parseFloat(paramK2.value).toFixed(2);
    valP1.textContent = parseFloat(paramP1.value).toFixed(2);
    valP2.textContent = parseFloat(paramP2.value).toFixed(2);
    valF.textContent = parseFloat(paramF.value).toFixed(2);
}

// ---------------------------------------------------------------------------
//  Bootstrap
// ---------------------------------------------------------------------------

start();
