/*
 *  example_pyramid.js
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
const stack = document.getElementById('pyramid-stack');
const levelsSlider = document.getElementById('levels');
const levelsDisplay = document.getElementById('val-levels');

async function start() {
    try {
        cv = await initWasm();
        document.getElementById('loader').classList.add('hidden');

        sourceImage = await loadImage('https://raw.githubusercontent.com/opencv/opencv/4.10.0/samples/data/butterfly.jpg');
        processImage();
    } catch (err) {
        console.error("WASM Initialization failed:", err);
        document.getElementById('loader').innerHTML = `<p style="color:red">Error loading WASM: ${err.message}</p>`;
    }
}

function processImage() {
    if (!sourceImage || !cv) return;

    stack.innerHTML = '';

    const { width, height } = getScaledDimensions(sourceImage, 512);
    const tempCanvas = document.createElement('canvas');
    tempCanvas.width = width;
    tempCanvas.height = height;
    const tempCtx = tempCanvas.getContext('2d', "willReadFrequently: true");
    tempCtx.drawImage(sourceImage, 0, 0, width, height);

    const mat = canvasToMat(cv, tempCanvas, tempCtx);
    const maxLevel = parseInt(levelsSlider.value);
    
    try {
        const levels = cv.buildPyramid(mat, maxLevel, cv.BORDER_REFLECT_101());

        levels.forEach((lvl, index) => {
            const levelBox = document.createElement('div');
            levelBox.className = 'level-box';
            
            const levelCanvas = document.createElement('canvas');
            matToCanvas(lvl, levelCanvas);
            
            const label = document.createElement('span');
            label.className = 'level-label';
            label.innerText = `L${index}: ${lvl.cols}x${lvl.rows}`;
            
            levelBox.appendChild(levelCanvas);
            levelBox.appendChild(label);
            stack.appendChild(levelBox);
            
            lvl.free();
        });
    } catch (e) {
        console.error("Pyramid error:", e);
    }

    mat.free();
}

levelsSlider.oninput = () => {
    levelsDisplay.innerText = levelsSlider.value;
    processImage();
};

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
