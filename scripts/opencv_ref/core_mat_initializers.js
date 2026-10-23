/*
 *  core_mat_initializers.js
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

// OpenCV 4.10.0 (opencv.js) reference values for Matrix::{ones, eye} (docs/parity/4.10.0/core.md).
//
// Python's cv2 does not bind Mat::ones / Mat::eye (numpy arrays stand in for Mat), so the
// measurement comes from the pinned opencv.js build. Both return a MatExpr whose scale is
// written as Scalar(alpha): only channel 0 receives the value on multi-channel types.
//
// Run from the repository root after `npm ci --prefix cpp_ref`:
//     node scripts/opencv_ref/core_mat_initializers.js
//
// The package is a legacy Emscripten build: never `await` the module (see cpp_ref/README.md).

const path = require("path");
const cv = require(path.join(__dirname, "..", "..", "cpp_ref", "node_modules", "@techstark", "opencv-js"));

function values(mat) {
  const depth = mat.depth();
  let v;
  if (depth === cv.CV_8U) v = Array.from(mat.data);
  else if (depth === cv.CV_32F) v = Array.from(mat.data32F);
  else v = Array.from(mat.data64F);
  const out = { rows: mat.rows, cols: mat.cols, channels: mat.channels(), data: v };
  mat.delete();
  return out;
}

cv.onRuntimeInitialized = () => {
  const versionLine = cv.getBuildInformation().match(/Version control:.*/)[0].trim();
  const results = {
    ones_1x1_8uc3: values(cv.Mat.ones(1, 1, cv.CV_8UC3)),
    ones_2x2_32fc2: values(cv.Mat.ones(2, 2, cv.CV_32FC2)),
    ones_2x3_8uc1: values(cv.Mat.ones(2, 3, cv.CV_8UC1)),
    eye_2x2_8uc2: values(cv.Mat.eye(2, 2, cv.CV_8UC2)),
    eye_3x3_32fc3: values(cv.Mat.eye(3, 3, cv.CV_32FC3)),
    eye_2x3_32fc1: values(cv.Mat.eye(2, 3, cv.CV_32FC1)),
  };
  console.log(JSON.stringify({ script: "core_mat_initializers.js", build_info: { opencv_js: versionLine }, results }, null, 1));
  process.exit(0);
};
