// OpenCV 4.10.0 (opencv.js) reference values for Matrix::convert_to (docs/parity/core.md).
//
// Python's cv2 does not bind Mat::convertTo, so the direct measurement of convertTo comes
// from the pinned opencv.js build: the same cvt kernels (saturate_cast) compiled to WASM.
//
// Run from the repository root after `npm ci --prefix cpp_ref`:
//     node scripts/opencv_ref/core_convert_to.js
//
// The package is a legacy Emscripten build: never `await` the module (see cpp_ref/README.md).

const path = require("path");
const cv = require(path.join(__dirname, "..", "..", "cpp_ref", "node_modules", "@techstark", "opencv-js"));

const SRC = [
  0.5, 1.5, 2.5, 3.5, -0.5, -1.5, -2.5, 254.5, 254.6, 255.5, 300.0, -1.0, -300.0,
  127.5, 128.5, -128.5, -129.0, 65535.5, 32767.5, -32768.5, 1e10, -1e10, NaN, Infinity, -Infinity,
];

function convert(srcType, dstType, alpha, beta) {
  const src = cv.matFromArray(1, SRC.length, srcType, SRC);
  const dst = new cv.Mat();
  src.convertTo(dst, dstType, alpha, beta);
  const out = Array.from(dst.data8S.length ? [] : []); // placeholder, replaced below
  let values;
  switch (dstType) {
    case cv.CV_8U: values = Array.from(dst.data); break;
    case cv.CV_8S: values = Array.from(dst.data8S); break;
    case cv.CV_16U: values = Array.from(dst.data16U); break;
    case cv.CV_16S: values = Array.from(dst.data16S); break;
    case cv.CV_32S: values = Array.from(dst.data32S); break;
    default: values = Array.from(dst.data32F);
  }
  src.delete();
  dst.delete();
  return values;
}

cv.onRuntimeInitialized = () => {
  const versionLine = cv.getBuildInformation().match(/Version control:.*/)[0].trim();
  const targets = { u8: cv.CV_8U, i8: cv.CV_8S, u16: cv.CV_16U, i16: cv.CV_16S, i32: cv.CV_32S };
  const results = { src: SRC.map((v) => (Number.isNaN(v) ? "nan" : v === Infinity ? "inf" : v === -Infinity ? "-inf" : v)) };
  for (const [srcName, srcType] of [["f32", cv.CV_32F], ["f64", cv.CV_64F]]) {
    const block = {};
    for (const [name, type] of Object.entries(targets)) {
      block[name] = convert(srcType, type, 1.0, 0.0);
      block[`${name}_alpha2_beta0.5`] = convert(srcType, type, 2.0, 0.5);
    }
    results[`convert_to_from_${srcName}`] = block;
  }
  console.log(JSON.stringify({ script: "core_convert_to.js", build_info: { opencv_js: versionLine }, results }, null, 1));
  process.exit(0);
};
