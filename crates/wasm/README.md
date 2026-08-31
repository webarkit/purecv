# @webarkit/purecv-wasm

This is the official WebAssembly binding for **PureCV**, providing high-performance, pure-Rust image processing and computer vision functions directly in the browser and Node.js.

[![NPM version](https://img.shields.io/npm/v/@webarkit/purecv-wasm.svg)](https://www.npmjs.com/package/@webarkit/purecv-wasm)
[![GitHub Stars](https://img.shields.io/github/stars/webarkit/purecv.svg?style=social)](https://github.com/webarkit/purecv/stargazers)
[![GitHub Forks](https://img.shields.io/github/forks/webarkit/purecv.svg?style=social)](https://github.com/webarkit/purecv/network/members)

## Dual-Build Architecture

This package comes heavily optimized out of the box with a **dual-build strategy**:
- **Standard (`dist-std`)**: Provides maximum browser compatibility.
- **SIMD (`dist-simd`)**: Delivers massive performance boosts in environments that support WebAssembly `simd128` (most modern browsers).

You can load either module depending on your runtime capabilities.

## Installation

```bash
npm install @webarkit/purecv-wasm
```

## Usage Example

```javascript
// Example using a bundler (webpack, vite) or in Node.js
import init, { Mat, gaussian_blur, CV_8UC3, BORDER_DEFAULT } from '@webarkit/purecv-wasm/dist-std/purecv_wasm.js';

async function run() {
    // Initialize WebAssembly
    await init();

    // Mat.new(rows, cols, mat_type)
    const mat = Mat.new(480, 640, CV_8UC3);
    
    // Process your image data here...
    // For example, uploading an ImageData array buffer to `mat`

    // gaussian_blur(src, ksize_w, ksize_h, sigma_x, sigma_y, border_type)
    const blurredMat = gaussian_blur(mat, 5, 5, 0.0, 0.0, BORDER_DEFAULT);

    // Free memory manually!
    mat.free();
    blurredMat.free();
}

run();
```

For a concrete, runnable example directly in an HTML file using `fetch`, check out the `www/` demo directory in our GitHub repository!

## Memory Management

Because WebAssembly runs linearly in memory and holds pointers to Rust `Vec` objects, memory allocated for matrices such as `Mat` are not automatically garbage collected by JavaScript. When you are done manipulating your matrix, **you must ensure you call `.free()`** to prevent memory leaks in the browser.

## API coverage

Right now we have covered a large majority of operations for `core` and `imgproc`, and have started on `calib3d` and `video`:

- **Core**: Arithmetic (`add`, `subtract`, `multiply`, `absdiff` etc.), Structural (`hconcat`, `vconcat`, `flip`), Geometry, constants etc.
- **ImgProc**: Filters (`blur`, `gaussian_blur`, `bilateral_filter`), Thresholding (`threshold`), Coloring (`cvt_color`), Edge Derivatives (`canny`, `sobel`, `laplacian`), Morphology (`erode`, `dilate`, `morphology_ex`, `get_structuring_element`), Pyramids (`pyr_down`, `pyr_up`, `build_pyramid`), Feature Detection (`good_features_to_track`, `corner_sub_pix`), Histograms (`calcHistUniform`, `calcHistNonUniform`, `calcBackProjectUniform`, `calcBackProjectNonUniform`, `compareHist`, `equalizeHist`, `Clahe`).
- **Video**: Optical Flow (`calc_optical_flow_pyr_lk`).
- **Calib3d**: Pose Estimation (`solve_pnp`, `solve_pnp_ransac`), Homography (`find_homography`), and geometry (`rodrigues`).

### Histogram & Contrast Operations

```javascript
import { Mat, MatVector, calcHistUniform, equalizeHist, Clahe } from '@webarkit/purecv-wasm';

// 1. Equalize Histogram (8-bit grayscale only)
const eqMat = equalizeHist(grayMat);

// 2. CLAHE (Contrast Limited Adaptive Histogram Equalization)
const clahe = new Clahe(40.0, 8, 8);
const enhancedMat = clahe.apply(grayMat);

// 3. Dense Histogram with uniform bins
const images = new MatVector();
images.push(grayMat);
const channels = [0];
const histSize = [256];
const ranges = [0.0, 256.0]; // [min, max] per channel
const hist = calcHistUniform(images, channels, undefined, histSize, ranges, false);
```

*Note:* `equalizeHist` and `Clahe.apply` currently support single-channel 8-bit images (`CV_8UC1`). Support for 16-bit images (`CV_16UC1`) is planned for a future release.

Note: To interface between JavaScript Typed Arrays and `purecv-wasm`, please use the available getter functions (`.data()`) which directly retrieve a Float32Array or Uint8Array view into WASM memory.
