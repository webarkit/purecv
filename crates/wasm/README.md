# @webarkit/purecv-wasm

This is the official WebAssembly binding for **PureCV**, providing high-performance, pure-Rust image processing and computer vision functions directly in the browser and Node.js.

[![NPM version](https://img.shields.io/npm/v/@webarkit/purecv-wasm.svg)](https://www.npmjs.com/package/@webarkit/purecv-wasm)

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

Right now we have covered a large majority of operations for `core` and `imgproc`:

- **Core**: Arithmetic (`add`, `subtract`, `multiply`, `absdiff` etc.), Structural (`hconcat`, `vconcat`, `flip`), Geometry, constants etc.
- **ImgProc**: Filters (`blur`, `gaussian_blur`, `bilateral_filter`), Thresholding (`threshold`), Coloring (`cvt_color`), Edge Derivatives (`canny`, `sobel`, `laplacian`), Morphology (`erode`, `dilate`, `morphology_ex`, `get_structuring_element`), Pyramids (`pyr_down`, `pyr_up`, `build_pyramid`).

Note: To interface between JavaScript Typed Arrays and `purecv-wasm`, please use the available getter functions (`.data()`) which directly retrieve a Float32Array or Uint8Array view into WASM memory.
