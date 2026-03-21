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
import init, { PureCvMatrixU8, gaussianBlur } from '@webarkit/purecv-wasm/dist-std/purecv.js';

async function run() {
    // Initialize WebAssembly
    await init();

    const width = 640;
    const height = 480;
    const channels = 3;

    // Create a new matrix
    const mat = PureCvMatrixU8.create(width, height, channels);
    
    // Process your image data here...
    // For example, uploading an ImageData array buffer to `mat`

    // Apply a Gaussian Blur of size 5x5
    const blurredMat = gaussianBlur(mat, 5, 5, 0.0, 0.0);

    // Free memory manually!
    mat.free();
    blurredMat.free();
}

run();
```

For a concrete, runnable example directly in an HTML file using `fetch`, check out the `www/` demo directory in our GitHub repository!

## Memory Management

Because WebAssembly runs linearly in memory and holds pointers to Rust `Vec` objects, memory allocated for matrices such as `PureCvMatrixU8` and `PureCvMatrix` are not automatically garbage collected by JavaScript. When you are done manipulating your matrix, **you must ensure you call `.free()`** to prevent memory leaks in the browser.

## API coverage

Right now we have covered a large majority of operations for `core` and `imgproc`:

- **Core**: Arithmetic (`add`, `subtract`, `multiply`, `absDiff` etc.), Structural (`hconcat`, `vconcat`, `flip`), Geometry, constants etc.
- **ImgProc**: Filters (`blur`, `gaussianBlur`, `bilateralFilter`), Thresholding (`threshold`), Coloring (`cvtColor`), Edge Derivatives (`canny`, `sobel`, `laplacian`).

Note: To interface between JavaScript Typed Arrays and `purecv-wasm`, please use the available getter functions (`.data()`) which directly retrieve a Float32Array or Uint8Array view into WASM memory.
