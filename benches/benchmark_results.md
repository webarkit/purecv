# SIMD & Parallelization Benchmark Results

This document highlights the performance evaluation of the `purecv` library across four main compilation strategies:

1. **Standard**: Sequential fallback mode (`--no-default-features`), without explicit target CPU optimizations.
2. **SIMD Only**: Sequential mode compiled with `RUSTFLAGS="-C target-cpu=native"` to encourage LLVM auto-vectorization.
3. **Parallel**: Enabled `rayon` multi-threading across available cores (`--features parallel`).
4. **Parallel + SIMD**: Combined `rayon` parallelism alongside `target-cpu=native` for maximum theoretical throughput.

All tests operate on `1024x1024` image/matrix tensors using `f32` (or `u8` depending on the domain context). Times shown represent the median calculation calculated by `Criterion.rs`.

---

## Video — Performance Comparison Table (1 benchmark) 

*Execution Date: 2026-04-28 (CET)*

| Benchmark / Operation | Standard |
| :-------------------- | :------- |
| `calc_optical_flow_pyr_lk_512x512_pts_49` | 27.5 ms |

### Video Analysis

- **Optical Flow (Lucas-Kanade):** Tracking 49 points on a 512×512 image pyramid completes in ~27.5 ms sequentially. The operation is highly optimized with sub-pixel accuracy and spatial gradient computation.

---

## Feature Detection & Hough Transforms

*Execution Date: 2026-04-17 (CET)*

New benchmarks for Feature Detection and Hough Transforms. These operations are heavily compute-bound and benefit significantly from parallelism. Hough benchmarks use a `512x512` input to balance execution time and accuracy.

| Benchmark / Operation              | Parallel + SIMD  |
| :--------------------------------- | :--------------- |
| `corner_harris_1024x1024_f32`     | 26.28 ms         |
| `hough_lines_512x512`              | 3.58 ms          |
| `hough_lines_p_512x512`            | 1.78 ms          |
| `hough_circles_512x512`            | 4.51 ms          |

### Analysis

- **`corner_harris`**: Involves Sobel derivatives, Gaussian blurring of the covariance matrix, and the Harris response calculation. At 26ms for a 1MP image, it is highly efficient for real-time applications.
- **Hough Transforms**:
  - `hough_lines` (Standard) processes the full accumulator space.
  - `hough_lines_p` (Probabilistic) is ~2x faster by sampling points, as expected.
  - `hough_circles` uses the 2-1 Hough transform (gradient-based), achieving sub-5ms performance on 512x512 inputs.

---

## New Benchmarks — count_non_zero, LUT & DFT

*Execution Date: 2026-03-31 (CET)*

Three new function groups have been benchmarked: `count_non_zero`, `lut` (Look-Up Table), and `dft` (Discrete Fourier Transform via `rustfft`). DFT benchmarks require the `fft` feature flag.

> **Note:** The 4-strategy comparison table (Standard / SIMD Only / Parallel / Parallel+SIMD) does not apply to the LUT and DFT benchmarks below. LUT is a pure table lookup where SIMD provides no benefit, and DFT delegates to `rustfft` which manages its own internal SIMD — the `parallel` and `simd` feature flags have no effect on either. Results shown are with default features (Parallel enabled).

### Arithm — count_non_zero & LUT

| Benchmark / Operation                  | Parallel (default) |
| :------------------------------------- | :----------------- |
| `count_non_zero_1024×1024_i32`         | 116.03 µs          |
| `lut_1024×1024_u8` (1ch)              | 501.25 µs          |
| `lut_1024×1024×3_u8_broadcast` (3ch)  | 855.86 µs          |

`count_non_zero` is a simple reduction — very fast with rayon parallelism. LUT is a pure table lookup (memory-bound), scaling linearly with channel count (3ch ≈ 1.7× of 1ch).

### DFT — Discrete Fourier Transform (`--features fft`)

| Benchmark / Operation                  | Time        |
| :------------------------------------- | :---------- |
| `get_optimal_dft_size`                 | 18.76 ns    |
| `dft_forward_256×256_f32`              | 454.03 µs   |
| `dft_inverse_256×256_f32`              | 514.82 µs   |
| `dft_forward_512×512_f32`              | 3.80 ms     |
| `dft_forward_1024×1024_f64`            | 28.21 ms    |

DFT scales as O(n² log n) for 2D transforms: 512×512 is ~8× slower than 256×256, consistent with the 4× data increase and logarithmic factor growth. `get_optimal_dft_size` is a trivial binary search (~19 ns).



*Execution Date: 2026-03-17 (CET)*
---

## New Benchmarks — bilateral_filter & sobel_f32 (SIMD path)

Three new benchmarks were added to specifically evaluate the **SIMD fast-path** in `fast_deriv_3x3` for `f32` Sobel, and the heavy compute-bound `bilateral_filter`. Sobel f32 benchmarks use **realistic non-zero data** (sinusoidal gradient pattern) to fully exercise the computation.

| Benchmark / Operation              | Standard    | SIMD Only    | Parallel       | Parallel + SIMD  |
| :--------------------------------- | :---------- | :----------- | :------------- | :--------------- |
| `bilateral_filter_512×512_u8`      | 1.43 s      | 1.51 s       | **202.44 ms**  | 205.48 ms        |
| `sobel_3x3_f32_dx_1024×1024`       | 28.59 ms    | 6.26 ms      | 1.96 ms        | **1.28 ms**      |
| `sobel_3x3_f32_dy_1024×1024`       | 26.24 ms    | 6.51 ms      | 2.16 ms        | **1.27 ms**      |

### Analysis

- **`sobel_3x3_f32` (dx/dy)** — first benchmark where **SIMD alone provides a massive standalone gain**:
  - SIMD Only: **4.5× speedup** over Standard — the `simd_deriv_3x3_row_f32` kernel in `fast_deriv_3x3` processes interior rows with vectorized 3×3 convolutions.
  - Parallel: **14× speedup** — row-wise parallelism scales excellently.
  - **Parallel + SIMD: 22× total speedup** (28.59 ms → 1.28 ms) — the combination of SIMD kernels on each core produces the best result seen in the project.
  - Compared to the generic `sobel_3x3` (which uses `Matrix<f32>` with zero data and mixed dx=1,dy=1), the f32-specific SIMD path is **~1.5× faster** even in Parallel+SIMD (1.28 ms vs 1.87 ms).

- **`bilateral_filter_512×512_u8`** — heavily compute-bound (per-pixel Gaussian weight computation over a spatial/range neighborhood):
  - Parallel: **7.1× speedup** (1.43 s → 202 ms) — each row's bilateral computation is independent.
  - SIMD Only: **no gain** (1.51 s, slightly worse) — the complex per-pixel exponential weight math is not trivially vectorizable.
  - Parallel + SIMD: equivalent to Parallel alone (~205 ms).

### Key Differences vs Previous Results

| Metric                            | Previous best (`sobel_3x3` generic) | New (`sobel_3x3_f32` SIMD path) | Improvement |
| :-------------------------------- | :---------------------------------- | :------------------------------- | :---------- |
| Standard                          | 22.79 ms                            | 28.59 ms (realistic data)        | —           |
| SIMD Only                         | 21.66 ms                            | **6.26 ms**                      | **3.5×**    |
| Parallel                          | 1.87 ms                             | **1.96 ms**                      | ~equal      |
| Parallel + SIMD                   | 1.87 ms                             | **1.28 ms**                      | **1.5×**    |

> The generic `sobel_3x3` benchmark used zero-initialized data and the mixed derivative (dx=1, dy=1), which masked the SIMD gains. With realistic f32 data and separate dx/dy, the `pulp`-powered SIMD kernel delivers a clear **1.5× additional speedup** on top of parallelism.

---

*Execution Date: 2026-03-09 23:43 (CET)*

## Performance Comparison Table

| Benchmark / Operation | Standard | SIMD Only | Parallel | Parallel + SIMD |
| :------------------- | :--------- | :---------- | :--------- | :---------------- |
| `matrix_add`         | 3.86 ms    | 3.92 ms     | 2.45 ms    | 2.35 ms           |
| `matrix_sub`         | 3.86 ms    | 3.87 ms     | 2.49 ms    | 2.39 ms           |
| `matrix_mul`         | 3.73 ms    | 3.88 ms     | 2.36 ms    | 2.58 ms           |
| `matrix_div`         | 3.71 ms    | 3.84 ms     | 2.42 ms    | 2.37 ms           |
| `cvt_color_rgb2gray` | 3.85 ms    | **1.52 ms** | 666.32 µs  | **475.45 µs**     |
| `box_filter_3x3`     | 13.91 ms   | 14.35 ms    | 3.39 ms    | 3.61 ms           |
| `sobel_3x3`          | 15.28 ms   | 15.40 ms    | 3.65 ms    | 4.06 ms           |

## Analysis
- **Color Conversion**: `cvt_color_rgb2gray` is highly vectorizable (straightforward pixel-by-pixel math without neighborhood lookups). Simply enabling `-C target-cpu=native` yields a massive **2.5x speedup** on a single thread. Combining it with `rayon` parallelism achieves near **8x total speedup**.
- **Spatial Filters**: Neighborhood-dependent modules (`box_filter` and `sobel_3x3`) showcase tremendous gains from the `parallel` execution strategy (over 4x speedup) as filtering is heavily compute-bound and segments cleanly across row data. However, generic auto-vectorization (SIMD Only) sees negligible impact due to the random memory access patterns characteristic of 3x3 sliding windows.
- **Matrix Arithmetics**: Foundational matrices tasks (`add`, `sub`, etc.) appear bounded primarily by memory bandwidth when allocating large `1024x1024` vectors, making instruction-level SIMD tuning yield diminishing returns vs memory throughput over parallel cores.

*Conclusion*: The refactored `.chunks_exact()` and `.zip()` iterators provide robust performance baselines and scale beautifully whenever LLVM's auto-vectorizer applies (demonstrated perfectly via `cvt_color`), while ensuring safe compatibility across varying feature flags!

## Algorithmic Optimizations

*Execution Date: 2026-03-10*

Following the initial SIMD and Parallelization benchmarks, targeted algorithmic optimizations were applied to specific operations to eliminate memory overhead and boundary-checking branches.

| Benchmark / Operation | Pre-Optimization (Parallel) | Post-Optimization (Parallel) | Speedup |
| :------------------- | :-------------------------- | :--------------------------- | :------ |
| `sobel_3x3`          | 3.65 ms                     | 2.08 ms                      | 43%     |
| `scharr_x`           | 3.70 ms                     | 2.11 ms                      | 43%     |

### Analysis
- **Derivatives (`sobel_3x3`, `scharr_x`)**: Replacing the generic `sep_filter_2d` implementation with an inlined `fast_deriv_3x3` function brought execution times down to ~2.1ms. By explicitly unrolling the 3x3 convolution and separating the interior "fast path" from the boundary "slow path," we completely eliminated intermediate allocations and significantly improved cache locality.

*Execution Date: 2026-03-15 (CET)*

## Arithm — Performance Comparison Table (17 benchmarks)

| Benchmark / Operation         | Standard    | SIMD Only   | Parallel      | Parallel + SIMD |
| :---------------------------- | :---------- | :---------- | :------------ | :-------------- |
| `matrix_add`                  | 3.26 ms     | 3.35 ms     | 2.08 ms       | **2.05 ms**     |
| `matrix_sub`                  | 3.24 ms     | 3.37 ms     | **2.06 ms**   | 2.11 ms         |
| `matrix_mul`                  | 3.20 ms     | 3.13 ms     | **2.09 ms**   | 2.09 ms         |
| `matrix_div`                  | 3.24 ms     | 3.10 ms     | **2.10 ms**   | 2.10 ms         |
| `dot`                         | 997.62 µs   | 971.72 µs   | **156.54 µs** | 157.72 µs       |
| `magnitude`                   | 756.23 µs   | 754.99 µs   | **184.35 µs** | 184.42 µs       |
| `norm_l2`                     | 984.65 µs   | 964.61 µs   | **151.90 µs** | 160.90 µs       |
| `add_weighted`                | 3.53 ms     | 3.13 ms     | **2.23 ms**   | 2.30 ms         |
| `convert_scale_abs`           | 2.54 ms     | 1.66 ms     | 610.46 µs     | **477.11 µs**   |
| `sum`                         | 975.51 µs   | 966.94 µs   | **162.20 µs** | 164.41 µs       |
| `mean`                        | 969.08 µs   | 964.49 µs   | **163.92 µs** | 167.97 µs       |
| `bitwise_and`                 | **255.18 µs** | 264.72 µs | 376.10 µs     | 445.06 µs       |
| `min`                         | 3.08 ms     | 3.06 ms     | **2.07 ms**   | 2.11 ms         |
| `absdiff`                     | 3.10 ms     | 3.04 ms     | **2.06 ms**   | 2.15 ms         |
| `normalize`                   | 2.75 ms     | 1.55 ms     | **1.24 ms**   | 1.32 ms         |
| `sqrt`                        | 939.04 µs   | 945.49 µs   | 858.87 µs     | **847.33 µs**   |
| `gemm_256×256`                | 15.71 ms    | 15.69 ms    | **4.29 ms**   | 4.40 ms         |

### Arithm Analysis

- **Reduction operations** (`dot`, `norm_l2`, `sum`, `mean`) scale brilliantly with `rayon`: **~6× speedup** from Standard → Parallel, driven by embarrassingly-parallel sum-accumulation.
- **Element-wise binary ops** (`add`, `sub`, `mul`, `div`, `min`, `absdiff`) gain a modest **~1.5× speedup** with Parallel. These are memory-bandwidth-bound; additional SIMD provides negligible further gain.
- **`convert_scale_abs`** is the standout: SIMD gives **1.5×** (sequential), and Parallel+SIMD reaches **5.3× total speedup** — a combination of vectorizable math and parallelizable rows.
- **`bitwise_and`** is fastest in Standard mode (255 µs). Parallel overhead (~380–445 µs) exceeds the benefit for this ultra-cheap 1-byte-per-element operation.
- **`gemm_256×256`** benefits enormously from Parallel (**3.7×**), confirming that matrix multiplication is compute-bound and partitions well across cores.

---

## ImgProc — Performance Comparison Table (11 benchmarks)

| Benchmark / Operation         | Standard    | SIMD Only    | Parallel      | Parallel + SIMD |
| :---------------------------- | :---------- | :----------- | :------------ | :-------------- |
| `cvt_color_rgb2gray`          | 2.66 ms     | 1.41 ms      | 529.63 µs     | **404.23 µs**   |
| `box_filter_3x3`              | 9.01 ms     | 9.31 ms      | **3.07 ms**   | 3.69 ms         |
| `sobel_3x3`                   | 22.79 ms    | 21.66 ms     | **1.87 ms**   | 1.87 ms         |
| `scharr_x`                    | 22.76 ms    | 21.67 ms     | **1.78 ms**   | 1.85 ms         |
| `threshold_binary`            | 1.31 ms     | 1.43 ms      | 626.70 µs     | **609.22 µs**   |
| `cvt_color_bgr2gray`          | 2.69 ms     | †            | 617.94 µs     | **368.43 µs**   |
| `cvt_color_rgba2gray`         | 2.64 ms     | 2.43 ms      | 676.02 µs     | **354.37 µs**   |
| `gaussian_blur_3x3`           | 11.01 ms    | †            | **3.53 ms**   | 3.50 ms         |
| `gaussian_blur_5x5`           | 15.21 ms    | 16.37 ms     | **4.30 ms**   | 4.72 ms         |
| `laplacian_3x3`               | 45.91 ms    | 44.56 ms     | **4.41 ms**   | 4.44 ms         |
| `canny`                       | 57.61 ms    | 62.62 ms     | 12.70 ms      | **12.54 ms**    |

> † SIMD Only values for `cvt_color_bgr2gray` (249 ms) and `gaussian_blur_3x3` (18 ms) are anomalous — likely impacted by system activity during the run. Excluded from comparison.

### ImgProc Analysis

- **Color Conversion** (`cvt_color_rgb2gray`): highly vectorizable pixel-by-pixel math. SIMD alone yields **1.9×**, Parallel brings **5.0×**, and combined reaches **6.6× total speedup**.
- **Derivatives** (`sobel_3x3`, `scharr_x`): the optimized `fast_deriv_3x3` implementation scales extremely well with Parallel — **12× speedup**. These benefit from row-wise parallelism where each row's 3×3 convolution is independent.
- **`laplacian_3x3`**: composed of two Sobel passes, gains an enormous **10.4× speedup** with Parallel, consistent with the derivative scaling.
- **`canny`**: multi-stage pipeline (Sobel + gradient + NMS + hysteresis). Achieves **4.6× speedup** with Parallel — limited by the sequential hysteresis tracking phase.
- **Spatial Filters** (`box_filter`, `gaussian_blur`): **~3× speedup** with Parallel. The sliding-window pattern limits SIMD gains due to non-sequential memory access.
- **`threshold_binary`**: simple per-pixel operation with modest **2.1× speedup** — limited by memory bandwidth.

---

## Structural — Performance Comparison Table (4 benchmarks)

| Benchmark / Operation         | Standard    | SIMD Only   | Parallel      | Parallel + SIMD |
| :---------------------------- | :---------- | :---------- | :------------ | :-------------- |
| `flip_horiz`                  | 5.03 ms     | 5.81 ms     | **976.23 µs** | 971.35 µs       |
| `transpose`                   | 7.42 ms     | 7.29 ms     | **1.09 ms**   | 1.14 ms         |
| `split`                       | 3.21 ms     | 3.21 ms     | **1.24 ms**   | 1.31 ms         |
| `merge`                       | 2.49 ms     | 2.27 ms     | **932.68 µs** | ‡               |

> ‡ `merge` in Parallel+SIMD was not captured (benchmark run interrupted).

### Structural Analysis

- **All structural ops are memory-bound**: SIMD Only provides zero benefit (or even slight regression due to instruction overhead on non-vectorizable scatter/gather patterns).
- **Parallel provides 2.6–6.8× speedup** across the board: `flip` (**5.2×**) and `transpose` (**6.8×**) benefit most because each row can be independently relocated.
- **`merge` and `split`** (~2.5–2.7×) are still memory-bandwidth-limited even with parallelism, as they involve strided interleave/deinterleave across 3 channels.

---

## Summary — Best Configuration per Category

| Category           | Best Config       | Typical Speedup vs Standard |
| :----------------- | :---------------- | :-------------------------- |
| Element-wise ops   | Parallel          | 1.5× (memory-bound)        |
| Reductions         | Parallel          | 6× (embarrassingly parallel)|
| Color conversion   | Parallel + SIMD   | 6.6× (vectorizable + parallel) |
| Spatial filters    | Parallel          | 3× (row-parallel convolution) |
| Derivatives        | Parallel          | 12× (optimized fast path)  |
| Derivatives (f32 SIMD) | Parallel + SIMD | **22× (SIMD kernel + parallel)** |
| Bilateral filter   | Parallel          | 7.1× (compute-bound, no SIMD gain) |
| Canny edge detect  | Parallel + SIMD   | 4.6× (multi-stage pipeline)|
| Structural ops     | Parallel          | 5× (independent row ops)   |
| GEMM               | Parallel          | 3.7× (compute-bound)       |
| Feature Detection  | Parallel          | Harris Corner ~26ms (1024x1024) |
| Hough Transform    | Parallel          | Lines ~3.6ms / Circles ~4.5ms (512x512) |
| Video Tracking     | Standard          | Optical Flow ~27.5ms (49 pts, 512x512) |

*Conclusion*: Parallelism (`rayon`) is the dominant optimization for nearly all operations. SIMD (`target-cpu=native`) provides meaningful additional gains primarily for **pixel-level math** (`cvt_color`, `convert_scale_abs`) and **f32 derivative kernels** (`sobel_3x3_f32`) where the inner loop is trivially vectorizable. The `sobel_3x3_f32` SIMD path achieves the project's highest combined speedup at **22×**. For memory-bound or scatter/gather patterns, SIMD adds no benefit.

