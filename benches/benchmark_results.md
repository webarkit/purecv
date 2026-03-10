# SIMD & Parallelization Benchmark Results

*Execution Date: 2026-03-09 23:43 (CET)*

This document highlights the performance evaluation of the `purecv` library across four main compilation strategies:

1. **Standard**: Sequential fallback mode (`--no-default-features`), without explicit target CPU optimizations.
2. **SIMD Only**: Sequential mode compiled with `RUSTFLAGS="-C target-cpu=native"` to encourage LLVM auto-vectorization.
3. **Parallel**: Enabled `rayon` multi-threading across available cores (`--features parallel`).
4. **Parallel + SIMD**: Combined `rayon` parallelism alongside `target-cpu=native` for maximum theoretical throughput.

All tests operate on `1024x1024` image/matrix tensors using `f32` (or `u8` depending on the domain context). Times shown represent the median calculation calculated by `Criterion.rs`.

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
