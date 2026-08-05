/*
 *  simd.rs
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

//! SIMD-accelerated inner kernels for the Lucas-Kanade optical flow solver.
//!
//! This module contains two hot-path reductions that are called at every
//! pyramid level and at every LK iteration:
//!
//! * [`simd_lk_accumulate_h`] — builds the 2×2 spatial-gradient matrix H from
//!   pre-gathered window samples of `Ix` and `Iy`.
//! * [`simd_lk_accumulate_mismatch`] — accumulates the mismatch vector **b**
//!   from pre-gathered `Ix`, `Iy`, `I1`, and `I2` window samples.
//!
//! Both functions wrap their inner loops inside `pulp::Arch::dispatch`, which
//! signals LLVM to apply the best available SIMD instruction set (AVX2, SSE4,
//! ARM NEON, or WASM `simd128`) at runtime without any source-level `unsafe`
//! code.
//!
//! # Performance note
//!
//! The gather step (bilinear interpolation from the image pyramid) remains
//! scalar because it involves non-contiguous memory access.  These SIMD
//! kernels operate *after* the gather, on the pre-collected contiguous `f32`
//! buffers, where auto-vectorisation applies cleanly.

#[allow(unused_imports)]
use num_traits::Float;

// ---------------------------------------------------------------------------
// H matrix accumulation
// ---------------------------------------------------------------------------

/// Accumulate the LK spatial-gradient matrix **H** from pre-gathered window
/// samples.
///
/// Computes:
/// * `h00 = Σ Ix²`
/// * `h01 = Σ Ix · Iy`
/// * `h11 = Σ Iy²`
///
/// summed over all `n` elements of the tracking window.
///
/// `ix_win` and `iy_win` must have equal lengths (= window area).
///
/// # Performance
///
/// Wraps the accumulation loop inside `pulp::Arch::dispatch`, enabling LLVM
/// auto-vectorisation with the best available SIMD ISA at runtime.
#[cfg(feature = "simd")]
pub(crate) fn simd_lk_accumulate_h(ix_win: &[f32], iy_win: &[f32]) -> (f64, f64, f64) {
    debug_assert_eq!(ix_win.len(), iy_win.len());

    let mut h00 = 0.0f64;
    let mut h01 = 0.0f64;
    let mut h11 = 0.0f64;

    let arch = pulp::Arch::new();
    arch.dispatch(|| {
        for (&ix, &iy) in ix_win.iter().zip(iy_win.iter()) {
            let ix = ix as f64;
            let iy = iy as f64;
            h00 += ix * ix;
            h01 += ix * iy;
            h11 += iy * iy;
        }
    });

    (h00, h01, h11)
}

// ---------------------------------------------------------------------------
// Mismatch vector accumulation
// ---------------------------------------------------------------------------

/// Accumulate the LK mismatch vector **b** from pre-gathered window samples.
///
/// Computes:
/// * `bx = −Σ Ix · (I2 − I1)`
/// * `by = −Σ Iy · (I2 − I1)`
///
/// where `I1` is the reference-frame patch and `I2` is the current estimate
/// of the tracked patch in the next frame.
///
/// All four slices must have equal lengths (= window area).
///
/// # Performance
///
/// Wraps the accumulation loop inside `pulp::Arch::dispatch`, enabling LLVM
/// auto-vectorisation with the best available SIMD ISA at runtime.
#[cfg(feature = "simd")]
pub(crate) fn simd_lk_accumulate_mismatch(
    ix_win: &[f32],
    iy_win: &[f32],
    i1_win: &[f32],
    i2_win: &[f32],
) -> (f64, f64) {
    debug_assert_eq!(ix_win.len(), iy_win.len());
    debug_assert_eq!(ix_win.len(), i1_win.len());
    debug_assert_eq!(ix_win.len(), i2_win.len());

    let mut bx = 0.0f64;
    let mut by = 0.0f64;

    let arch = pulp::Arch::new();
    arch.dispatch(|| {
        for ((&ix, &iy), (&i1, &i2)) in ix_win
            .iter()
            .zip(iy_win.iter())
            .zip(i1_win.iter().zip(i2_win.iter()))
        {
            let ix = ix as f64;
            let iy = iy as f64;
            let it = i2 as f64 - i1 as f64;
            bx -= ix * it;
            by -= iy * it;
        }
    });

    (bx, by)
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    #[cfg(feature = "simd")]
    use super::*;

    #[cfg(feature = "simd")]
    #[test]
    fn test_accumulate_h_known_values() {
        // ix = [1, 2, 3], iy = [4, 5, 6]
        // h00 = 1+4+9 = 14, h01 = 4+10+18 = 32, h11 = 16+25+36 = 77
        let ix = [1.0f32, 2.0, 3.0];
        let iy = [4.0f32, 5.0, 6.0];
        let (h00, h01, h11) = simd_lk_accumulate_h(&ix, &iy);
        assert!((h00 - 14.0).abs() < 1e-9);
        assert!((h01 - 32.0).abs() < 1e-9);
        assert!((h11 - 77.0).abs() < 1e-9);
    }

    #[cfg(feature = "simd")]
    #[test]
    fn test_accumulate_mismatch_known_values() {
        // ix=[1,1], iy=[0,0], i1=[10,20], i2=[12,22]
        // it=[2,2], bx = -(1*2 + 1*2) = -4, by = -(0*2 + 0*2) = 0
        let ix = [1.0f32, 1.0];
        let iy = [0.0f32, 0.0];
        let i1 = [10.0f32, 20.0];
        let i2 = [12.0f32, 22.0];
        let (bx, by) = simd_lk_accumulate_mismatch(&ix, &iy, &i1, &i2);
        assert!((bx - (-4.0)).abs() < 1e-9);
        assert!((by - 0.0).abs() < 1e-9);
    }
}
