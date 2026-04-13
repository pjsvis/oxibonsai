//! 1-bit Linear layer using Q1\_0\_g128 weights.
//!
//! Wraps the kernel GEMV/GEMM operations with a layer abstraction.

use oxibonsai_core::tensor::BlockQ1_0G128;
use oxibonsai_kernels::traits::OneBitKernel;
use oxibonsai_kernels::GpuWeightHandle;

use crate::error::ModelResult;

/// A linear layer with Q1\_0\_g128 (1-bit) weights.
///
/// Computes `output = weights @ input` (without bias — Qwen3 has no bias).
#[derive(Debug)]
pub struct Linear1Bit<'a> {
    /// Weight blocks in row-major order: [out_features × (in_features / 128)] blocks.
    blocks: &'a [BlockQ1_0G128],
    /// Number of output features (rows).
    out_features: usize,
    /// Number of input features (columns, must be multiple of 128).
    in_features: usize,
    /// GPU-resident weight handle, populated after [`upload_to_gpu()`](Self::upload_to_gpu).
    gpu_handle: Option<GpuWeightHandle>,
}

impl<'a> Linear1Bit<'a> {
    /// Create a 1-bit linear layer.
    ///
    /// - `blocks`: Q1\_0\_g128 weight blocks in row-major order.
    /// - `out_features`: Number of output features.
    /// - `in_features`: Number of input features (must be multiple of 128).
    pub fn new(blocks: &'a [BlockQ1_0G128], out_features: usize, in_features: usize) -> Self {
        debug_assert_eq!(in_features % 128, 0);
        debug_assert_eq!(blocks.len(), out_features * (in_features / 128));
        Self {
            blocks,
            out_features,
            in_features,
            gpu_handle: None,
        }
    }

    /// Number of output features (rows).
    pub fn out_features(&self) -> usize {
        self.out_features
    }

    /// Raw block references (for fused weight concatenation).
    pub fn blocks(&self) -> &[BlockQ1_0G128] {
        self.blocks
    }

    /// Access the GPU-resident weight handle, if uploaded.
    pub fn gpu_handle(&self) -> Option<GpuWeightHandle> {
        self.gpu_handle
    }

    /// Upload weights to GPU memory if the kernel tier supports caching.
    ///
    /// After a successful upload, all subsequent [`forward_vec`](Self::forward_vec)
    /// calls will use the GPU-resident buffer instead of copying weights
    /// every time.
    pub fn upload_to_gpu(&mut self, kernel: &dyn OneBitKernel) {
        self.gpu_handle = kernel.upload_weights(self.blocks);
    }

    /// Forward pass: vector input (GEMV).
    ///
    /// - `input`: FP32 vector of length `in_features`.
    /// - `output`: FP32 vector of length `out_features`.
    /// - `kernel`: Kernel implementation to use.
    pub fn forward_vec(
        &self,
        input: &[f32],
        output: &mut [f32],
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<()> {
        // Try the cached GPU path first (no host→device weight copy).
        if let Some(handle) = self.gpu_handle {
            if kernel
                .gemv_cached(handle, input, output, self.out_features, self.in_features)
                .is_ok()
            {
                return Ok(());
            }
        }
        // Fallback to the regular (uncached) GEMV.
        kernel.gemv(
            self.blocks,
            input,
            output,
            self.out_features,
            self.in_features,
        )?;
        Ok(())
    }

    /// Forward pass: matrix input (GEMM) for batched/prefill operation.
    ///
    /// - `input`: Row-major FP32 matrix [m × in_features].
    /// - `output`: Row-major FP32 matrix [m × out_features].
    /// - `m`: Batch/sequence dimension.
    /// - `kernel`: Kernel implementation to use.
    pub fn forward_mat(
        &self,
        input: &[f32],
        output: &mut [f32],
        m: usize,
        kernel: &dyn OneBitKernel,
    ) -> ModelResult<()> {
        kernel.gemm(
            self.blocks,
            input,
            output,
            m,
            self.out_features,
            self.in_features,
        )?;
        Ok(())
    }

    /// Input dimension.
    pub fn in_features(&self) -> usize {
        self.in_features
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use half::f16;
    use oxibonsai_kernels::KernelDispatcher;

    fn make_block(scale: f32, bits: [u8; 16]) -> BlockQ1_0G128 {
        BlockQ1_0G128 {
            d: f16::from_f32(scale),
            qs: bits,
        }
    }

    #[test]
    fn linear_1bit_gemv() {
        // 2 output features, 128 input features
        let blocks = vec![
            make_block(1.0, [0xFF; 16]), // row 0: all +1
            make_block(1.0, [0x00; 16]), // row 1: all -1
        ];
        let layer = Linear1Bit::new(&blocks, 2, 128);
        let kernel = KernelDispatcher::auto_detect();

        let input = vec![1.0f32; 128];
        let mut output = vec![0.0f32; 2];
        layer
            .forward_vec(&input, &mut output, &kernel)
            .expect("linear forward should succeed");

        assert!((output[0] - 128.0).abs() < 1.0);
        assert!((output[1] + 128.0).abs() < 1.0);
    }
}
