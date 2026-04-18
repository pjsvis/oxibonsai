//! # CudaGraph - reformat_tq2_blocks_to_soa_group Methods
//!
//! This module contains method implementations for `CudaGraph`.
//!
//! 🤖 Generated with [SplitRS](https://github.com/cool-japan/splitrs)

use std::sync::Arc;

use super::types::CudaGraphError;

use super::cudagraph_type::CudaGraph;

impl CudaGraph {
    /// Reformat TQ2_0_g128 blocks (`qs: [u8;32]`, `d: f16`) to SoA bytes:
    /// `[N×2 bytes FP16 scales LE][N×32 bytes qs]`.
    fn reformat_tq2_blocks_to_soa(blocks: &[oxibonsai_core::BlockTQ2_0_g128]) -> Vec<u8> {
        let n = blocks.len();
        let mut soa = Vec::with_capacity(n * 34);
        for block in blocks {
            let bits = block.d.to_bits().to_le_bytes();
            soa.push(bits[0]);
            soa.push(bits[1]);
        }
        for block in blocks {
            soa.extend_from_slice(&block.qs);
        }
        soa
    }
    /// Upload TQ2_0_g128 weights in SoA layout under a new handle id.
    pub fn upload_weight_tq2_soa(
        &self,
        handle_id: u64,
        blocks: &[oxibonsai_core::BlockTQ2_0_g128],
    ) -> Result<(), CudaGraphError> {
        let soa = Self::reformat_tq2_blocks_to_soa(blocks);
        let d_weight = self
            .stream
            .clone_htod(&soa)
            .map_err(|e| CudaGraphError::DriverError(format!("clone_htod tq2: {e}")))?;
        let mut cache = self
            .weight_cache
            .lock()
            .map_err(|_| CudaGraphError::LockPoisoned)?;
        cache.insert(handle_id, Arc::new(d_weight));
        Ok(())
    }
}
