//! In-memory flat vector store with cosine similarity search.
//!
//! The [`VectorStore`] holds a flat list of [`VectorEntry`] items.  Search is
//! performed with a brute-force linear scan over all entries, computing cosine
//! similarity against the query vector.  This is appropriate for corpora up to
//! tens of thousands of chunks; larger corpora benefit from approximate nearest
//! neighbour indices (out of scope for this crate).

use serde::{Deserialize, Serialize};

use crate::chunker::Chunk;
use crate::error::RagError;

// ─────────────────────────────────────────────────────────────────────────────
// Math primitives
// ─────────────────────────────────────────────────────────────────────────────

/// Compute the dot product of two equal-length slices.
///
/// Returns 0.0 if either slice is empty or they have different lengths.
#[inline]
pub fn dot_product(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2-normalise `v` in place.
///
/// If the Euclidean norm is smaller than `1e-10` the vector is left unchanged
/// to prevent NaN propagation.
#[inline]
pub fn l2_normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

/// Cosine similarity between two equal-length vectors.
///
/// Both vectors are assumed to be *unit vectors* (L2-normalised).  Under that
/// assumption, cosine similarity == dot product and we skip the denominator
/// computation entirely.
///
/// Returns a value in `[-1.0, 1.0]`.  Returns `0.0` for empty or mismatched
/// inputs rather than panicking.
#[inline]
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.is_empty() || a.len() != b.len() {
        return 0.0;
    }
    // Fast path: if both are unit vectors, dot product == cosine similarity.
    // We always normalise vectors before storing, so this is safe.
    dot_product(a, b).clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────────────────────────────────────
// VectorEntry & SearchResult
// ─────────────────────────────────────────────────────────────────────────────

/// A single indexed entry in the vector store.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VectorEntry {
    /// Unique identifier assigned at insertion time.
    pub id: usize,
    /// L2-normalised embedding vector.
    pub vector: Vec<f32>,
    /// The chunk this entry was derived from.
    pub chunk: Chunk,
}

/// A result returned by a similarity search.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Cosine similarity score in `[-1.0, 1.0]`.
    pub score: f32,
    /// The chunk associated with this result.
    pub chunk: Chunk,
    /// The entry's unique identifier in the store.
    pub id: usize,
}

// ─────────────────────────────────────────────────────────────────────────────
// VectorStore
// ─────────────────────────────────────────────────────────────────────────────

/// In-memory flat vector store backed by a `Vec<VectorEntry>`.
///
/// All inserted vectors are L2-normalised so that cosine similarity reduces to
/// a plain dot product at query time.
#[derive(Debug, Default, Serialize, Deserialize)]
pub struct VectorStore {
    entries: Vec<VectorEntry>,
    dim: usize,
}

impl VectorStore {
    /// Create an empty store that accepts vectors of dimensionality `dim`.
    pub fn new(dim: usize) -> Self {
        Self {
            entries: Vec::new(),
            dim,
        }
    }

    /// Insert a vector+chunk pair into the store.
    ///
    /// The vector is L2-normalised before storage.  Returns the assigned entry
    /// id, or [`RagError::DimensionMismatch`] if the vector has the wrong size.
    pub fn insert(&mut self, mut vector: Vec<f32>, chunk: Chunk) -> Result<usize, RagError> {
        if vector.len() != self.dim {
            return Err(RagError::DimensionMismatch {
                expected: self.dim,
                got: vector.len(),
            });
        }
        l2_normalize(&mut vector);
        let id = self.entries.len();
        self.entries.push(VectorEntry { id, vector, chunk });
        Ok(id)
    }

    /// Return the top-`top_k` entries by cosine similarity to `query`.
    ///
    /// The query vector is normalised internally; it is not mutated.
    /// Results are returned in descending score order.
    pub fn search(&self, query: &[f32], top_k: usize) -> Vec<SearchResult> {
        self.search_with_threshold(query, top_k, f32::NEG_INFINITY)
    }

    /// Like [`Self::search`] but discards results whose score is below `min_score`.
    pub fn search_with_threshold(
        &self,
        query: &[f32],
        top_k: usize,
        min_score: f32,
    ) -> Vec<SearchResult> {
        if self.entries.is_empty() || top_k == 0 || query.len() != self.dim {
            return Vec::new();
        }

        // Normalise a local copy of the query
        let mut q = query.to_vec();
        l2_normalize(&mut q);

        // Score every entry
        let mut scored: Vec<(f32, usize)> = self
            .entries
            .iter()
            .map(|e| (cosine_similarity(&q, &e.vector), e.id))
            .filter(|(score, _)| *score >= min_score)
            .collect();

        // Partial-sort: we only need the top-k
        scored.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
        scored.truncate(top_k);

        scored
            .into_iter()
            .map(|(score, id)| SearchResult {
                score,
                chunk: self.entries[id].chunk.clone(),
                id,
            })
            .collect()
    }

    /// Number of entries currently in the store.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns `true` if the store contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Remove all entries from the store (preserves the configured dimension).
    pub fn clear(&mut self) {
        self.entries.clear();
    }

    /// Approximate heap memory used by the stored vectors and chunk texts.
    ///
    /// This is a lower-bound estimate: it counts vector bytes and chunk text
    /// bytes, but does not account for allocator overhead or struct padding.
    pub fn memory_usage_bytes(&self) -> usize {
        self.entries.iter().fold(0usize, |acc, e| {
            acc + e.vector.len() * std::mem::size_of::<f32>()
                + e.chunk.text.len()
                + std::mem::size_of::<VectorEntry>()
        })
    }

    /// The embedding dimensionality this store was constructed with.
    pub fn dim(&self) -> usize {
        self.dim
    }
}
