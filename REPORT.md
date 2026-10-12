# GStar Assignment 1: Approach and Key Learnings

## Overview

I built the attention implementation step by step. I started with the tiled algorithm in PyTorch, moved it to Triton, and then added causal masking, grouped query attention (GQA), sliding windows, and attention sinks. This helped me understand the math before working through the GPU implementation details. The main idea was to compute attention without storing the full attention score or probability matrix.

## Implementation Approach

### Foundations: PyTorch and Triton (Problems 1–3)

In Problem 1, I implemented the forward pass using query and key/value tiles. For each query tile, I kept track of the running maximum, the sum of exponentials, and the weighted sum of values. When a new key tile increased the maximum, I rescaled the previous sums before adding the new tile's contribution. After processing all key tiles, I normalized the output and saved the log-sum-exp statistics. This helped me understand how online softmax works before implementing it on the GPU.

Problem 2 introduced Triton through a weighted row-sum kernel. I assigned one program instance to each row and processed the columns in blocks. I masked loads at the boundary and summed the accumulated products. This exercise helped me learn program IDs, pointer offsets, vectorized loads, and reductions.

In Problem 3, I moved the tiled attention computation to Triton. Each program handles a query block for one batch/head pair and loops over key/value blocks. I used float32 for the softmax statistics and output accumulators. I also used base-2 exponentials and adjusted the score scale accordingly. Processing scores one tile at a time avoids storing the full attention matrix, whose size grows quadratically with sequence length.

### Extending the Forward Pass (Problems 4–7)

For causal attention in Problem 4, I split the computation into earlier key blocks and blocks overlapping the query tile. Earlier blocks need no causal mask. In overlapping blocks, I masked individual scores to prevent queries from attending to future positions. The kernel only visits blocks that can contain valid attention scores.

In Problem 5, I added GQA by mapping each query head to its shared key/value head using integer division by the number of query heads per group. The kernel reads the shared K/V tensors directly, so there is no need to repeat them in memory. Each query head still produces its own output, while multiple query heads share the same K/V storage.

In Problem 6, I limited the key blocks to those within the sliding window and aligned the starting position to a key-tile boundary. Since queries in a tile have different positions, I also masked individual scores based on each query's window. The loop bounds reduce unnecessary work, and the mask keeps the window correct for each query.

In Problem 7, I added attention sinks so that queries can attend to the first few tokens as well as their local window, while still respecting the causal mask. I processed sink tokens separately and excluded them from the later window and diagonal phases. Counting a token twice would change both the softmax denominator and the weighted output.

### Optional Backward Pass (Problem 8)

I coded a backward pass with a preprocessing kernel that computes `delta = sum(O * dO)` for each row, followed by separate kernels for `dQ` and `dK`/`dV`. The backward pass recomputes attention probabilities from Q, K, and the saved normalization statistics. This avoids storing the full probability matrix during the forward pass.

**Problem 8 remains incomplete: the backward implementation has numerical inaccuracies, and its tests do not yet pass.** It is still a work in progress. The next step is to check gradient scaling, precision, and how gradients are mapped and summed across query heads that share a K/V head.

## Key Learnings and Validation

The main lesson was that FlashAttention saves memory by changing how attention is computed. Online softmax combines results from different tiles without keeping all the scores in memory. Although the algorithm is mathematically exact, the implementation still needs careful handling of numerical precision, scaling, and masks.

I also learned that load masks and attention masks serve different purposes. A load mask prevents reading outside a tensor's bounds. An attention mask controls which scores contribute to the softmax. Choosing the right blocks to visit is not always enough; individual scores still need masking near causal and sliding-window boundaries.
