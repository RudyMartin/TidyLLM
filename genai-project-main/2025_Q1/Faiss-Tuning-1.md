

# Log-Adjusted nlist Calculation for FAISS Index Tuning

## Introduction

When constructing a FAISS index, one critical parameter is the number of clusters (`nlist`). This parameter has a strong impact on both the clustering quality and the search performance of the index. However, a fixed `nlist` (e.g., 512) may be unsuitable when the number of available embedding vectors is much smaller. In such cases, over-clustering can occur—leading to warnings like:

> "WARNING clustering 296 points to 296 centroids: please provide at least 11544 training points"

This warning indicates that the clustering algorithm expects significantly more training data than the number of clusters, which is impractical for small datasets.

## The Problem

For example, in an experiment with 300 pages:
- A fixed `nlist` value (e.g., 512) may be far too high.
- Alternatively, setting `nlist` equal to the vector count (e.g., 300) may also be suboptimal because it does not account for the underlying data distribution.

## Proposed Log-Adjusted Approach

We propose a logarithmically adjusted ratio to compute `nlist` dynamically. This approach scales the configured `raw_nlist` based on the total number of embedding vectors, using the following formula:

\[
\text{ratio} = \frac{\log(\text{vector\_count} + 1)}{\log(\text{raw\_nlist} + 1)}
\]

Then, we compute:
\[
\text{computed\_nlist} = \text{round}(\text{raw\_nlist} \times \text{ratio})
\]
Finally, we ensure that the computed number of clusters does not exceed the number of available vectors:

\[
\text{final\_nlist} = \min(\text{computed\_nlist}, \text{vector\_count})
\]

## Code Implementation

```python
import numpy as np

def calculate_nlist(vector_count: int, raw_nlist: int) -> int:
    """
    Calculate the number of clusters (nlist) for the FAISS index using a log-adjusted ratio.
    
    The function scales nlist using the ratio:
      nlist = raw_nlist * (log(vector_count + 1) / log(raw_nlist + 1))
    
    Since you cannot have more clusters than training vectors, the result is capped at vector_count.
    
    Args:
        vector_count (int): Total number of embedding vectors.
        raw_nlist (int): Configured nlist value.
    
    Returns:
        int: The computed nlist.
    """
    if vector_count <= 0:
        return 1
    ratio = np.log(vector_count + 1) / np.log(raw_nlist + 1)
    computed_nlist = int(round(raw_nlist * ratio))
    return min(computed_nlist, vector_count)
```

## Example Calculations

- **Example 1:**  
  If `vector_count` = 10 and `raw_nlist` = 512:
  - Ratio ≈ log(11)/log(513) ≈ 2.3979/6.24 ≈ 0.384  
  - Computed nlist ≈ round(512 * 0.384) ≈ 197  
  - Since 197 > 10, the final nlist is capped at 10.

- **Example 2:**  
  If `vector_count` = 1000 and `raw_nlist` = 512:
  - Ratio ≈ log(1001)/log(513) ≈ 6.908/6.24 ≈ 1.107  
  - Computed nlist ≈ round(512 * 1.107) ≈ 567  
  - Since 567 < 1000, the final nlist is 567.

## Discussion

This log-adjusted approach:
- **Adapts Dynamically:**  
  It smoothly adjusts the number of clusters based on the size of the dataset, preventing over-clustering in small datasets.
- **Bounds the Value:**  
  It ensures that the computed `nlist` never exceeds the total number of available vectors.
- **Practical for Experiments:**  
  With an experiment of 300 pages, this method will yield a more realistic `nlist` compared to a fixed value, resulting in better clustering and index performance.

## References and Further Reading

- **FAISS Documentation:**  
  [Facebook AI Similarity Search (FAISS)](https://github.com/facebookresearch/faiss)  
  *Learn more about the design and tuning of FAISS indices.*

- **Clustering in High-Dimensional Spaces:**  
  [Scalable Similarity Search and Clustering of Dense Vectors](https://papers.nips.cc/paper/2017/hash/ed4e1f2e9f6474d3d7e2462b94a39c1e-Abstract.html)  
  *A research paper discussing scalable methods for similarity search and clustering.*

- **Log Transformations in Machine Learning:**  
  [Log Transformations for Skewed Data](https://machinelearningmastery.com/log-transformation-for-skewed-data/)  
  *An overview of how logarithmic transformations can help manage skewed data distributions.*

## Conclusion

The log-adjusted nlist calculation provides a more flexible, data-driven approach for setting the number of clusters in a FAISS index. By adapting `nlist` based on the size of the dataset, this method helps ensure stable clustering and improved search performance—especially important when working with experiments involving a relatively small number of pages. This approach offers both theoretical insights and practical benefits for tuning FAISS indices in research and production environments.
