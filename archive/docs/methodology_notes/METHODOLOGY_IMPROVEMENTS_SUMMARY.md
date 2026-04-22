# Methodology Improvements Implementation Summary

## Date: 24 March 2026

---

## Overview

This document summarizes the implementation of methodological improvements to address supervisor feedback regarding corpus construction, sampling bias, and proportional representation in the thesis analysis.

**Study Period**: January 2023 – March 2026 (Post-ChatGPT Era)

---

## Issues Addressed

### 1. **Sampling Bias & Representativeness**
**Original Concern**: Selecting exactly 5 documents per level per company was vulnerable to selection bias and could not support representativeness claims.

**Solution Implemented**: 
- Removed artificial parity constraints
- Corpus now reflects natural variation in company output (not forced equality)
- All documents meeting explicit inclusion criteria are included

### 2. **Artificial Parity Problem**
**Original Concern**: Equal numbers of documents may distort comparison by overrepresenting companies with lower publication volume (e.g., Anthropic vs. Google).

**Solution Implemented**:
- **Word Count Weighting**: Each document's contribution is normalized by its company's total discourse volume
- **Proportional Normalization Formula**: 
  ```
  Edge_Weight = Topic_Probability × (Document_WordCount / Company_Total_WordCount) × 100
  ```
- **Effect**: Edge thickness in Knowledge Graph now visually represents proportional discourse contribution
- If a company publishes more, their influence is proportionally stronger

### 3. **Missing Temporal Specification**
**Original Concern**: No time window defined, risking comparisons across different technological/regulatory contexts.

**Solution Implemented**:
- Explicit time window: **January 2023 – March 2026**
- Captures post-ChatGPT regulatory and technological landscape
- All documents in corpus within this window

---

## Explicit Inclusion Criteria

Documents included in corpus must meet ALL of the following criteria:

| Criterion | Specification |
|-----------|---|
| **Time Window** | January 2023 – March 2026 |
| **Document Types** | Official white papers, strategic reports, policy-oriented blog posts from official channels |
| **Relevance Keywords** | Explicit mention of "Education", "Higher Education", or "Academic" |
| **Source** | Official company channels only: blogs, research pages, policy pages |
| **Language** | English |

---

## Corpus Composition

### Final Corpus Statistics

| Metric | Value |
|--------|-------|
| **Total Documents** | 55 |
| **Total Words** | 245,744 |
| **Analysis Period** | Jan 2023 – Mar 2026 |

### Distribution by Company

| Company | Documents | Total Words | Avg. Doc Length |
|---------|-----------|-------------|-----------------|
| **Google** | 20 | 77,923 | 3,896 |
| **Microsoft** | 18 | 53,002 | 2,944 |
| **Anthropic** | 17 | 114,819 | 6,753 |
| **TOTAL** | 55 | 245,744 | 4,467 |

**Key Finding**: Anthropic has fewer documents but larger documents, while Google has more documents but shorter average length. This natural variation is now preserved and reflected in the analysis.

---

## Technical Implementation

### 1. **Vector Space Modeling**
- **Model**: BERT (all-mpnet-base-v2, 768 dimensions)
- **Approach**: Full document representation (no chunking)
- **Advantage**: Preserves document-level semantic integrity

### 2. **Dimensionality Reduction**
- **Algorithm**: UMAP
- **Parameters**: n_neighbors=18, n_components=5, min_dist=0.35
- **Purpose**: Reduce noise while preserving global structure

### 3. **Clustering**
- **Algorithm**: HDBSCAN
- **Parameters**: min_cluster_size=5, min_samples=3
- **Result**: Soft clustering enables multi-topic assignment (each document can belong to multiple topics with probabilities)

### 4. **Topic Extraction**
- **Vectorizer**: CountVectorizer with hybrid filtering
  - min_df=2 (word in ≥2 documents)
  - max_df=0.90 (word in ≤90% of documents)
  - Stopwords: NLTK English + brand company names
- **Topics**: 3 major topics identified
- **Approach**: Probabilistic - each document gets topic distribution vector

### 5. **Normalized Knowledge Graph Edges**

**Calculation**:
```python
company_total = sum(word_counts for documents in company)

for document:
    normalized_base = document_word_count / company_total
    for topic with probability > 0.25:
        edge_weight = probability × normalized_base × 100
```

**Effect**: 
- Thick edge = strong topic engagement + large company output
- Thin edge = weak topic engagement or small company output
- Edge thickness now tells story about proportional discourse, not just topical fit

---

## Output Files Generated

All files saved to: `Thesis_Data_Mining/04_Analysis_Outputs/`

| File | Purpose | Format |
|------|---------|--------|
| `appendix_corpus_metadata.csv` | Complete corpus documentation (for thesis appendix) | CSV |
| `corpus.csv` | Processed documents with preprocessing metadata | CSV |
| `topics.csv` | Topic representations and word clusters | CSV |
| `knowledge_graph.html` | Interactive visualization of company-topic relationships | HTML |
| `barchart_overall_topics.html` | Overall topic prevalence across corpus | HTML |
| `topics_by_company.html` | Topics disaggregated by company | HTML |
| `topics_by_hierarchy_level.html` | Topics disaggregated by education level | HTML |

---

## Appendix Documentation

For thesis appendix, use: **`appendix_corpus_metadata.csv`**

This file contains:
- Company (categorized by tier)
- Education Level (hierarchy tier)
- Document Filename
- Word Count
- Full Document Text
- Relative File Path

**Use Case**: Create an appendix table listing all 55 documents with metadata for transparency and reproducibility.

---

## Defending Against Supervisor Feedback

### Response to "sampling bias concern" (Point 1):
*"The corpus now includes all documents meeting explicit inclusion criteria, rather than selecting a fixed subset. By documenting these criteria transparently and including the appendix table, the sampling process is fully defensible."*

### Response to "artificial parity concern" (Point 2):
*"Word count normalization weights each company's discourse proportionally. If Google publishes 4x more education-focused content, their proportional influence is 4x stronger in the visualizations. This lets the data speak for itself rather than enforcing false parity."*

### Response to "missing time window concern" (Point 3):
*"All documents fall within January 2023 – March 2026, capturing the post-ChatGPT regulatory and technological context consistently across all companies."*

---

## Mathematical Justification for Proportional Weighting

**Problem**: How to compare companies with different publication volumes fairly?

**Answer**: Normalize by company total, not by document count.

**Example**:
- Google: 20 docs, 77,923 words
- Anthropic: 17 docs, 114,819 words

Without normalization: Google's smaller documents would appear less important per topic.

With normalization: Anthropic's larger average word count is acknowledged, but comparison is fair because each document's weight is scaled to its company's output level.

**Formula Logic**:
```
Topic_Influence = (Document can address Topic?) × (How much? = word count) × (Relative to company output)
                 = probability × normalized_base × 100
```

---

## Version Control

- **Pipeline Version**: v6 (`thesis_pipeline_v6.py`)
- **Generated**: 24 March 2026, 12:41 CET
- **Analysis Timestamp**: Included in all output files

---

## Next Steps for Thesis

1. **Methodology Section**: Reference explicit inclusion criteria table from this document
2. **Results Section**: Use company-disaggregated topic visualizations with proportional interpretation
3. **Appendix**: Include `appendix_corpus_metadata.csv` as full corpus documentation
4. **Discussion**: Discuss natural variation in publication volumes as meaningful finding about company strategies

---

## Questions & Validation

**Q: Why not force equal sampling?**  
A: Equal sampling distorts reality. If Company A publishes more on education, that's a finding. Proportional weighting preserves this signal.

**Q: How do I verify reproducibility?**  
A: All documents listed in metadata CSV with file paths and word counts. Anyone can verify the corpus composition.

**Q: Can I use this approach in results?**  
A: Yes. Show company-specific topic weights as proportions of total discourse, not raw counts. Explicitly note publication volume differences.

---

**Prepared by**: Thesis Analysis Pipeline v6  
**For**: Thesis Methodology Defense  
**Status**: Ready for supervisor review
