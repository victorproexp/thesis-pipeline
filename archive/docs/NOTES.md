# Pipeline Version History

## Version Timeline

### v8 (27 Mar - Experimental)
- LLM-based representation with distilgpt2 (GPU optimized)
- Alternative to KeyBERTInspired
- Not converged — kept for reference only

### v7 (26 Mar - Experimental LLM)
- First attempt at LLM-based representation
- Colab-ready with GPU auto-detection
- Distilgpt2 for topic description generation
- Archived: v8 is the improved LLM version

### v6 (27 Mar - Current / Stable)
- **ACTIVE PIPELINE**
- Embedding-based BERTopic with KeyBERTInspired
- 3 stable topics + 2 noise documents
- Curated stopwords (95+ domain-specific terms)
- Values term frequency export for democratic public values analysis
- Proportional edge weighting in knowledge graph
- **Best for thesis**: Combines quantitative topic structure with qualitative values analysis

### v5 (23 Mar - Pre-finalization)
- Tuned UMAP/HDBSCAN parameters
- Added min-max normalization to knowledge graph
- Iterative stopword refinement

### v4 (23 Mar - Parameter exploration)
- Experimented with cluster size adjustments
- Attempted topic recovery via parameter tuning

### v3 (23 Mar - Clustering refinement)
- Focus on HDBSCAN configuration
- Early noise document handling

### v2 (23 Mar - First stable version)
- Initial BERTopic setup
- Basic preprocessing pipeline
- Early visualization attempts

### v1 (Original - Pre-thesis)
- Prototype implementation
- No longer present (too early to archive)

---

## Decision Rationale

**Why v6 is the choice:**

1. **Stability** — 3 topics consistently recover across runs. HDBSCAN parameters are balanced.
2. **Theory alignment** — Topics map to sociotechnical imaginary dimensions:
   - Topic 0: EdTech deployment & learning frameworks
   - Topic 1: Conversational AI & ethical reasoning  
   - Topic 2: AI as engineering/innovation system
3. **Values export** — Captures safety/governance/democratic public values analytically without forcing a fragile 4th topic
4. **Per-company differences** — Knowledge graph and values export show connector vs. complementor positioning:
   - Microsoft: security (2.09‰), governance (1.23‰) → infrastructure connector
   - Anthropic: ethical (0.99‰), oversight (0.41‰) → safety complementor
   - Google: responsible (1.03‰), security (1.09‰) → balanced positioning

---

## Experimentation Lessons

### What didn't work:
- **v7/v8 LLM representation** — Distilgpt2 generated verbose, redundant descriptions. KeyBERTInspired (embedding-based keywords) cleaner and more interpretable.
- **4-topic forcing** — min_cluster_size=3 or nr_topics=4 doubled noise (2→4 docs). Fragile micro-clusters don't add value.
- **Custom brand stopwords alone** — Over-filtering caused 8 docs into noise. Required balanced approach: NLTK stopwords + minimal domain filtering.

### What works:
- **HDBSCAN(min_cluster_size=4, min_samples=2)** — Good balance between sensitivity and stability
- **UMAP(n_neighbors=15, min_dist=0.35)** — Spreads clusters visually without over-fragmenting
- **Min-max normalization in knowledge graph** — Better visual differentiation of document weights within companies
- **Values term frequency export** — Projects corporate positioning onto democratic values framework without forcing topic structure

---

## How to Compare Versions

```bash
cd archive

# Show last 20 lines (key parameters)
tail -20 thesis_pipeline_v5.py
diff thesis_pipeline_v5.py thesis_pipeline_v6.py | head -50
```

Archives preserved for thesis methodology section (showing iterative refinement).
