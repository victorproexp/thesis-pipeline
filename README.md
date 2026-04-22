# Thesis Pipeline: AI in Education (Sociotechnical Imaginaries)

## Quick Start

```bash
python run_pipeline.py
```

This executes the main thesis pipeline (`thesis_pipeline_v6.py`), which:
1. Mines PDFs from `Thesis_Data_Mining/Level_*_*/Company_*/` folders
2. Preprocesses text (tokenization, lemmatization, normalization)
3. Fits BERTopic model with embeddings (all-mpnet-base-v2)
4. Generates visualizations and knowledge graph
5. Exports outputs to `Thesis_Data_Mining/04_Analysis_Outputs/`

## Environment

```bash
source thesis_env/bin/activate
python run_pipeline.py
```

## Outputs

Located in `Thesis_Data_Mining/04_Analysis_Outputs/`:
- `topics.csv` — Topic IDs, word counts, topic keywords
- `corpus_with_topics.csv` — All documents with assigned topics and probabilities
- `values_term_frequency.csv` — Per-company normalized frequencies for democratic values terms
- `knowledge_graph.html` — Interactive network visualization (company ↔ topic)
- `barchart_overall_topics.html` — Overall topic distribution
- `topics_by_company.html` — Topics per company
- `topics_by_hierarchy_level.html` — Topics per organizational level

## File Structure

```
Speciale/
├── README.md                                  # This file
├── run_pipeline.py                            # Entry point
├── thesis_pipeline_v6.py                      # Active pipeline ⭐
├── archive/                                   # Archived experiments and legacy material
│   ├── pipeline_versions/                     # thesis_pipeline.py and v2-v5, v7-v8
│   ├── debug_tools/                           # One-off inspection/debug scripts
│   ├── colab/                                 # Colab experiments and outputs
│   ├── docs/                                  # Archived notes, PDFs, methodology drafts
│   └── runtime/                               # Old env/cache/log artifacts
│
├── Excluded_Documents_Low_Relevance/          # Documents screening out (all levels)
│   └── [PDFs not included in final corpus]
│
├── description/                               # Thesis planning and project documents
│   └── *.pdf
│
├── literature/                                # References & papers
├── lib/                                       # Dependencies (vis.js, tom-select, bindings)
├── thesis_env/                                # Python environment
└── Thesis_Data_Mining/                        # Corpus and analysis outputs
    ├── 04_Analysis_Outputs/                   # Pipeline outputs (topics, graphs, etc.)
    ├── Level_1_General_AI/
    ├── Level_2_Education/
    └── Level_3_Higher_ed/
```

## Version History

**v6** (current): Stable embedding-based BERTopic with KeyBERTInspired representation. Current run produces 5 topics plus 2 noise documents, along with audit-oriented corpus documentation.

**v7–v8** (archived): Experimental LLM-based representation with distilgpt2. Moved to archive for comparison.

**v2–v5** (archived): Early iterations. See `archive/` for methodology evolution.

## Troubleshooting

- **"No data found"**: Check that PDFs exist in `Thesis_Data_Mining/Level_*_*/Company_*/`
- **Out of memory**: Reduce `n_neighbors` or `doc_length` in the pipeline
- **GPU not detected**: Pipeline falls back to CPU gracefully
# thesis-pipeline
