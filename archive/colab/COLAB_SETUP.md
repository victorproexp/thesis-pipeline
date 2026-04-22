# Running thesis_pipeline_v7.py on Google Colab

## Quick Start

### Step 1: Prepare Your Drive
- Ensure your `Thesis_Data_Mining/` folder is in `/My Drive/Speciale/` on Google Drive
- Example structure:
  ```
  /My Drive/Speciale/Thesis_Data_Mining/
  ├── 04_Analysis_Outputs/
  ├── Level_1_General_AI/
  ├── Level_2_Education/
  └── Level_3_Higher_ed/
  ```

### Step 2: Create Colab Notebook
1. Go to [colab.research.google.com](https://colab.research.google.com)
2. Create new notebook
3. Set GPU: **Runtime → Change runtime type → GPU (T4)**

### Step 3: Run Cells

**Cell 1:** Install dependencies
```python
!pip install bertopic sentence-transformers umap-learn hdbscan PyMuPDF nltk pyvis -q
```

**Cell 2:** Download script (option A: from Drive)
```python
import shutil
shutil.copy('/content/drive/MyDrive/Speciale/thesis_pipeline_v7.py', '/content/')
```

Or **(option B: paste directly)**
- Copy entire contents of `thesis_pipeline_v7.py`
- Paste into Colab cell
- Run

**Cell 3:** Execute pipeline
```python
exec(open('thesis_pipeline_v7.py').read())
```

## What Happens

1. **Auto-detection**: Script detects Colab environment and:
   - Mounts Google Drive at `/content/drive`
   - Uses GPU device (T4 or better)
   - Loads data from Drive path: `/content/drive/MyDrive/Speciale/Thesis_Data_Mining`

2. **Processing** (~15-20 min on T4 GPU):
   - Extracts PDFs (45 documents)
   - Generates embeddings (all-mpnet-base-v2)
   - UMAP + HDBSCAN clustering
   - **LLM-based representation**: Uses distilgpt2 to generate topic labels
   - Generates visualizations

3. **Output**: All files saved to Drive at:
   - `/content/drive/MyDrive/Speciale/Thesis_Data_Mining/04_Analysis_Outputs/`
   - `topics.csv` (with LLM-generated labels)
   - `knowledge_graph.html`
   - `barchart_overall_topics.html`
   - `topics_by_company.html`
   - `topics_by_hierarchy_level.html`
   - `corpus.csv`
   - `appendix_corpus_metadata.csv`

## Troubleshooting

| Issue | Solution |
|-------|----------|
| `ModuleNotFoundError: bertopic` | Re-run Cell 1 pip install |
| `FileNotFoundError: Thesis_Data_Mining` | Check Drive path and folder permissions |
| `CUDA out of memory` | Switch to T4 GPU (should have enough VRAM) |
| `Output files not saved` | Check `/content/drive/MyDrive/Speciale/Thesis_Data_Mining/04_Analysis_Outputs/` |

## Key Differences from v6

| Feature | v6 | v7 |
|---------|----|----|
| Representation | KeyBERTInspired (embeddings) | distilgpt2 (LLM) |
| Reproducibility | Deterministic | Deterministic (fixed seed) |
| Topic labels | Keyword-based | Generated descriptions |
| Local compatibility | Yes (CPU) | No (requires GPU) |
| Colab compatibility | Not optimized | Optimized ✓ |

## Expected Runtime

- **T4 GPU**: ~15-20 minutes total
- **A100 GPU**: ~8-10 minutes
- **CPU (not recommended)**: Would hit OOM

## Notes

- First run downloads ~800MB of models (cached after)
- Colab session times out after 12 hours; code completes well before then
- You can download output files directly from Colab or sync via Drive
- Google Drive auto-syncs so outputs are immediately available on your Mac
