#!/usr/bin/env python3
"""
QUICK REFERENCE: Running thesis_pipeline_v7.py on Google Colab

This is the fastest way to get LLM-based topic representation working.
"""

# ============================================
# COLAB NOTEBOOK TEMPLATE
# ============================================

# CELL 1: Install dependencies (run once)
"""
!pip install bertopic sentence-transformers umap-learn hdbscan PyMuPDF nltk pyvis -q
"""

# CELL 2: Mount Drive and run pipeline
"""
import os
os.chdir('/content')

# Copy script from Drive
import shutil
shutil.copy('/content/drive/MyDrive/Speciale/thesis_pipeline_v7.py', '/content/pipeline.py')

# Run it
exec(open('pipeline.py').read())
"""

# ============================================
# KEY ENVIRONMENT VARIABLES
# ============================================

# The script auto-detects Colab and:
# 1. Mounts Google Drive at /content/drive
# 2. Uses GPU device (default: T4)
# 3. Looks for data at: /content/drive/MyDrive/Speciale/Thesis_Data_Mining
# 4. Saves outputs same location

# ============================================
# EXPECTED RUNTIME
# ============================================

# T4 GPU:   ~15–20 min
# A100 GPU: ~8–10 min
# CPU only: ❌ OOM (don't use)

# ============================================
# FILE STRUCTURE ON DRIVE
# ============================================

"""
/My Drive/
└── Speciale/
    ├── thesis_pipeline_v7.py (upload here)
    ├── COLAB_SETUP.md (instructions)
    └── Thesis_Data_Mining/
        ├── 04_Analysis_Outputs/ (outputs saved here)
        ├── Level_1_General_AI/
        ├── Level_2_Education/
        └── Level_3_Higher_ed/
"""

# ============================================
# V7 vs V6 COMPARISON
# ============================================

"""
ASPECT              V6 (KeyBERTInspired)       V7 (distilgpt2)
─────────────────────────────────────────────────────────────
Representation      Embeddings → Keywords      LLM → Descriptions
GPU Required        No                         Yes
Runtime (local)     ~5 min (CPU)              💥 OOM
Runtime (Colab)     ~25 min (no optimization) ~15 min (optimized)
Reproducibility     Deterministic             Deterministic (seeded)
Methodology         Embedding-based           LLM-based
Thesis Bonus        Less novel                More novel feature
"""

# ============================================
# AFTER EXECUTION
# ============================================

"""
Check /content/drive/MyDrive/Speciale/Thesis_Data_Mining/04_Analysis_Outputs/

Generated files:
✓ topics.csv (with LLM-generated topic labels)
✓ corpus.csv
✓ knowledge_graph.html
✓ barchart_overall_topics.html
✓ topics_by_company.html
✓ topics_by_hierarchy_level.html
✓ appendix_corpus_metadata.csv

All automatically sync to your Drive!
"""
