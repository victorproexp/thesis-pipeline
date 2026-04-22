import pandas as pd
import numpy as np

df = pd.read_csv('Thesis_Data_Mining/04_Analysis_Outputs/corpus_with_topics.csv')

print('=== CORPUS STATS ===')
print(f'Total docs: {len(df)}')
for c in sorted(df['Company'].unique()):
    cd = df[df['Company']==c]
    print(f'  {c}: {len(cd)} docs, {cd["Word_Count"].sum():,} words')
print(f'Total words: {df["Word_Count"].sum():,}')

print('\n=== TOPIC DISTRIBUTION PER COMPANY ===')
print(pd.crosstab(df['Company'], df['Topic']))

print('\n=== TOPIC DISTRIBUTION PER LEVEL ===')
print(pd.crosstab(df['Level'], df['Topic']))

print('\n=== TOPICS ===')
topics = pd.read_csv('Thesis_Data_Mining/04_Analysis_Outputs/topics.csv')
for _, r in topics.iterrows():
    print(f'  Topic {r.Topic} ({r.Count} docs): {r.Representation[:100]}')

print('\n=== MICROSOFT L1 DOCS ===')
ms_l1 = df[(df['Company']=='Microsoft') & (df['Level']=='Level_1_General_AI')]
for _, r in ms_l1.iterrows():
    print(f'  {r.FileName[:70]:70} | Topic {r.Topic} | {r.Word_Count:>6} words')

print('\n=== NOISE DOCS ===')
noise = df[df['Topic']==-1]
if noise.empty:
    print('  None!')
else:
    for _, r in noise.iterrows():
        print(f'  {r.Company:12} | {r.FileName[:55]:55} | {r.Word_Count} words')

print('\n=== TOPIC PROBABILITIES (avg per company) ===')
prob = pd.read_csv('Thesis_Data_Mining/04_Analysis_Outputs/topic_probability_by_company.csv')
for c in sorted(prob['Company'].unique()):
    cp = prob[prob['Company']==c]
    s = ', '.join(f"T{int(r['Topic'])}={r['Avg_Probability']:.2%}" for _, r in cp.iterrows())
    print(f'  {c}: {s}')

print('\n=== VALUES TOP 5 PER COMPANY ===')
vals = pd.read_csv('Thesis_Data_Mining/04_Analysis_Outputs/values_term_frequency.csv')
for c in sorted(vals['Company'].unique()):
    top = vals[vals['Company']==c].nlargest(5, 'Per1kTokens')
    s = ', '.join(f"{r['Term']}({r['Per1kTokens']})" for _, r in top.iterrows())
    print(f'  {c}: {s}')

print('\n=== EXPECTATIONAL GAPS ===')
gaps = pd.read_csv('Thesis_Data_Mining/04_Analysis_Outputs/expectational_gaps.csv')
for c in sorted(gaps['Company'].unique()):
    terms = ', '.join(gaps[gaps['Company']==c]['Absent_Term'].tolist())
    print(f'  {c}: {terms}')

# Compare before/after
print('\n=== BEFORE vs AFTER (reference: 40 docs, 170,615 words) ===')
print(f'  Now: {len(df)} docs, {df["Word_Count"].sum():,} words')
print(f'  Delta: +{len(df)-40} docs, +{df["Word_Count"].sum()-170615:,} words')
