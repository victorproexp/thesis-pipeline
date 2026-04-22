import pandas as pd

# Topics
t = pd.read_csv('Thesis_Data_Mining/04_Analysis_Outputs/topics.csv')
print('=== CURRENT TOPICS ===')
for _, r in t.iterrows():
    print(f'  Topic {r["Topic"]} ({r["Count"]} docs): {r["Representation"]}')

# Per-company breakdown
df = pd.read_csv('Thesis_Data_Mining/04_Analysis_Outputs/corpus_with_topics.csv')
print('\n=== TOPIC DISTRIBUTION PER COMPANY ===')
ct = pd.crosstab(df['Company'], df['Topic'])
print(ct)

# Check which value-laden terms survive in processed text
print('\n=== KEY THEORY TERMS IN CORPUS (term frequency) ===')
all_text = ' '.join(df['Processed_Text'].tolist())
for term in ['governance', 'transparency', 'accountability', 'autonomy', 'privacy',
             'security', 'equity', 'oversight', 'democratic', 'agency', 'ethical',
             'legitimacy', 'public', 'responsible', 'regulation', 'infrastructure',
             'performative', 'imaginary', 'determinism', 'openness', 'egalitarian']:
    count = all_text.split().count(term)
    if count > 0:
        print(f'  {term}: {count}')

# Per-company term frequency for key values
print('\n=== VALUES TERMS PER COMPANY ===')
values_terms = ['governance', 'transparency', 'accountability', 'autonomy', 'privacy',
                'security', 'equity', 'oversight', 'agency', 'responsible', 'ethical',
                'public', 'openness', 'regulation', 'infrastructure']
for company in ['Anthropic', 'Google', 'Microsoft']:
    company_text = ' '.join(df[df['Company'] == company]['Processed_Text'].tolist())
    tokens = company_text.split()
    total = len(tokens)
    print(f'\n  {company} ({total} tokens):')
    for term in values_terms:
        count = tokens.count(term)
        if count > 0:
            print(f'    {term}: {count} ({count/total*1000:.1f} per 1k)')
