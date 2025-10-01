import pandas as pd
import ast
from collections import defaultdict
from sentence_transformers import util, SentenceTransformer
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.manifold import TSNE

# Load the validation sample
companies = pd.read_csv("validation_sample.csv")

# Ensure predicted_labels and human_label are stored as lists
def ensure_list(x):
    if isinstance(x, str):
        try:
            return ast.literal_eval(x)
        except:
            return [x]
    elif isinstance(x, list):
        return x
    else:
        return [x]

companies['predicted_labels'] = companies['predicted_labels'].apply(ensure_list)
companies['human_label'] = companies['human_label'].apply(ensure_list)

# Load model and taxonomy embeddings
model = SentenceTransformer("saved_model")
taxonomy_labels = pd.read_csv("insurance_taxonomy.csv")['label'].tolist()
taxonomy_embeddings = torch.tensor(np.load("taxonomy_embeddings.npy"))

# Compute similarity between each company and taxonomy
cos_sims = []
for _, row in companies.iterrows():
    emb = model.encode(row['combined_text'], convert_to_tensor=True)
    scores = util.cos_sim(emb, taxonomy_embeddings)[0]
    top_idx = torch.argmax(scores).item()
    cos_sims.append({'top_score': scores[top_idx].item(), 'top_label': taxonomy_labels[top_idx]})

cos_df = pd.DataFrame(cos_sims)
companies['top_score'] = cos_df['top_score']
companies['top_label_emb'] = cos_df['top_label']

# Visualize distribution of top similarities
plt.figure(figsize=(10,5))
plt.hist(companies['top_score'], bins=30)
plt.title("Distribution of Top Cosine Similarities")
plt.xlabel("Cosine Similarity")
plt.ylabel("Number of Companies")
plt.show()

# Check coverage based on similarity
coverage = (companies['top_score'] > 0.0).mean()
print(f"Coverage by embedding similarity: {coverage:.3f}")

# t-SNE visualization of company embeddings
company_embs = torch.stack([model.encode(text, convert_to_tensor=True) for text in companies['combined_text']])
emb_2d = TSNE(n_components=2, random_state=42).fit_transform(company_embs.cpu().numpy())

plt.figure(figsize=(12,8))
sns.scatterplot(
    x=emb_2d[:,0], y=emb_2d[:,1],
    hue=[pred[0] if pred else 'None' for pred in companies['predicted_labels']],
    palette='tab20', legend='full'
)
plt.title("t-SNE of Company Embeddings colored by Top Predicted Label")
plt.show()

# Label similarity matrix
label_sims = util.cos_sim(taxonomy_embeddings, taxonomy_embeddings).cpu().numpy()
label_sims_df = pd.DataFrame(label_sims, index=taxonomy_labels, columns=taxonomy_labels)

plt.figure(figsize=(12,10))
sns.heatmap(label_sims_df, cmap="coolwarm", vmin=0, vmax=1)
plt.title("Label Embedding Cosine Similarity Matrix")
plt.show()

# Performance metrics
def precision_at_k(y_true, y_pred, k=3):
    hits = sum(any(label in pred[:k] for label in true) for true, pred in zip(y_true, y_pred))
    return hits / len(y_true)

def mean_reciprocal_rank(y_true, y_pred):
    rr_total = 0
    for true, pred in zip(y_true, y_pred):
        rr_total += next((1/(i+1) for i, label in enumerate(pred) if label in true), 0)
    return rr_total / len(y_true)

precision1 = precision_at_k(companies['human_label'], companies['predicted_labels'], k=1)
precision3 = precision_at_k(companies['human_label'], companies['predicted_labels'], k=3)
mrr_score = mean_reciprocal_rank(companies['human_label'], companies['predicted_labels'])

print(f"Precision@1: {precision1:.3f}")
print(f"Precision@3: {precision3:.3f}")
print(f"MRR: {mrr_score:.3f}")
print(f"Coverage: {coverage:.3f}")

# Per-label precision
per_label_hits = defaultdict(int)
per_label_total = defaultdict(int)
for true, pred in zip(companies['human_label'], companies['predicted_labels']):
    for t in true:
        per_label_total[t] += 1
        if t in pred:
            per_label_hits[t] += 1

per_label_precision = {label: per_label_hits[label] / per_label_total[label] for label in per_label_total}
top10 = sorted(per_label_precision.items(), key=lambda x: x[1], reverse=True)[:10]

print("\nTop 10 per-label precision:")
for label, prec in top10:
    print(f"{label}: {prec:.3f}")

# Error analysis
tp_examples, fp_examples, fn_examples = [], [], []

for i, row in companies.iterrows():
    pred, true = set(row['predicted_labels']), set(row['human_label'])
    tp, fp, fn = pred & true, pred - true, true - pred

    if tp:
        tp_examples.append({'index': i, 'combined_text': row['combined_text'], 'predicted': list(tp), 'true': list(true)})
    if fp:
        fp_examples.append({'index': i, 'combined_text': row['combined_text'], 'predicted': list(fp), 'true': list(true)})
    if fn:
        fn_examples.append({'index': i, 'combined_text': row['combined_text'], 'predicted': list(pred), 'true': list(fn)})

def print_sample_examples(title, examples, n=5):
    print(f"\nSample {title}:")
    for ex in examples[:n]:
        print(f"Index {ex['index']} | Pred: {ex['predicted']} | True: {ex['true']}")
        print(f"Snippet: {ex['combined_text'][:150]}...\n")

print_sample_examples("True Positives", tp_examples)
print_sample_examples("False Positives", fp_examples)
print_sample_examples("False Negatives", fn_examples)
