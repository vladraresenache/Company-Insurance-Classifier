import pandas as pd
from sentence_transformers import SentenceTransformer, util
from tqdm import tqdm
import torch
import numpy as np
import os
from transformers import AutoTokenizer, AutoModel

# Load datasets
companies = pd.read_csv("ml_insurance_challenge.csv")
taxonomy = pd.read_csv("insurance_taxonomy.csv")
print("Companies:", companies.shape)
print("Taxonomy:", taxonomy.shape)

# Combine relevant fields into a single string for embeddings
def combine_fields(row):
    desc = str(row['description'])
    tags = " ".join(eval(row['business_tags'])) if isinstance(row['business_tags'], str) else ""
    return f"{desc}. Business tags: {tags}. Sector: {row['sector']}. Category: {row['category']}. Niche: {row['niche']}"

companies['combined_text'] = companies.apply(combine_fields, axis=1)

# Load or initialize embedding model
model_path = "saved_model"
taxonomy_labels = taxonomy['label'].tolist()

if os.path.exists(model_path):
    print("Loading saved model...")
    model = SentenceTransformer(model_path)
else:
    print("Downloading model...")
    tokenizer = AutoTokenizer.from_pretrained("cambridgeltl/business-bert")
    bert_model = AutoModel.from_pretrained("cambridgeltl/business-bert")
    model = SentenceTransformer(modules=[bert_model, tokenizer])
    model.save(model_path)

# Load or compute taxonomy embeddings
taxonomy_emb_file = "taxonomy_embeddings.npy"
if os.path.exists(taxonomy_emb_file):
    print("Loading taxonomy embeddings...")
    taxonomy_embeddings = torch.tensor(np.load(taxonomy_emb_file))
else:
    print("Computing taxonomy embeddings...")
    taxonomy_embeddings = model.encode(taxonomy_labels, convert_to_tensor=True)
    np.save(taxonomy_emb_file, taxonomy_embeddings.cpu().numpy())

# Function to compute weighted embeddings for companies
def encode_company(row, weights=None):
    if weights is None:
        weights = {'description': 0.6, 'business_tags': 0.2, 'sector': 0.1, 'category': 0.05, 'niche': 0.05}

    desc = str(row['description'])
    tags = " ".join(eval(row['business_tags'])) if isinstance(row['business_tags'], str) else ""
    sector, category, niche = row['sector'], row['category'], row['niche']

    emb_desc = model.encode(desc, convert_to_tensor=True)
    emb_tags = model.encode(tags, convert_to_tensor=True)
    emb_sector = model.encode(sector, convert_to_tensor=True)
    emb_category = model.encode(category, convert_to_tensor=True)
    emb_niche = model.encode(niche, convert_to_tensor=True)

    combined_emb = (
        weights['description'] * emb_desc +
        weights['business_tags'] * emb_tags +
        weights['sector'] * emb_sector +
        weights['category'] * emb_category +
        weights['niche'] * emb_niche
    )
    return combined_emb

# Load or compute company embeddings
company_emb_file = "company_embeddings.npy"
if os.path.exists(company_emb_file):
    print("Loading company embeddings...")
    company_embeddings = torch.tensor(np.load(company_emb_file))
else:
    print("Computing company embeddings...")
    company_embeddings = torch.stack([encode_company(row) for _, row in tqdm(companies.iterrows(), total=len(companies))])
    np.save(company_emb_file, company_embeddings.cpu().numpy())

# Classify companies (top-3 predictions)
predicted_labels, max_scores, max_labels = [], [], []

for emb in tqdm(company_embeddings, desc="Classifying companies"):
    cos_scores = util.cos_sim(emb, taxonomy_embeddings)[0]
    top_scores, top_indices = torch.topk(cos_scores, k=3)
    predicted_labels.append([taxonomy_labels[i] for i in top_indices])
    max_scores.append(top_scores[0].item())
    max_labels.append(taxonomy_labels[top_indices[0]])

companies['predicted_labels'] = predicted_labels
companies['max_score'] = max_scores
companies['max_label'] = max_labels

companies.to_csv("classified_companies.csv", index=False)
print("Classification complete. Saved to classified_companies.csv")

# Update validation sample without overwriting human labels
validation = pd.read_csv("validation_sample.csv")
classified = companies[['combined_text', 'predicted_labels', 'max_label']]

validation_updated = validation.merge(
    classified,
    on='combined_text',
    how='left',
    suffixes=('', '_new')
)

validation_updated['predicted_labels'] = validation_updated['predicted_labels_new']
validation_updated['max_label'] = validation_updated['max_label_new']
validation_updated.drop(columns=['predicted_labels_new', 'max_label_new'], inplace=True)

validation_updated.to_csv("validation_sample.csv", index=False)
print("Validation sample updated with predicted labels. human_label preserved.")
