import json
import numpy as np
from tqdm import tqdm

# Load your JSON file
with open('extracted_data/embeddings/final_data_train.json', 'r') as f:
    data = json.load(f)

new_data = []
total = sum(1 for _ in data)

for entry in tqdm(data, total=total, desc="Processing data"):
    # Ensure the embeddings are 2D arrays
    ent_src_nm = np.array(entry["ent_src_nm"])
    ent_src_desc = np.array(entry["ent_src_desc"]) if entry["ent_src_desc"] else np.zeros((1, 768))
    ent_src_ctxt = np.array(entry["ent_src_ctxt"]) if entry["ent_src_ctxt"] else np.zeros((1, 768))

    ent_trgt_nm = np.array(entry["ent_trgt_nm"])
    ent_trgt_desc = np.array(entry["ent_trgt_desc"]) if entry["ent_trgt_desc"] else np.zeros((1, 768))
    ent_trgt_ctxt = np.array(entry["ent_trgt_ctxt"]) if entry["ent_trgt_ctxt"] else np.zeros((1, 768))

    # Perform max pooling
    entity_src = np.max([ent_src_nm, ent_src_desc, ent_src_ctxt], axis=0).tolist()
    entity_trgt = np.max([ent_trgt_nm, ent_trgt_desc, ent_trgt_ctxt], axis=0).tolist()

    # Create new data entry
    new_entry = {
        "entities_hash": entry["entities_hash"],
        "ent_src": entity_src,
        "ent_trgt": entity_trgt,
        "rel_type": entry["rel_type"]
    }
    new_data.append(new_entry)

# Save to a new JSON file
with open('extracted_data/embeddings/ent_src_trgt_vectors.json', 'w') as f:
    json.dump(new_data, f)

