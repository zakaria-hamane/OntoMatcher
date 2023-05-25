import json
import os
from sklearn.model_selection import train_test_split
import hashlib


def generate_hash(input_string_1, input_string_2):
    combined_string = input_string_1 + input_string_2
    return hashlib.sha256(combined_string.encode()).hexdigest()

def replace_bad_search(data):
    for item in data:
        if "No good Google Search Result was found" in item['ent_src_ctxt']:
            item['ent_src_ctxt'] = []
        if "No good Google Search Result was found" in item['ent_trgt_ctxt']:
            item['ent_trgt_ctxt'] = []
    return data

def fill_missing_data(relations):
    aggregated_data = []

    for relation in relations:
        # Assumes file names are in the form: relation_sample.json
        sample_file = f'extracted_data/{relation}_sample.json'

        if os.path.isfile(sample_file):
            with open(sample_file, 'r') as f:
                sample_data = json.load(f)
        else:
            print(f"No file found for relation {relation}")
            continue

        with open('extracted_data/enriched/all_empty_desc.json', 'r') as f:
            desc_data = json.load(f)

        with open('extracted_data/enriched/all_empty_cntx.json', 'r') as f:
            cntx_data = json.load(f)

        desc_dict = {list(d.keys())[0]: list(d.values())[0] for d in desc_data}
        cntx_dict = {list(d.keys())[0]: list(d.values())[0] for d in cntx_data}

        for item in sample_data:
            # Add a hash for ent_src_nm and ent_trgt_nm combined
            item['entities_hash'] = generate_hash(item['ent_src_nm'], item['ent_trgt_nm'])

            if item['ent_src_desc'] == "":
                if item['ent_src_nm'] in desc_dict:
                    item['ent_src_desc'] = desc_dict[item['ent_src_nm']]
            if item['ent_trgt_desc'] == "":
                if item['ent_trgt_nm'] in desc_dict:
                    item['ent_trgt_desc'] = desc_dict[item['ent_trgt_nm']]
            if not item['ent_src_ctxt']:
                if item['ent_src_nm'] in cntx_dict:
                    item['ent_src_ctxt'] = cntx_dict[item['ent_src_nm']]
            if not item['ent_trgt_ctxt']:
                if item['ent_trgt_nm'] in cntx_dict:
                    item['ent_trgt_ctxt'] = cntx_dict[item['ent_trgt_nm']]

        # Add the enriched data to the aggregated list
        aggregated_data.extend(sample_data)

    # Write the aggregated data to the final file
    with open('extracted_data/final_data.json', 'w') as f:
        json.dump(aggregated_data, f, indent=4)

def clean_labels(data):
    cleaned_data = []
    for item in data:
        # replace the labels
        if item['rel_type'] == 'child_of':
            item['rel_type'] = 2
        elif item['rel_type'] == 'parent_of':
            item['rel_type'] = 3
        # assuming that other labels are either 0 or 1 and they remain as they are

        # reorder the keys such that 'rel_type' is last
        keys_order = [k for k in item if k != 'rel_type'] + ['rel_type']
        reordered_item = {k: item[k] for k in keys_order}
        cleaned_data.append(reordered_item)
    return cleaned_data

def split_data(filename, test_size=0.2):
    # Load the data
    with open(filename, 'r') as f:
        data = json.load(f)

    # Extract the labels
    labels = [d['rel_type'] for d in data]

    # Perform the split
    train_data, test_data = train_test_split(data, test_size=test_size, stratify=labels, random_state=42)

    # Save the split data
    with open('extracted_data/train/final_data_train.json', 'w') as f:
        json.dump(train_data, f, indent=4)

    with open('extracted_data/train/final_data_test.json', 'w') as f:
        json.dump(test_data, f, indent=4)


if __name__ == "__main__":
    fill_missing_data(['child_of', 'parent_of', 'exact_match'])
    with open('extracted_data/final_data.json', 'r') as f:
        enriched_data = json.load(f)
    enriched_data = replace_bad_search(enriched_data)  # add this line
    cleaned_data = clean_labels(enriched_data)
    # write the cleaned data back to the file
    with open('extracted_data/final_data.json', 'w') as f:
        json.dump(cleaned_data, f, indent=4)
    split_data('extracted_data/final_data.json')
