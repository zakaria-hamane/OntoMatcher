import json


# from f'extracted_data/{label}_sample.json' extract all the entities that have a description empty and put them in a list
def get_empty_desc(label):
    with open(f'extracted_data/empty/{label}_sample.json') as json_file:
        data = json.load(json_file)
        empty_desc = []
        for p in data:
            if p['ent_src_desc'] == '':
                empty_desc.append(p['ent_src_nm'])
            if p['ent_trgt_desc'] == '':
                empty_desc.append(p['ent_trgt_nm'])
        empty_desc = list(dict.fromkeys(empty_desc))
        return empty_desc

# save the list of entities with empty description in a file as a list of {entity_name: entity_description} the entity_description is empty
def save_to_file_desc(label, empty_desc):
    for i in range(len(empty_desc)):
        empty_desc[i] = {empty_desc[i]: ''}
    with open(f'extracted_data/empty/{label}_empty_desc.json', 'w') as outfile:
        json.dump(empty_desc, outfile, indent=4)

# get the list of entities with empty description
labels = ['child_of', 'parent_of', 'exact_match']
# get the list of entities with empty description for all the labels in one file
all_empty_desc = []
for label in labels:
    empty_desc = get_empty_desc(label)
    all_empty_desc.extend(empty_desc)
all_empty_desc = list(dict.fromkeys(all_empty_desc))
save_to_file_desc('all', all_empty_desc)

# count how many entities have empty description in 'all_empty_desc.json'
with open('extracted_data/empty/all_empty_desc.json') as json_file:
    data = json.load(json_file)
    print(f"Number of entities with empty description: {len(data)}")


# get empty ent_src_ctxt or ent_trg_ctxt
def get_empty_cntx(label):
    with open(f'extracted_data/empty/{label}_sample.json') as json_file:
        data = json.load(json_file)
        empty_cntx = []
        for p in data:
            if p['ent_src_ctxt'] == []:
                empty_cntx.append(p['ent_src_nm'])
            if p['ent_trgt_ctxt'] == []:
                empty_cntx.append(p['ent_trgt_nm'])
        empty_cntx = list(dict.fromkeys(empty_cntx))
        return empty_cntx

def save_to_file_cntx(label, empty_cntx):
    for i in range(len(empty_cntx)):
        empty_cntx[i] = {empty_cntx[i]: []}
    with open(f'extracted_data/empty/{label}_empty_cntx.json', 'w') as outfile:
        json.dump(empty_cntx, outfile, indent=4)

all_empty_cntx = []
for label in labels:
    empty_cntx = get_empty_cntx(label)
    all_empty_cntx.extend(empty_cntx)
all_empty_cntx = list(dict.fromkeys(all_empty_cntx))
save_to_file_cntx('all', all_empty_cntx)

with open('extracted_data/empty/all_empty_cntx.json') as json_file:
    data = json.load(json_file)
    print(f"Number of entities with empty context: {len(data)}")