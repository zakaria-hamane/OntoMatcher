from tqdm import tqdm
import json
import tensorflow as tf
from tensorflow.keras.layers import Conv1D, Dense, LSTM, Bidirectional, Input, MaxPooling1D
from tensorflow.keras.models import Model
from transformers import TFBertModel, BertTokenizer, BertConfig


def create_cnn_model(input_shape):
    inputs = Input(shape=input_shape)
    conv1d = Conv1D(128, 1, activation='relu')(inputs)
    max_pool = MaxPooling1D()(conv1d)
    flattened = tf.keras.layers.Flatten()(max_pool)
    outputs = Dense(768, activation='relu')(flattened)
    model = Model(inputs, outputs)
    return model


def create_bi_lstm_model():
    inputs = Input(shape=(None, 768))
    bi_lstm = Bidirectional(LSTM(128))(inputs)
    outputs = Dense(768, activation='relu')(bi_lstm)
    return Model(inputs, outputs)


def load_json_file(path):
    with open(path, "r") as f:
        data = json.load(f)
    return data


def save_json_file(data, path):
    with open(path, "w") as f:
        json.dump(data, f)


def get_entity_embeddings(data, tokenizer, biobert_model):
    for item in tqdm(data, desc="Getting embeddings"):
        entity_text = item["ent_nm"]
        entity_biobert_embedding = biobert_model(
            **tokenizer(entity_text, return_tensors="tf")).last_hidden_state.numpy()
        entity_cnn_model = create_cnn_model((entity_biobert_embedding.shape[1], entity_biobert_embedding.shape[2]))
        entity_cnn_embedding = entity_cnn_model(entity_biobert_embedding).numpy()
        entity_highway_output = entity_cnn_embedding * (
                    1 - entity_biobert_embedding) + entity_biobert_embedding * entity_biobert_embedding
        entity_bi_lstm_model = create_bi_lstm_model()
        entity_final_embedding = entity_bi_lstm_model(entity_highway_output).numpy()
        item["ent_nm_embedd"] = entity_final_embedding.tolist()


def create_embedding_dict(unique_entities):
    return {entity['ent_nm']: entity['ent_nm_embedd'] for entity in unique_entities}


def replace_with_embedding(data, entity_embedding_dict):
    for item in tqdm(data, desc="Replacing with embeddings"):
        for attr in ['ent_src_nm', 'ent_trgt_nm']:
            if attr in item and item[attr] in entity_embedding_dict:
                # Replace the entity name with its embedding
                item[attr] = entity_embedding_dict[item[attr]]


def extract_unique_entities(input_path, output_path):
    entity_list = []
    data = load_json_file(input_path)
    for item in data:
        entity_list.append(item.get('ent_src_nm', ''))
        entity_list.append(item.get('ent_trgt_nm', ''))
    unique_entities = list(set(entity_list))
    entity_dicts = [{"ent_nm": entity, "ent_nm_embedd": ""} for entity in unique_entities]
    save_json_file(entity_dicts, output_path)
    print(f"Number of unique entities: {len(unique_entities)}")
    return unique_entities


def main():
    unique_entities = extract_unique_entities('extracted_data/embeddings/final_data_train.json',
                                              'extracted_data/embeddings/unique_entities.json')
    data = load_json_file("extracted_data/embeddings/unique_entities.json")
    tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    config = BertConfig.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
    biobert_model = TFBertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1", from_pt=True, config=config)
    get_entity_embeddings(data, tokenizer, biobert_model)
    save_json_file(data, "extracted_data/embeddings/unique_entities.json")
    entity_embedding_dict = create_embedding_dict(unique_entities)
    data = load_json_file("extracted_data/embeddings/final_data_train.json")
    replace_with_embedding(data, entity_embedding_dict)
    save_json_file(data, "extracted_data/embeddings/final_data_train.json")


if __name__ == "__main__":
    main()
