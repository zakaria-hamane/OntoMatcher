import json
from transformers import TFBertModel, BertTokenizer, BertConfig
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input
from tensorflow.keras.models import Model
from tqdm import tqdm
import tensorflow as tf


# Moved these outside of the processing function to load once
tokenizer = BertTokenizer.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
config = BertConfig.from_pretrained("dmis-lab/biobert-base-cased-v1.1")
biobert_model = TFBertModel.from_pretrained("dmis-lab/biobert-base-cased-v1.1", from_pt=True, config=config)

MAX_LEN = 512  # set the max length

def create_bi_lstm_model():
    inputs = Input(shape=(None, 768))
    bi_lstm = Bidirectional(LSTM(128))(inputs)
    outputs = Dense(768, activation='relu')(bi_lstm)
    return Model(inputs, outputs)


# load the lstm model once
entity_bi_lstm_model = create_bi_lstm_model()


def process_entity(entity, tokenizer, biobert_model, entity_bi_lstm_model):
    for attr in ['ent_src_desc', 'ent_trgt_desc']:
        if entity[attr]:  # Check if the description is not empty
            entity_text = entity[attr]

            # tokenize and check length
            input_ids = tokenizer.encode(entity_text, add_special_tokens=True)
            if len(input_ids) > MAX_LEN:
                input_ids = input_ids[:MAX_LEN - 1] + [tokenizer.sep_token_id]  # ensure it ends with [SEP] token

            input_ids = tf.reshape(input_ids, (1, -1))  # reshape for batch size

            entity_biobert_embedding = biobert_model(input_ids=input_ids).last_hidden_state.numpy()
            entity_final_embedding = entity_bi_lstm_model(entity_biobert_embedding).numpy()
            entity[attr] = entity_final_embedding.tolist()

    return entity


def run():
    # Load the data
    with open("extracted_data/embeddings/final_data_train.json", "r") as f:
        data = json.load(f)

    # Process each entity with a progress bar
    for i in tqdm(range(len(data)), desc="Processing entities"):
        data[i] = process_entity(data[i], tokenizer, biobert_model, entity_bi_lstm_model)

    # Save the updated entities back to the JSON file
    with open("extracted_data/embeddings/final_data_train.json", "w") as f:
        json.dump(data, f)


if __name__ == "__main__":
    run()
