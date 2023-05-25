import json
import numpy as np
import tensorflow as tf
from transformers import TFBertModel, BertTokenizer, BertConfig
from tensorflow.keras.layers import Bidirectional, LSTM, Dense, Input
from tensorflow.keras.models import Model
from tqdm import tqdm

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
    for attr in ['ent_src_ctxt', 'ent_trgt_ctxt']:
        if entity[attr]:  # Check if the context list is not empty
            # Take only the first 3 contexts
            entity[attr] = entity[attr][:3]

            avg_embedding = np.zeros((1, 768))
            for context_text in entity[attr]:
                input_ids = tokenizer.encode(context_text, add_special_tokens=True)
                if len(input_ids) > MAX_LEN:
                    input_ids = input_ids[:MAX_LEN - 1] + [tokenizer.sep_token_id]

                input_ids = tf.reshape(input_ids, (1, -1))  # reshape for batch size
                context_biobert_embedding = biobert_model(input_ids=input_ids).last_hidden_state.numpy()
                context_final_embedding = entity_bi_lstm_model(context_biobert_embedding).numpy()
                avg_embedding += context_final_embedding
            avg_embedding /= len(entity[attr])
            entity[attr] = avg_embedding.tolist()
        else:
            entity[attr] = np.zeros((1, 768)).tolist()

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
