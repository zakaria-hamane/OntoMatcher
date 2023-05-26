import json
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from tensorflow.keras.optimizers import Adam
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import GridSearchCV
from keras_tuner import HyperParameters
from keras_tuner.tuners import RandomSearch
from tensorflow.keras.models import load_model
import os


class DataLoader:
    def __init__(self, filename):
        self.filename = filename

    def load_data(self):
        with open(self.filename, 'r') as f:
            data = json.load(f)

        ent_src_data = np.array([entry['ent_src'] for entry in data])
        ent_trgt_data = np.array([entry['ent_trgt'] for entry in data])
        labels = np.array([entry['rel_type'] for entry in data])

        return ent_src_data, ent_trgt_data, labels


class SiameseNetwork:
    def __init__(self, input_shape):
        self.input_shape = input_shape

    def build_model(self, hp):
        input_shape = self.input_shape

        base_network = self.create_base_network(input_shape, hp)

        input_a = layers.Input(shape=input_shape)
        input_b = layers.Input(shape=input_shape)
        processed_a = base_network(input_a)
        processed_b = base_network(input_b)

        combined = layers.concatenate([processed_a, processed_b])

        combined = layers.Flatten()(combined)
        output = layers.Dense(1, activation='sigmoid')(combined)

        siamese_model = Model([input_a, input_b], output)
        siamese_model.compile(loss='binary_crossentropy', optimizer=Adam(), metrics=['accuracy'])

        return siamese_model

    def create_base_network(self, input_shape, hp):
        input = layers.Input(shape=input_shape)
        units = hp.Int('units', min_value=32, max_value=512, step=32, default=128)
        x = layers.Dense(units, activation='relu')(input)
        x = layers.Dense(units, activation='relu')(x)
        x = layers.Dense(units, activation='relu')(x)
        return Model(input, x)

    def tune(self, ent_src_data, ent_trgt_data, labels, epochs=10, batch_size=32):
        binary_labels = (labels > 0).astype(int)
        tuner = RandomSearch(
            self.build_model,
            objective='val_accuracy',
            max_trials=5,
            executions_per_trial=3,
            directory='model/siamese/random_search',
            project_name='model/siamese/siamese_tuning',
            overwrite=True
        )
        tuner.search([ent_src_data, ent_trgt_data], binary_labels, epochs=epochs, validation_split=0.2, batch_size=batch_size)
        best_hp = tuner.get_best_hyperparameters(num_trials=1)[0]
        self.model = self.build_model(best_hp)

    def train(self, ent_src_data, ent_trgt_data, labels, epochs=10, batch_size=32):
        binary_labels = (labels > 0).astype(int)
        if not hasattr(self, 'model'):
            # Define a default HyperParameters object with default values
            hp = HyperParameters()
            hp.Int('units', min_value=32, max_value=512, step=32, default=128)
            self.model = self.build_model(hp)
        self.model.fit([ent_src_data, ent_trgt_data], binary_labels, epochs=epochs, batch_size=batch_size)

    def predict(self, ent_src_data, ent_trgt_data, threshold=0.5):
        predictions = self.model.predict([ent_src_data, ent_trgt_data])
        matches = predictions > threshold
        return matches

    def evaluate(self, ent_src_data, ent_trgt_data, labels, threshold=0.5):
        predictions = self.predict(ent_src_data, ent_trgt_data)
        preds = predictions.flatten().astype(int)
        true_labels = (labels > 0).astype(int)

        accuracy = accuracy_score(true_labels, preds)
        precision = precision_score(true_labels, preds)
        recall = recall_score(true_labels, preds)
        f1 = f1_score(true_labels, preds)

        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')

    def save(self, filepath):
        self.model.save(filepath)

    def load(self, filepath):
        self.model = load_model(filepath)

class XGBoostModel:
    def __init__(self, params, num_round):
        self.params = params
        self.num_round = num_round

    def train(self, X_train, y_train):
        dtrain = xgb.DMatrix(X_train, label=y_train)
        self.model = xgb.train(self.params, dtrain, self.num_round)

    def predict(self, X_test):
        dtest = xgb.DMatrix(X_test)
        preds_proba = self.model.predict(dtest)
        preds = np.argmax(preds_proba, axis=1)  # get class labels
        return preds

    def evaluate(self, X_test, y_test):
        preds = self.predict(X_test)

        accuracy = accuracy_score(y_test, preds)
        precision = precision_score(y_test, preds, average='macro')
        recall = recall_score(y_test, preds, average='macro')
        f1 = f1_score(y_test, preds, average='macro')

        print(f'Accuracy: {accuracy}')
        print(f'Precision: {precision}')
        print(f'Recall: {recall}')
        print(f'F1-Score: {f1}')

    def tune(self, X_train, y_train):
        param_grid = {
            'max_depth': [3, 7, 10],  # Typical values for max_depth
            'min_child_weight': [1, 5],  # Lower values (1, 2) often used, but you can add more if your model overfits
            'gamma': [0.0, 0.2],  # Start with 0 and a small value
            'subsample': [0.6, 1.0],  # Start with lower and upper limit
            'colsample_bytree': [0.6, 1.0],  # Start with lower and upper limit
            'eta': [0.01, 0.2]
            # Lower values (0.01-0.2) often used, but you can add more if your model is stable enough
        }

        xgb_estimator = xgb.XGBClassifier(objective='multi:softprob', num_class=4)

        grid_search = GridSearchCV(estimator=xgb_estimator, param_grid=param_grid, cv=3, scoring='f1_micro', verbose=2, n_jobs=-1)
        grid_search.fit(X_train, y_train)

        # Print and update the best parameters
        print(grid_search.best_params_)
        self.params = grid_search.best_params_

        # Train the model with the best parameters
        self.train(X_train, y_train)

    def save(self, filepath):
        self.model.save_model(filepath)

    def load(self, filepath):
        self.model = xgb.Booster()  # init model
        self.model.load_model(filepath)  # load data


def main(tune=True):
    # Load data
    loader = DataLoader('extracted_data/embeddings/ent_src_trgt_vectors.json')
    ent_src_data, ent_trgt_data, labels = loader.load_data()

    # Create binary labels for Siamese Network
    binary_labels = (labels > 0).astype(int)

    # Split data into training and test set
    ent_src_train, ent_src_test, ent_trgt_train, ent_trgt_test, labels_train, labels_test, binary_labels_train, binary_labels_test = train_test_split(ent_src_data, ent_trgt_data, labels, binary_labels, test_size=0.2, random_state=42)

    input_shape = ent_src_train.shape[1:]
    siamese = SiameseNetwork(input_shape)

    if not os.path.exists('model/siamese_model.h5'):
        if tune:
            # Tune Siamese Network
            print("Tuning Siamese Network:")
            best_params_siamese = siamese.tune(ent_src_train, ent_trgt_train, binary_labels_train)

            # Save the best parameters
            with open('siamese_best_params.json', 'w') as file:
                json.dump(best_params_siamese, file)

        # Train Siamese Network
        siamese.train(ent_src_train, ent_trgt_train, binary_labels_train)
        siamese.save('model/siamese_model.h5')
    else:
        siamese.load('model/siamese_model.h5')

    # Evaluate Siamese Network on Test Set
    print("Siamese Network Evaluation on Test Set:")
    siamese.evaluate(ent_src_test, ent_trgt_test, binary_labels_test)

    # Get matched data
    matches_train = siamese.predict(ent_src_train, ent_trgt_train)
    matches_test = siamese.predict(ent_src_test, ent_trgt_test)

    matched_src_train = ent_src_train[matches_train]
    matched_trgt_train = ent_trgt_train[matches_train]
    matched_src_test = ent_src_test[matches_test]
    matched_trgt_test = ent_trgt_test[matches_test]

    matches_train = matches_train.flatten()
    matches_test = matches_test.flatten()

    matched_labels_train = labels_train[matches_train]
    matched_labels_test = labels_test[matches_test]

    # Only feed the XGBoost model with pairs that have labels 1, 2, or 3
    xgb_indices_train = np.where((matched_labels_train == 1) | (matched_labels_train == 2) | (matched_labels_train == 3))
    xgb_indices_test = np.where((matched_labels_test == 1) | (matched_labels_test == 2) | (matched_labels_test == 3))

    xgb_data_train = np.concatenate([matched_src_train[xgb_indices_train], matched_trgt_train[xgb_indices_train]], axis=1)
    xgb_data_test = np.concatenate([matched_src_test[xgb_indices_test], matched_trgt_test[xgb_indices_test]], axis=1)

    xgb_labels_train = matched_labels_train[xgb_indices_train]
    xgb_labels_test = matched_labels_test[xgb_indices_test]

    # Train XGBoost
    xgb_params = {'max_depth': 3, 'eta': 0.3, 'objective': 'multi:softprob', 'num_class': 4, 'nthread': -1}
    xgb_model = XGBoostModel(xgb_params, num_round=20)

    if not os.path.exists('model/xgb_model.json'):
        if tune:
            # Tune XGBoost
            print("Tuning XGBoost:")
            best_params_xgb = xgb_model.tune(xgb_data_train, xgb_labels_train)

            # Save the best parameters
            with open('xgb_best_params.json', 'w') as file:
                json.dump(best_params_xgb, file)

        # Train XGBoost
        xgb_model.train(xgb_data_train, xgb_labels_train)
        xgb_model.save('model/xgb_model.json')
    else:
        xgb_model.load('model/xgb_model.json')

    preds = xgb_model.predict(xgb_data_test)

    # Evaluate XGBoost on Test Set
    print("XGBoost Evaluation on Test Set:")
    xgb_model.evaluate(xgb_data_test, xgb_labels_test)

    return preds


if __name__ == "__main__":
    main(tune=False)
