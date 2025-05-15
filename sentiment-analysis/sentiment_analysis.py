import os
import keras_tuner
import keras
from sklearn import model_selection
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

class customNN:
    def __init__(self, input_dim, x_train, y_train, tuner_dir='.', project_name='nn', val_split=0.2, random_state=42, patience=5, max_epochs=50):
        os.environ['KERAS_BACKEND'] = 'tensorflow'
        self.input_dim = input_dim
        self.x_train = x_train
        self.y_train = y_train
        self.tuner_dir = tuner_dir
        self.project_name = project_name
        self.val_split = val_split
        self.random_state = random_state
        self.patience = patience
        self.max_epochs = max_epochs
        self.tuner = None
        self.best_model = None
        self.best_hyperparameters = None

    def nn_model(self, hp):
        model = keras.models.Sequential()
        model.add(keras.layers.Input(shape=(1024,), name='input_layer'))

        structure = hp.Choice('structure', [
            '1x16', '1x32',
            '2x32_16', '2x32_16_d', '2x32_16_b', '2x32_16_db', 
            '2x64_32', '2x64_32_d', '2x64_32_b', '2x64_32_db', 
            '3x64_32_16', '3x64_32_16_d', '3x64_32_16_b', '3x64_32_16_db', 
            '3x128_64_32', '3x128_64_32_d', '3x128_64_32_b', '3x128_64_32_db'
        ])

        if structure == '1x16':
            model.add(keras.layers.Dense(16, activation='relu'))
        elif structure == '1x32':
            model.add(keras.layers.Dense(32, activation='relu'))
        elif structure == '2x32_16':
            model.add(keras.layers.Dense(32, activation='relu'))
            model.add(keras.layers.Dense(16, activation='relu'))
        elif structure == '2x32_16_d':
            model.add(keras.layers.Dense(32, activation='relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(16, activation='relu'))
        elif structure == '2x32_16_b':
            model.add(keras.layers.Dense(32))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dense(16, activation='relu'))
        elif structure == '2x32_16_db':
            model.add(keras.layers.Dense(32))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(16, activation='relu'))
        elif structure == '2x64_32':
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dense(32, activation='relu'))
        elif structure == '2x64_32_d':
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(32, activation='relu'))
        elif structure == '2x64_32_b':
            model.add(keras.layers.Dense(64))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dense(32, activation='relu'))
        elif structure == '2x64_32_db':
            model.add(keras.layers.Dense(64))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(32, activation='relu'))
        elif structure == '3x64_32_16':
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dense(32, activation='relu'))
            model.add(keras.layers.Dense(16, activation='relu'))
        elif structure == '3x64_32_16_d':
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(32, activation='relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(16, activation='relu'))
        elif structure == '3x64_32_16_b':
            model.add(keras.layers.Dense(64))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dense(32))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dense(16, activation='relu'))
        elif structure == '3x64_32_16_db':
            model.add(keras.layers.Dense(64))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(32))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(16, activation='relu'))
        elif structure == '3x128_64_32':
            model.add(keras.layers.Dense(128, activation='relu'))
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dense(32, activation='relu'))
        elif structure == '3x128_64_32_d':
            model.add(keras.layers.Dense(128, activation='relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(64, activation='relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(32, activation='relu'))
        elif structure == '3x128_64_32_b':
            model.add(keras.layers.Dense(128))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dense(64))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dense(32, activation='relu'))
        elif structure == '3x128_64_32_db':
            model.add(keras.layers.Dense(128))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(64))
            model.add(keras.layers.BatchNormalization())
            model.add(keras.layers.Activation('relu'))
            model.add(keras.layers.Dropout(0.2))
            model.add(keras.layers.Dense(32, activation='relu'))


        model.add(keras.layers.Dense(1, activation='sigmoid', name='output_layer'))
        lr = hp.Choice('learning_rate', [0.0001, 0.001, 0.01])
        optimizer = keras.optimizers.Adam(learning_rate=lr)
        model.compile(loss='binary_crossentropy', optimizer=optimizer, metrics=['accuracy'])
        return model
    
    def search(self):
        early_stop = keras.callbacks.EarlyStopping(patience=self.patience, restore_best_weights=True)
        self.tuner = keras_tuner.tuners.GridSearch(
            self.nn_model,
            objective='val_accuracy',
            directory=self.tuner_dir,
            project_name=self.project_name
        )
        x_tr, x_val, y_tr, y_val = model_selection.train_test_split(
            self.x_train,
            self.y_train,
            test_size=self.val_split,
            random_state=self.random_state,
            stratify=self.y_train
        )
        self.tuner.search(
            x_tr,
            y_tr,
            validation_data=(x_val, y_val),
            epochs=self.max_epochs,
            callbacks=[early_stop]
        )
        self.best_model = self.tuner.get_best_models(num_models=1)[0]
        self.best_hyperparameters = self.tuner.get_best_hyperparameters(num_trials=1)[0]
        return (self.best_model, self.best_hyperparameters)
    
    def evaluate(self, x_test, y_test):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        y_pred = self.best_model.predict(x_test).flatten()
        y_pred = (y_pred > 0.5).astype(int)
        accuracy = accuracy_score(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)
        cr = classification_report(y_test, y_pred)
        return (accuracy, cm, cr)
    
    def predict(self, x):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        y_pred = self.best_model.predict(x).flatten()
        y_pred = (y_pred > 0.5).astype(int)
        return y_pred
    
    def save_model(self, model_path):
        if self.best_model is None:
            raise ValueError("Model has not been trained yet.")
        self.best_model.save(model_path)

    def load_model(self, model_path):
        self.best_model = keras.models.load_model(model_path)
    
