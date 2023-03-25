import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM
import matplotlib.pyplot as plt
import os
import optuna
from tensorflow.keras.callbacks import EarlyStopping
from datetime import datetime
from tensorflow.keras import backend as K
from tensorflow.keras.layers import Bidirectional
from sklearn.metrics import mean_squared_error
from tensorflow.keras.layers import Dropout, BatchNormalization
#import schedule
import time


class RNNModel:
    def __init__(self, input_file):
        self.input_file = input_file

    def read_data(self):
        """Llegeix l'arxiu CSV"""
        self.data = pd.read_csv(self.input_file, sep=";", parse_dates=[0])

    def remove_nan(self):
        """Elimina les files amb valors NaN"""
        self.data = self.data.dropna()

    def set_columns(self):
        """Defineix la columna de data, categòrica i numèriques"""
        self.date_col = self.data.columns[0]
        self.cat_feature = self.data.columns[1]
        self.num_features = [col for col in self.data.columns[2:]]

    def split_data(self):
        """Divideix les dades en entrenament i prova"""
        X = self.data[[f"{self.date_col}_year", f"{self.date_col}_month", f"{self.date_col}_day", self.cat_feature]]
        y = self.data[self.num_features]

        # Reconstruir la columna de fecha original
        reconstructed_date = pd.to_datetime(
            X[[f"{self.date_col}_year", f"{self.date_col}_month", f"{self.date_col}_day"]].rename(columns={
                f"{self.date_col}_year": "year",
                f"{self.date_col}_month": "month",
                f"{self.date_col}_day": "day"
            }))

        # Establecer la fecha de corte en el 1 de enero de 2023
        cutoff_date = pd.to_datetime('2023-01-01')

        # Separar los datos en entrenamiento y prueba según la fecha de corte
        train_mask = reconstructed_date < cutoff_date
        test_mask = reconstructed_date >= cutoff_date

        self.X_train = X[train_mask]
        self.X_test = X[test_mask]
        self.y_train = y[train_mask]
        self.y_test = y[test_mask]

    def extract_date_features(self, data, date_col):
        """Extrae características numéricas de las fechas."""
        data[date_col + '_year'] = data[date_col].dt.year
        data[date_col + '_month'] = data[date_col].dt.month
        data[date_col + '_day'] = data[date_col].dt.day
        data = data.drop(date_col, axis=1)
        return data

    def preprocess_data(self):
        """Preprocessament de les dades"""
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), [3]),
            ('num', 'passthrough', [0, 1, 2])
        ], remainder='drop')

        self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_transformed = self.preprocessor.transform(self.X_test)

    def weighted_mse(self, y_true, y_pred):
        weights = np.array([1.0] * 6 + [0.2] * (len(self.num_features) - 6))
        y_true = K.cast(y_true, 'float32')
        y_pred = K.cast(y_pred, 'float32')
        return K.mean(K.square(y_true - y_pred) * K.constant(weights), axis=-1)

    def create_model(self, trial=None, params=None):
        if trial is not None:
            n_layers = trial.suggest_int("n_layers", 10, 50)
        else:
            n_layers = params["n_layers"]

        model = Sequential()
        for i in range(n_layers):
            if trial is not None:
                num_units = trial.suggest_int(f"num_units_layer_{i + 1}", 10, 100)
            else:
                num_units = params[f"num_units_layer_{i + 1}"]

            if i == 0:
                model.add(Bidirectional(
                    LSTM(num_units, activation='relu', input_shape=(self.X_train_transformed.shape[1], 1),
                         return_sequences=True)))
                model.add(Bidirectional(LSTM(num_units, activation='relu', return_sequences=True)))
            elif i == n_layers - 1:
                model.add(LSTM(num_units, activation='relu'))
            else:
                model.add(LSTM(num_units, activation='relu', return_sequences=True))

            model.add(Dropout(rate=0.5))
            model.add(BatchNormalization())

        model.add(Dense(len(self.num_features)))

        if trial is not None:
            lr = trial.suggest_float("lr", 1e-5, 1e-2, log=True)
        else:
            lr = params["lr"]

        model.compile(optimizer=Adam(learning_rate=lr), loss=self.weighted_mse)
        return model

    def objective(self, trial):
        model = self.create_model(trial)
        early_stop = EarlyStopping(monitor='val_loss', patience=5)
        epochs = trial.suggest_int("epochs", 10, 50)
        history = model.fit(
            self.X_train_transformed.reshape(-1, self.X_train_transformed.shape[1], 1),
            self.y_train,
            epochs=epochs,
            verbose=1,
            validation_data=(self.X_test_transformed.reshape(-1, self.X_test_transformed.shape[1], 1), self.y_test),
            callbacks=[early_stop],
        )
        return history.history["val_loss"][-1]

    def train_model(self):
        """Entrena el model"""
        study = optuna.create_study(direction="minimize")
        study.optimize(self.objective, n_trials=1)

        self.best_params = study.best_params
        self.model = self.create_model(params=self.best_params)
        self.model.fit(
            self.X_train_transformed.reshape(-1, self.X_train_transformed.shape[1], 1),
            self.y_train,
            epochs=self.best_params["epochs"],
            verbose=1
        )

    def adjust_data(self):
        """Ajusta les dades"""
        # Extraer características numéricas de las fechas
        self.data = self.extract_date_features(self.data, self.date_col)

        # Divideix les dades en entrenament i prova
        X = self.data[[f"{self.date_col}_year", f"{self.date_col}_month", f"{self.date_col}_day", self.cat_feature]]
        y = self.data[self.num_features]

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Preprocessament de les dades
        self.preprocessor = ColumnTransformer(transformers=[
            ('cat', OneHotEncoder(handle_unknown='ignore'), [3]),
            ('num', 'passthrough', [0, 1, 2])
        ], remainder='drop')
        self.X_train_transformed = self.preprocessor.fit_transform(self.X_train)
        self.X_test_transformed = self.preprocessor.transform(self.X_test)

        # Guardar las categorías únicas para futuras predicciones
        self.unique_categories = sorted(self.data[self.cat_feature].unique())

    def generate_future_data(self):
        """Genera dades futures"""
        # Genera las fechas para el primer día de cada mes en 2023
        future_dates = pd.date_range(start='2023-01-01', end='2023-12-01', freq='MS')

        # Crea un DataFrame buit per a emmagatzemar les combinacions de data i categoria
        self.future_data = pd.DataFrame(columns=[self.date_col, self.cat_feature] + self.num_features)

        # Genera totes les combinacions possibles de data i categoria
        for future_date in future_dates:
            for category in self.unique_categories:
                new_row = pd.DataFrame({self.date_col: [future_date],
                                        self.cat_feature: [category],
                                        **{feat: [np.nan] for feat in self.num_features}})
                self.future_data = pd.concat([self.future_data, new_row], ignore_index=True)

    def predict_future_data(self):
        """Preveure dades futures"""
        # Extraer características numéricas de las fechas
        self.future_data = self.extract_date_features(self.future_data, self.date_col)

        # Usa las columnas extraídas de la fecha y la columna categórica
        self.future_data_transformed = self.preprocessor.transform(self.future_data[[f"{self.date_col}_year",
                                                                                     f"{self.date_col}_month",
                                                                                     f"{self.date_col}_day",
                                                                                     self.cat_feature]])
        # Convierte los datos de entrada a float32
        self.future_data_transformed = self.future_data_transformed.astype(np.float32)

        self.future_predictions = self.model.predict(self.future_data_transformed.reshape(-1, self.future_data_transformed.shape[1], 1))

        # Guarda les prediccions en un arxiu Excel
        self.save_predictions()

    def save_predictions(self):
        """Guarda les prediccions en un arxiu Excel"""
        current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
        output_file = f".\\previsio\\prediccions_RNNModel_2023_{current_time}.xlsx"

        # Reconstruir la columna de fecha original
        reconstructed_date = pd.to_datetime(
            self.future_data[[f"{self.date_col}_year", f"{self.date_col}_month", f"{self.date_col}_day"]].rename(
                columns={
                    f"{self.date_col}_year": "year",
                    f"{self.date_col}_month": "month",
                    f"{self.date_col}_day": "day"
                }))

        # Agregar la columna de fecha reconstruida al DataFrame
        self.future_data[self.date_col] = reconstructed_date

        # Eliminar las columnas de año, mes y día
        self.future_data.drop(columns=[f"{self.date_col}_year", f"{self.date_col}_month", f"{self.date_col}_day"],
                              inplace=True)

        # Guarda las primeras 6 columnas de características numéricas
        # Esto haria que escribiera todas las columnas de predicciones en el archivo de salida
        num_features_to_save = len(self.num_features)
        for i in range(num_features_to_save):
            self.future_data[self.num_features[i]] = self.future_predictions[:, i]

        # Reorganizar las columnas para que estén en el orden deseado: fecha, categoría, características numéricas
        columns_order = [self.date_col, self.cat_feature] + self.num_features[:num_features_to_save]
        self.future_data = self.future_data[columns_order]

        # Guardar el DataFrame en un archivo Excel
        self.future_data.to_excel(output_file, engine='openpyxl', index=False)


    def plot_comparison(self):
        y_pred = self.model.predict(self.X_test_transformed.reshape(-1, self.X_test_transformed.shape[1], 1))
        n_plots = len(self.num_features) # numero de columnes
        # Grafic de solament 6 variables
        for i in range(6):
            plt.figure()
            plt.plot(self.y_test.reset_index(drop=True).iloc[:, i], label="Real", linestyle='-')
            plt.plot(y_pred[:, i], label="Predicció", linestyle='--')
            plt.legend()
            plt.xlabel("Temps")
            plt.ylabel("Valor")
            plt.title(f"Comparació de dades reals i prediccions per a {self.num_features[i]}")
            current_time = datetime.now().strftime("%d-%m-%Y_%H-%M")
            plt.savefig(f".\\GRAFIC\\grafic_{self.num_features[i]}_{current_time}.png")



if __name__ == '__main__':
    predictor = RNNModel(input_file=".\\dades\\Dades_Grups_2.csv")
    predictor.read_data()
    predictor.remove_nan()
    predictor.set_columns()
    predictor.adjust_data()
    predictor.split_data()
    predictor.preprocess_data()
    predictor.train_model()
    predictor.generate_future_data()
    predictor.predict_future_data()
    predictor.plot_comparison()
