import pandas as pd
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import LSTM


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
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)

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

    def train_model(self):
        """Entrena el model"""
        self.model = Sequential()
        self.model.add(LSTM(64, input_shape=(self.X_train_transformed.shape[1], 1), activation='relu'))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(len(self.num_features), activation='linear'))
        self.model.compile(optimizer=Adam(learning_rate=0.001), loss='mse')

        self.model.fit(self.X_train_transformed.reshape(-1, self.X_train_transformed.shape[1], 1), self.y_train,
                       epochs=100, verbose=1)

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
        # Genera la data per al 1 d'abril de 2023
        future_date = pd.to_datetime('2023-04-01')

        # Crea un DataFrame buit per a emmagatzemar les combinacions de data i categoria
        self.future_data = pd.DataFrame(columns=[f"{self.date_col}_year", f"{self.date_col}_month", f"{self.date_col}_day", self.cat_feature] + self.num_features)

        # Genera totes les combinacions possibles de data i categoria
        for category in self.unique_categories:
            new_row = pd.DataFrame({f"{self.date_col}_year": [future_date.year], f"{self.date_col}_month": [future_date.month], f"{self.date_col}_day": [future_date.day], self.cat_feature: [category],
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
        output_file = "prediccions_RNNModel_2023.xlsx"
        self.future_data[self.num_features] = self.future_predictions
        self.future_data.to_excel(output_file, engine='openpyxl', index=False)


if __name__ == '__main__':
    predictor = RNNModel(input_file=".\\dades\\Dades_Grups.csv")
    predictor.read_data()
    predictor.remove_nan()
    predictor.set_columns()
    predictor.adjust_data()
    predictor.split_data()
    predictor.preprocess_data()
    predictor.train_model()
    predictor.generate_future_data()
    predictor.predict_future_data()