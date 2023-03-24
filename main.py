import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import SimpleRNN, Dense
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import datetime
import os


class RNNModel:
    def __init__(self, sequence_length=30, n_units=50):
        self.sequence_length = sequence_length
        self.n_units = n_units
        self.scaler = MinMaxScaler()
        self.model = None


    def preprocess_data(self, df):
        category_col = "Grup monitoritzacio"
        numeric_cols = df.columns[1:]
        df_numeric = df[numeric_cols]  # Create a separate DataFrame for numeric columns
        self.scaler.fit(df_numeric)
        df_numeric = self.scaler.transform(df_numeric)

        df = pd.concat([df[[category_col]], pd.DataFrame(df_numeric, columns=numeric_cols, index=df.index)], axis=1)
        df = pd.get_dummies(df, columns=[category_col])

        n_features = df.shape[1]
        X, y = [], []

        for i in range(self.sequence_length, len(df)):
            X.append(df.iloc[i - self.sequence_length:i].values)
            y.append(df.iloc[i].values)

        X, y = np.array(X), np.array(y)
        return X, y, n_features

    def inverse_transform(self, data, original_columns):
        data_numeric = self.scaler.inverse_transform(data[:, :len(original_columns) - 1])
        data_combined = np.concatenate([data_numeric, data[:, len(original_columns) - 1:]], axis=1)
        return data_combined


    def build(self, input_shape):
        self.model = Sequential()
        self.model.add(SimpleRNN(units=self.n_units, activation="relu", input_shape=input_shape))
        self.model.add(Dense(units=input_shape[1]))
        self.model.compile(optimizer="adam", loss="mean_squared_error")

    def train(self, X_train, y_train, epochs=100, batch_size=32):
        self.model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size)

    def predict(self, X):
        return self.model.predict(X)

    @staticmethod
    def save_forecast_to_excel(forecast_df, output_file):
        """
        Guarda un dataframe de pronóstico en un archivo Excel.
        Si el archivo de salida ya existe, lo sobrescribe.
        """
        if os.path.isfile(output_file):
            os.remove(output_file)
            print(f"'{output_file}' fue eliminado para ser sobrescrito")

        forecast_df.to_excel(output_file)
        print(f"Pronóstico guardado en '{output_file}'")

    def main(self):
        # Load data
        file_path = ".\\dades\\ICS LLE IQ foto diaria - Dades Brut.csv"
        df = pd.read_csv(file_path, delimiter=";", parse_dates=["Data Tall"], dayfirst=True)
        df = df.dropna()
        df.set_index("Data Tall", inplace=True)

        rnn = RNNModel(sequence_length=30, n_units=50)

        # Preprocess data
        X, y, n_features = rnn.preprocess_data(df)

        # Train-test split
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

        # Build and train RNN model
        input_shape = (rnn.sequence_length, n_features)
        rnn.build(input_shape)
        rnn.train(X_train, y_train, epochs=2, batch_size=32)

        # Predict data
        y_pred = rnn.predict(X_test)
        y_pred_inv = rnn.inverse_transform(y_pred, df.columns)
        y_test_inv = rnn.inverse_transform(y_test, df.columns)

        # Calculate performance using Mean Squared Error
        mse = mean_squared_error(y_test_inv, y_pred_inv)

        # Forecast data for 2023
        last_known_data = X[-1].reshape(1, rnn.sequence_length, n_features)
        forecasted_data = rnn.predict(last_known_data)
        forecasted_data_inv = rnn.inverse_transform(forecasted_data, df.columns)

        # Create a DataFrame with the forecasted data
        forecast_date = df.index[-1] + datetime.timedelta(days=1)
        modified_columns = pd.get_dummies(df.reset_index(), columns=['Grup monitoritzacio']).columns[
                           1:]  # Get the modified columns after creating dummy columns
        forecast_df = pd.DataFrame(forecasted_data_inv, columns=modified_columns, index=[forecast_date])

        print("\nForecast for", forecast_date.strftime("%Y-%m-%d"))

        rnn.save_forecast_to_excel(forecast_df, "foto_diaria_2023.xlsx")

    @staticmethod
    def save_forecast_to_excel(forecast_df, output_file):
        """
        Guarda un dataframe de pronóstico en un archivo Excel.
        Si el archivo de salida ya existe, lo sobrescribe.
        """
        if os.path.isfile(output_file):
            os.remove(output_file)
            print(f"'{output_file}' fue eliminado para ser sobrescrito")

        forecast_df.to_excel(output_file)
        print(f"Pronóstico guardado en '{output_file}'")


if __name__ == "__main__":
    predictorDT = RNNModel()
    predictorDT.main()