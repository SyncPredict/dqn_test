import pandas as pd
from sklearn.preprocessing import StandardScaler, MinMaxScaler

def preprocess(df):
    # Преобразование строки DATE_TIME в datetime и установка как индекс
    df['DATE_TIME'] = pd.to_datetime(df['DATE_TIME'])
    df.set_index('DATE_TIME', inplace=True)

    # Удаление дубликатов, сохраняя последнее значение в каждую секунду
    df = df[~df.index.duplicated(keep='last')]

    # Выбор метода масштабирования
    scaler = StandardScaler()  # или MinMaxScaler()
    scaled_values = scaler.fit_transform(df[['CLOSE']])

    # Возвращение pandas Series с сохраненным индексом
    return pd.Series(scaled_values.flatten(), index=df.index)

class DataPreprocessor:
    def __init__(self, file_path):
        self.file_path = file_path

    def load_data(self, test_len):
        # Загрузка данных из CSV файла
        df = pd.read_csv(self.file_path, delimiter=';')

        # Обработка данных
        processed_df = preprocess(df)

        # Деление данных на две части
        train_df = processed_df.iloc[:-test_len]
        test_df = processed_df.iloc[-test_len:]

        return train_df, test_df
