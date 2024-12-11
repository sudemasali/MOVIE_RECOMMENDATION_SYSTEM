import pandas as pd
import re
def load_data(file_path):
    return pd.read_csv(file_path)

data=load_data("data/raw/imdb_top_1000.csv")

def preprocess_data(data):
    data=data.dropna()

    data['Genre'] = data['Genre'].apply(lambda x: x.split(','))
    data['Overview'] = data['Overview'].apply(lambda x: clean_text(x))
    return data

def clean_text(text):
    """
    Clean text data by removing punctuation and converting to lowercase.
    """
    # Küçük harfe çevir ve noktalama işaretlerini kaldır
    text = text.lower()  # Küçük harfe çevir
    text = re.sub(r'[^\w\s]', '', text)  # Noktalama işaretlerini kaldır
    return text