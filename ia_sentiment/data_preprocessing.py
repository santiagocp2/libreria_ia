
import pandas as pd
import re
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import spacy


nlp = spacy.load('en_core_web_sm')


def load_data(filepath):
    """
    Carga datos desde un archivo CSV o JSON.
    """
    if filepath.endswith('.csv'):
        data = pd.read_csv(filepath)
    elif filepath.endswith('.json'):
        data = pd.read_json(filepath)
    else:
        raise ValueError("Formato de archivo no soportado")
    return data


def clean_text(text):
    """
    Limpia el texto eliminando URLs, números, caracteres especiales y normaliza a minúsculas.
    """
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)  # Elimina URLs
    text = re.sub(r'\d+', '', text)             # Elimina números
    # Elimina caracteres especiales
    text = re.sub(r'[^\w\s]', '', text)
    text = text.strip()
    return text


def tokenize_and_lemmatize(text):
    """
    Tokeniza el texto y aplica lematización utilizando nltk y spacy.
    """
    tokens = word_tokenize(text)
    tokens = [
        word for word in tokens if word not in stopwords.words('english')]
    doc = nlp(" ".join(tokens))
    lemmas = [token.lemma_ for token in doc]
    return lemmas


def encode_labels(ratings):
    """
    Convierte puntuaciones numéricas en etiquetas de sentimiento:
      7-10 → 'positive'
      4-6  → 'neutral'
      1-3  → 'negative'
    """
    def assign_label(rating):
        if rating >= 7:
            return 'positive'
        elif rating >= 4:
            return 'neutral'
        else:
            return 'negative'
    return ratings.apply(assign_label)


def vectorize_text(texts, num_words=10000, max_len=100):
    """
    Transforma una lista de textos en secuencias numéricas utilizando un Tokenizer de Keras.
    """
    from tensorflow.keras.preprocessing.text import Tokenizer
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    tokenizer = Tokenizer(num_words=num_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=max_len, padding='post')
    return padded, tokenizer


def split_dataset(data, labels, train_size=0.8, val_size=0.1):
    """
    Divide el dataset en conjuntos de entrenamiento, validación y prueba.
    """
    from sklearn.model_selection import train_test_split
    X_train, X_temp, y_train, y_temp = train_test_split(
        data, labels, train_size=train_size, random_state=42)
    relative_val = val_size / (1 - train_size)
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, train_size=relative_val, random_state=42)
    return X_train, X_val, X_test, y_train, y_val, y_test
