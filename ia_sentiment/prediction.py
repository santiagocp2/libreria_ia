
def predict_sentiment(model, text, tokenizer, max_len=100):
    """
    Predice el sentimiento de un texto dado utilizando el modelo entrenado.
    """
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    from ia_sentiment.data_preprocessing import clean_text
    text_clean = clean_text(text)
    sequence = tokenizer.texts_to_sequences([text_clean])
    padded = pad_sequences(sequence, maxlen=max_len, padding='post')
    prediction = model.predict(padded)
    sentiment = ['negative', 'neutral', 'positive'][prediction.argmax()]
    return sentiment


def get_sentiment_probabilities(prediction):
    """
    Devuelve las probabilidades asociadas a cada clase de sentimiento.
    """
    return prediction.tolist()


def analyze_emotion(text):
    """
    Analiza el tono emocional del texto. (Implementación básica utilizando TextBlob)
    """
    from textblob import TextBlob
    analysis = TextBlob(text)
    # Ejemplo: si la polaridad es positiva, se considera 'alegría', de lo contrario 'tristeza'
    emotion = "alegría" if analysis.sentiment.polarity > 0 else "tristeza"
    return emotion


def aspect_analysis(text):
    """
    Realiza un análisis de sentimientos por aspectos (por ejemplo: actuación, guion).
    Esta función es un placeholder y se debe ampliar según el método elegido.
    """
    # Ejemplo dummy
    return {"actuación": "positiva", "guion": "neutral", "dirección": "positiva"}


def extract_key_phrases(text):
    """
    Extrae frases clave del texto utilizando técnicas básicas (por ejemplo, con TextBlob).
    """
    from textblob import TextBlob
    blob = TextBlob(text)
    return blob.noun_phrases
