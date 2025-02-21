
from sklearn.preprocessing import LabelEncoder
from ia_sentiment import load_data, clean_text, tokenize_and_lemmatize, encode_labels, vectorize_text, split_dataset
from ia_sentiment import build_model, train_model, predict_sentiment, analyze_emotion, aspect_analysis, extract_key_phrases
from tensorflow.keras.utils import to_categorical

# 1. Cargar el dataset
# Asegúrate de que el CSV contenga columnas como 'ReviewText' y 'Rating'
data = load_data("reviews.csv")

# 2. Preprocesar el texto
data['CleanText'] = data['ReviewText'].apply(clean_text)
data['Tokens'] = data['CleanText'].apply(tokenize_and_lemmatize)

# 3. Codificar las etiquetas a partir de la columna 'Rating'
data['SentimentLabel'] = encode_labels(data['Rating'])

# 4. Vectorizar el texto (por ejemplo, utilizando la columna 'CleanText')
padded_texts, tokenizer = vectorize_text(data['CleanText'])

# 5. Convertir las etiquetas a formato categórico (one-hot encoding)
le = LabelEncoder()
labels_encoded = le.fit_transform(data['SentimentLabel'])
labels_cat = to_categorical(labels_encoded)

# 6. Dividir el dataset
X_train, X_val, X_test, y_train, y_val, y_test = split_dataset(
    padded_texts, labels_cat)

# 7. Construir y entrenar el modelo
vocab_size = 10000  # Debe coincidir con el parámetro en vectorize_text
input_length = padded_texts.shape[1]
model = build_model(vocab_size, input_length)
history = train_model(model, X_train, y_train, X_val,
                      y_val, epochs=5, batch_size=32)

# 8. Realizar una predicción
sample_text = "La película fue increíble, con actuaciones excepcionales y un guion que atrapa."
predicted_sentiment = predict_sentiment(model, sample_text, tokenizer)
emotion = analyze_emotion(sample_text)
aspects = aspect_analysis(sample_text)
key_phrases = extract_key_phrases(sample_text)

print("Sentimiento predicho:", predicted_sentiment)
print("Emoción detectada:", emotion)
print("Análisis por aspectos:", aspects)
print("Frases clave:", key_phrases)
