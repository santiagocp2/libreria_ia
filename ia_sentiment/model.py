
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense


def build_model(vocab_size, input_length):
    """
    Construye y compila un modelo LSTM para clasificación de sentimientos.
    """
    model = Sequential()
    model.add(Embedding(input_dim=vocab_size,
              output_dim=128, input_length=input_length))
    model.add(LSTM(64))
    # Tres clases: positivo, neutral y negativo
    model.add(Dense(3, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam', metrics=['accuracy'])
    return model


def train_model(model, train_data, train_labels, val_data, val_labels, epochs=10, batch_size=32):
    """
    Entrena el modelo con los datos de entrenamiento y validación.
    """
    history = model.fit(train_data, train_labels,
                        validation_data=(val_data, val_labels),
                        epochs=epochs,
                        batch_size=batch_size)
    return history
