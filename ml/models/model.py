import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras.layers import Dense

def build_model(input_dim=8, num_classes=6):

    model = Sequential([
        Dense(16, activation='relu', input_shape=(input_dim,)),
        Dense(8, activation='relu'),
        Dense(num_classes, activation='softmax')
    ])

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model
