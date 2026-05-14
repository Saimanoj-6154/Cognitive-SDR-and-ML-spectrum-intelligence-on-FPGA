import numpy as np
import tensorflow as tf

MODEL_PATH = "ml/saved_models/cognitive_radio_model.keras"

model = tf.keras.models.load_model(MODEL_PATH)

sample = np.random.rand(1, 8)

prediction = model.predict(sample)

channel = np.argmax(prediction)

print(f"Predicted Best Channel: Ch{channel+1}")
print(prediction)
