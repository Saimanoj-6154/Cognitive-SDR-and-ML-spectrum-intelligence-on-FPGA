import os
import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical

from ml.models.model import build_model

FEATURE_PATH = "dataset/features/features.npy"
LABEL_PATH   = "dataset/features/labels.npy"

MODEL_DIR = "ml/saved_models"
os.makedirs(MODEL_DIR, exist_ok=True)

X = np.load(FEATURE_PATH)
y = np.load(LABEL_PATH)

Y = to_categorical(y)

X_train, X_test, Y_train, Y_test, y_train, y_test = train_test_split(
    X, Y, y,
    test_size=0.2,
    random_state=42
)

model = build_model()

history = model.fit(
    X_train,
    Y_train,
    validation_split=0.1,
    epochs=20,
    batch_size=64
)

model.save(os.path.join(MODEL_DIR, "cognitive_radio_model.keras"))

np.save(os.path.join(MODEL_DIR, "X_test.npy"), X_test)
np.save(os.path.join(MODEL_DIR, "Y_test.npy"), Y_test)
np.save(os.path.join(MODEL_DIR, "y_test.npy"), y_test)

print("Training complete.")
