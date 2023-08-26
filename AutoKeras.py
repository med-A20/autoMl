import numpy as np
import tensorflow as tf
from tensorflow.keras.datasets import mnist

import autokeras as ak

# Load Data
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# Initialize the image classifier
clf = ak.ImageClassifier(overwrite=True, max_trials=1)

# Train the models
clf.fit(x_train, y_train, epochs=5)

# Predict with the best model.
predicted_y = clf.predict(x_test)
print(predicted_y)

# Evaluate the best model with testing data.
print(clf.evaluate(x_test, y_test))

# Export model and save it
model = clf.export_model()
model.save("AutoMLModel.h5")