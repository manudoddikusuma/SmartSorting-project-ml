import tensorflow as tf
from tensorflow.keras.models import load_model

# Path to your saved model file
model_path = 'healthy_vs_rotten.h5'

# Load the model
loaded_model = load_model(model_path)

# Now you can use loaded_model to make predictions
loaded_model.summary() # To see its architecture again