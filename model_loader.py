import tensorflow as tf

# Load the pre-trained MobileNetV2 model
model = tf.keras.applications.MobileNetV2(weights="imagenet")

# Save the model in Keras V3 format
model.save("mobilenet_model.keras")  # Correct format for Keras 3
