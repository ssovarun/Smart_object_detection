import tensorflow as tf

# Load TFLite model
interpreter = tf.lite.Interpreter(model_path="mobilenet_robotic.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded successfully!")
print("Input details:", input_details)
print("Output details:", output_details)
