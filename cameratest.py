# main3.py

import cv2
import numpy as np
import tensorflow as tf

# ---------------------------
# 1. Load TFLite model
# ---------------------------
interpreter = tf.lite.Interpreter(model_path="/Users/sovaruninvan/Desktop/Roboticsss/mobilenetv2_rorobotics.tflite")
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

print("TFLite model loaded successfully!")
print("Input details:", input_details)
print("Output details:", output_details)

# ---------------------------
# 2. Define labels and actions
# ---------------------------
object_labels = ["Zip-top Can", "Book", "Newspaper", "Old School Bag"]
actions = ["Move Forward", "Turn Left", "Turn Right", "Stop / Reverse"]

# ---------------------------
# 3. Open camera
# ---------------------------
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open camera.")
    exit()

# ---------------------------
# 4. Real-time detection loop
# ---------------------------
while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Preprocess frame for MobileNetV2
    img = cv2.resize(frame, (224, 224))
    img = img.astype(np.float32) / 255.0  # normalize to [0,1]
    img = np.expand_dims(img, axis=0)

    # Run inference
    interpreter.set_tensor(input_details[0]['index'], img)
    interpreter.invoke()
    output = interpreter.get_tensor(output_details[0]['index'])

    # Map output to label and action
    pred_index = np.argmax(output)
    pred_label = object_labels[pred_index]
    pred_action = actions[pred_index]

    # Print to terminal
    print(f"Detected Object: {pred_label} -> Robot Action: {pred_action}")

    # Overlay on video
    cv2.putText(frame, f"{pred_label}: {pred_action}", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Camera", frame)

    # Press 'q' to quit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# ---------------------------
# 5. Release resources
# ---------------------------
cap.release()
cv2.destroyAllWindows()
