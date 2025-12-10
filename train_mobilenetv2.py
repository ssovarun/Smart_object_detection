import tensorflow as tf
from tensorflow.keras import layers, models
import os

# ------------------------------
# PATH TO YOUR DATASET
# ------------------------------
DATASET_PATH = "/Users/sovaruninvan/Desktop/Roboticsss/robotics_dataset"

IMG_SIZE = 224
BATCH_SIZE = 32
AUTOTUNE = tf.data.AUTOTUNE

# ------------------------------
# LOAD DATA FROM FOLDERS
# ------------------------------
train_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "train"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

val_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "valid"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

test_ds = tf.keras.preprocessing.image_dataset_from_directory(
    os.path.join(DATASET_PATH, "test"),
    image_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE
)

# Prefetch for performance
train_ds = train_ds.prefetch(AUTOTUNE)
val_ds = val_ds.prefetch(AUTOTUNE)
test_ds = test_ds.prefetch(AUTOTUNE)

# ------------------------------
# DATA AUGMENTATION
# ------------------------------
data_augmentation = tf.keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomBrightness(0.2)
])

# ------------------------------
# BUILD MODEL (TRANSFER LEARNING)
# ------------------------------
base_model = tf.keras.applications.MobileNetV2(
    input_shape=(IMG_SIZE, IMG_SIZE, 3),
    include_top=False,
    weights="imagenet"
)
base_model.trainable = False  # Freeze base model for head training

# Add custom layers
inputs = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 3))
x = data_augmentation(inputs)  # apply augmentation
x = tf.keras.applications.mobilenet_v2.preprocess_input(x)
x = base_model(x, training=False)
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(128, activation="relu")(x)
x = layers.Dropout(0.3)(x)
outputs = layers.Dense(4, activation="softmax")(x)

model = models.Model(inputs, outputs)

# ------------------------------
# COMPILE MODEL
# ------------------------------
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-4),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ------------------------------
# TRAIN HEAD ONLY
# ------------------------------
history = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ------------------------------
# FINE-TUNE LAST 20 LAYERS
# ------------------------------
base_model.trainable = True
for layer in base_model.layers[:-20]:
    layer.trainable = False

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-5),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"]
)

history_ft = model.fit(
    train_ds,
    validation_data=val_ds,
    epochs=10
)

# ------------------------------
# SAVE MODEL
# ------------------------------
model.save("mobilenetv2_robotics.h5")
print("Training complete! Model saved as mobilenetv2_robotics.h5")










if action_start_time is None:
    if label == "Bag": move_forward(bot)
    elif label == "Book": move_backward(bot)
    elif label == "Can": turn_right(bot)
    elif label == "Newspaper": turn_left(bot)



if elapsed >= action_duration:
    set_tank(bot, 0, 0)
    action_start_time = None
    current_action = None

