# ==============================================================================
# Step 1: Import All Necessary Libraries
# ==============================================================================
import tensorflow as tf
from keras.models import Model
from keras.layers import Input, Conv2D, GlobalAveragePooling2D, Dropout, Dense
from keras.applications import MobileNetV2
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

print("TensorFlow Version:", tf.__version__)

# ==============================================================================
# Step 2: Load and Prepare the Dataset from Folders
# ==============================================================================

# ---!!! IMPORTANT !!!---
# Update this path to your main dataset folder
DATASET_PATH = r'C:\OMEGA\Project-MoodFit\FerEmotions'

# Check if the path exists
if not os.path.exists(DATASET_PATH):
    print(f"Error: Directory not found at '{DATASET_PATH}'")
    print("Please make sure the path to your 'FerEmotions' folder is correct.")
    exit()

# Define paths for training and testing directories
train_dir = os.path.join(DATASET_PATH, 'train')
test_dir = os.path.join(DATASET_PATH, 'test')

# Define image and training parameters
IMG_WIDTH, IMG_HEIGHT = 48, 48
BATCH_SIZE = 64
NUM_CLASSES = 7

print("Loading data using image_dataset_from_directory...")

# Create the training dataset and a validation split from it
train_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='training',
    seed=123
)

# Create the validation dataset
validation_dataset = tf.keras.utils.image_dataset_from_directory(
    train_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    validation_split=0.2,
    subset='validation',
    seed=123
)

# Create the test dataset
test_dataset = tf.keras.utils.image_dataset_from_directory(
    test_dir,
    labels='inferred',
    label_mode='categorical',
    color_mode='grayscale',
    image_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE
)

# Configure the dataset for performance
AUTOTUNE = tf.data.AUTOTUNE
train_dataset = train_dataset.cache().prefetch(buffer_size=AUTOTUNE)
validation_dataset = validation_dataset.cache().prefetch(buffer_size=AUTOTUNE)
test_dataset = test_dataset.cache().prefetch(buffer_size=AUTOTUNE)

# ==============================================================================
# Step 3: Build the Model using Transfer Learning (CORRECTED METHOD)
# ==============================================================================
print("Building the Transfer Learning model with MobileNetV2...")

# Load the MobileNetV2 model, pre-trained on ImageNet.
# We specify the input shape as (48, 48, 3) because it expects 3-channel images.
base_model = MobileNetV2(
    input_shape=(IMG_WIDTH, IMG_HEIGHT, 3),
    include_top=False, # Do not include the final classification layer
    weights='imagenet'
)

# Freeze the layers of the base model so they don't get updated during initial training
base_model.trainable = False

# Now, we build our custom model on top of it using the Functional API
# 1. Define the actual input layer for our 1-channel grayscale images
inputs = Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1))

# 2. Add a Conv2D layer to convert our 1-channel image to a 3-channel image
x = Conv2D(3, (3, 3), padding='same')(inputs)

# 3. Pass the 3-channel image to the frozen base_model
x = base_model(x, training=False)

# 4. Add our custom classification head
x = GlobalAveragePooling2D()(x)
x = Dropout(0.5)(x)
x = Dense(128, activation='relu')(x)
outputs = Dense(NUM_CLASSES, activation='softmax')(x)

# 5. Create the final model
model = Model(inputs=inputs, outputs=outputs)

model.summary()


# ==============================================================================
# Step 4: Compile the Model and Define Callbacks
# ==============================================================================
print("Compiling the model...")
model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define Callbacks to prevent overfitting and adjust learning rate
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=10,
    restore_best_weights=True
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,
    patience=5,
    min_lr=1e-6
)

callbacks = [early_stopping, reduce_lr]

# ==============================================================================
# Step 5: Train the Model
# ==============================================================================
print("Starting model training with Transfer Learning...")

EPOCHS = 100

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("Training complete.")

# ==============================================================================
# Step 6: Evaluate and Plot Results
# ==============================================================================
print("Evaluating the model on the test data...")
score = model.evaluate(test_dataset, verbose=0)
print(f"\nTest Loss: {score[0]:.4f}")
print(f"Test Accuracy: {score[1]:.4f}")

print("Generating training and validation graphs...")
acc = history.history['accuracy']
val_acc = history.history['val_accuracy']
loss = history.history['loss']
val_loss = history.history['val_loss']

epochs_range = range(1, len(acc) + 1)

plt.figure(figsize=(12, 5))
plt.subplot(1, 2, 1)
plt.plot(epochs_range, acc, 'bo-', label='Training Accuracy')
plt.plot(epochs_range, val_acc, 'ro-', label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()

plt.subplot(1, 2, 2)
plt.plot(epochs_range, loss, 'bo-', label='Training Loss')
plt.plot(epochs_range, val_loss, 'ro-', label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()

plt.savefig('training_performance_transfer_learning.png')
print("Graphs have been saved as 'training_performance_transfer_learning.png'")
plt.show()

# ==============================================================================
# Step 7: Save the Trained Model
# ==============================================================================
MODEL_SAVE_PATH = "emotion_detection_model_transfer_learning.h5"
print(f"Saving the model to '{MODEL_SAVE_PATH}'...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully!")