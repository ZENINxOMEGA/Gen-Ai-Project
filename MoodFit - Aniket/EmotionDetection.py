# Step 1: Import All Necessary Libraries

import tensorflow as tf
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Dropout, Flatten, Dense, BatchNormalization, Rescaling
from keras.layers import RandomFlip, RandomRotation, RandomZoom
from keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import os

print("TensorFlow Version:", tf.__version__)

# Step 2: Load and Prepare the Dataset from Folders

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

# Step 3: Create Data Augmentation Layer

data_augmentation = Sequential(
    [
        RandomFlip("horizontal"),
        RandomRotation(0.1),
        RandomZoom(0.1),
    ],
    name="data_augmentation",
)

# Step 4: Build the CNN Model Architecture (with Augmentation)

print("Building the CNN model architecture...")

model = Sequential([
    # Input layer
    tf.keras.Input(shape=(IMG_WIDTH, IMG_HEIGHT, 1)),
    
    # Apply data augmentation ONLY during training
    data_augmentation,
    
    # Rescale pixel values from [0, 255] to [0, 1]
    Rescaling(1./255),
    
    # Block 1
    Conv2D(32, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 2
    Conv2D(64, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Block 3
    Conv2D(128, (3, 3), padding='same', activation='relu'),
    BatchNormalization(),
    MaxPooling2D(pool_size=(2, 2)),
    Dropout(0.25),

    # Flattening and Dense layers
    Flatten(),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.5),

    # Output Layer
    Dense(NUM_CLASSES, activation='softmax')
])

model.summary()

# Step 5: Compile the Model and Define Callbacks

print("Compiling the model...")
model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

# Define Callbacks to prevent overfitting and adjust learning rate
early_stopping = EarlyStopping(
    monitor='val_loss',         # Metric to monitor
    patience=10,                # Number of epochs with no improvement to wait before stopping
    restore_best_weights=True   # Restores model weights from the epoch with the best value of the monitored metric
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',
    factor=0.2,                 # Factor by which the learning rate will be reduced
    patience=5,                 # Number of epochs with no improvement to wait before reducing LR
    min_lr=1e-6                 # Lower bound on the learning rate
)

callbacks = [early_stopping, reduce_lr]

# Step 6: Train the Model

print("Starting model training... (with anti-overfitting measures)")

# We can set a higher number of epochs because EarlyStopping will halt the training automatically
EPOCHS = 100 

history = model.fit(
    train_dataset,
    validation_data=validation_dataset,
    epochs=EPOCHS,
    callbacks=callbacks, # Pass the callbacks here
    verbose=1
)

print("Training complete.")

# Step 7: Evaluate and Plot Results

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

plt.savefig('training_performance_graphs_fixed.png')
print("Graphs have been saved as 'training_performance_graphs_fixed.png'")
plt.show()

# Step 8: Save the Trained Model

MODEL_SAVE_PATH = "emotion_detection_model_fixed.h5"
print(f"Saving the model to '{MODEL_SAVE_PATH}'...")
model.save(MODEL_SAVE_PATH)
print("Model saved successfully!")