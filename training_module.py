import os
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam


# Dataset path
data_path = "PokemonData"

# Image dimensions
img_width, img_height = 150, 150

# Prepare data for training
train_datagen = ImageDataGenerator(
    rescale=1.0/255.0,
    validation_split=0.2
)

train_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(img_width, img_height),
    batch_size=32,
    subset="training",
    class_mode="categorical"
)

validation_generator = train_datagen.flow_from_directory(
    data_path,
    target_size=(img_width, img_height),
    batch_size=32,
    subset="validation",
    class_mode="categorical"
)

# Build the Neural Network Model
model = Sequential()

model.add(Conv2D(32, (3, 3), activation="relu", input_shape=(img_width, img_height, 3)))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(64, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Conv2D(128, (3, 3), activation="relu"))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(Flatten())
model.add(Dense(128, activation="relu"))
model.add(Dropout(0.5))
model.add(Dense(len(train_generator.class_indices), activation="softmax"))

model.compile(optimizer=Adam(learning_rate=0.001),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Train the Model
epochs = 20

history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // train_generator.batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // validation_generator.batch_size
)

# Evaluate the Model
# Plot accuracy and loss curves
plt.plot(history.history["accuracy"])
plt.plot(history.history["val_accuracy"])
plt.title("Model Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

plt.plot(history.history["loss"])
plt.plot(history.history["val_loss"])
plt.title("Model Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend(["Train", "Validation"], loc="upper left")
plt.show()

# Evaluate on the validation set
score = model.evaluate(validation_generator, verbose=0)
print("Validation Loss:", score[0])
print("Validation Accuracy:", score[1])

# Making Predictions
# Load a sample image for prediction
sample_img_path = "image.png"
sample_img = load_img(sample_img_path, target_size=(img_width, img_height))
sample_img_array = img_to_array(sample_img)
sample_img_array = np.expand_dims(sample_img_array, axis=0)

# Normalize the pixel values
sample_img_array = sample_img_array / 255.0

# Make the prediction
prediction = model.predict(sample_img_array)
predicted_class = np.argmax(prediction[0])
class_labels = list(train_generator.class_indices.keys())
predicted_class_name = class_labels[predicted_class]

print("Predicted Pokemon Type:", predicted_class_name)
