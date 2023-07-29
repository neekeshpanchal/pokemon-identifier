import os
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.models import load_model, Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.optimizers import Adam
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk

# Dataset path
data_path = "PokemonData"

# Image dimensions
img_width, img_height = 150, 150

# Optimizations for faster model training
# Set GPU memory growth (if available)
gpus = tf.config.experimental.list_physical_devices("GPU")
if gpus:
    try:
        for gpu in gpus:
            tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError as e:
        print(e)

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

# Load the trained model
if os.path.exists("pokemon_type_model.h5"):
    model = load_model("pokemon_type_model.h5")
    print("Model loaded successfully.")
else:
    print("Please train the model first!")

# Tkinter GUI code
def upload_image():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.jpg;*.png;*.jpeg")])
    if file_path:
        sample_img = Image.open(file_path)
        sample_img = sample_img.resize((img_width, img_height))
        sample_img_array = img_to_array(sample_img)
        sample_img_array = np.expand_dims(sample_img_array, axis=0)
        sample_img_array = sample_img_array / 255.0

        prediction = model.predict(sample_img_array)
        predicted_class = np.argmax(prediction[0])
        class_labels = list(train_generator.class_indices.keys())
        predicted_class_name = class_labels[predicted_class]

        # Display the uploaded image
        img = ImageTk.PhotoImage(sample_img)
        image_label.config(image=img)
        image_label.image = img

        result_label.config(text=f"Predicted Pokemon Type: {predicted_class_name}")

# Create the Tkinter application window
app = tk.Tk()
app.title("Pokemon Type Prediction")

# Button to upload the image
upload_button = tk.Button(app, text="Upload Image", command=upload_image)
upload_button.pack(pady=20)

# Label to display the uploaded image
image_label = tk.Label(app)
image_label.pack(pady=20)

# Label to display the prediction
result_label = tk.Label(app, text="", font=("Arial", 14))
result_label.pack()

# Start the Tkinter main loop
app.mainloop()
