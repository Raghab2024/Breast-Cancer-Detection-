import tkinter as tk
from tkinter import filedialog
import os 
import shutil
import subprocess
import tensorflow
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense,Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
import os
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.preprocessing import image
import numpy as np
import cv2
import time
import matplotlib.pyplot as plt

# Define the model
model = Sequential()
model.add(Conv2D(32, (6, 6), activation='relu', input_shape=(224, 224, 3)))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(64, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))
model.add(Conv2D(128, (6, 6), activation='relu'))
model.add(MaxPooling2D(pool_size=(4, 4)))


# Flatten layer
model.add(Flatten())

# Fully connected layers
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))  # Adding dropout for regularization
model.add(Dense(1, activation='sigmoid'))

def rum():
    start_time = time.time()
    # Compile the model with Adam optimizer and learning rate scheduling
    optimizer = tf.keras.optimizers.Adam(learning_rate=0.0001)
    model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])

    # Data generators
    base_path = 'data'  #will be change base file path
    train_dir = os.path.join(base_path, 'train')
    test_dir = os.path.join(base_path, 'test')

    datagen = ImageDataGenerator(rescale=1./255)

    train_generator = datagen.flow_from_directory(
        train_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        classes=['benign', 'malignant']
    )
    test_generator = datagen.flow_from_directory(
        test_dir,
        target_size=(224, 224),
        batch_size=32,
        class_mode='binary',
        classes=['benign', 'malignant']
    )
    # Train the model
    history = model.fit(
        train_generator,
        epochs=30,
        validation_data=test_generator
    )

    # Evaluate the model
    test_loss, test_accuracy = model.evaluate(test_generator)
    print(f'Test Accuracy: {test_accuracy}')
    print("Run Successfully")
    # Extracting training history
    train_accuracy = history.history['accuracy']
    val_accuracy = history.history['val_accuracy']
    train_loss = history.history['loss']
    val_loss = history.history['val_loss']

    epochs = range(1, len(train_accuracy) + 1)

    # Plotting accuracy
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_accuracy, 'b', label='Training Accuracy')
    plt.plot(epochs, val_accuracy, 'r', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()

    # Show plot
    plt.show()

    # Plotting loss
    plt.figure(figsize=(12, 6))
    plt.plot(epochs, train_loss, 'b', label='Training Loss')
    plt.plot(epochs, val_loss, 'r', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    # Show plot
    plt.show()
    output1="Run Successfully \n Test Accuracy:  "+str(test_accuracy)+""
    output_label1.config(text=output1)

    end_time = time.time()
    total_time = end_time - start_time
    print("Total Time Taken:", total_time)
    output1 += f"\nTotal Time Taken: {total_time} seconds"
    output_label1.config(text=output1)

def upload_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        # Load the image
        fixed_name = "input.png"
        save_folder = "input"
        # Define the destination path
        destination_path = os.path.join(save_folder, fixed_name)
        # Move the uploaded file to the specified folder with the fixed name
        shutil.copy(file_path, destination_path)
        
        print("Image Uploaded Successfully")
        output2="Image Uploaded Successfully"
        output_label2.config(text=output2)

def test_result():
     img_path = 'input/input.png'
     img = image.load_img(img_path, target_size=(224, 224))
     img_array = image.img_to_array(img)
     img_array = np.expand_dims(img_array, axis=0)

     prediction = model.predict(img_array)
     if prediction[0][0] > 0.5:
         print("Malignant")
         output3="Cells are Malignant"
         output_label3.config(text=output3)
     else:
         print("Benign")
         output3="Cells are Benign"
         output_label3.config(text=output3)

# Create main window
root = tk.Tk()
root.title("Breast Cancer Detection")
root.configure(borderwidth=5, relief="ridge",)

custom_font = ('Helvetica', 20)
# Add some text
text_label = tk.Label(root, text="Breast Cancer Detection",font=custom_font)
text_label.pack()

# Add a photo
photo = tk.PhotoImage(file="image/breast.png")  # Change "sample_image.png" to your image file path
# Define the desired dimensions for the image
new_width = 350
new_height = 200
# Resize the image to the desired dimensions
resized_image = photo.subsample(photo.width() // new_width, photo.height() // new_height)
photo_label = tk.Label(root, image=resized_image)
photo_label.pack()

# Compile The Model Button with Icon
compile_icon = tk.PhotoImage(file="image/compile.png")
compile_button = tk.Button(root, text="Compile The Model", width=200, pady=10,command=rum, font=2, bg="light blue", fg="black", image=compile_icon, compound="right")
compile_button.pack(pady=10)
output_label1 = tk.Label(root, text="")
output_label1.pack(pady=2)

# Add Image Upload Button with Icon
upload_icon = tk.PhotoImage(file="image/upload.png")
upload_button = tk.Button(root, text="Upload Image", command=upload_image, width=200,pady=10, font=2, bg="light green", fg="black", image=upload_icon, compound="right")
upload_button.pack(pady=10)
output_label2 = tk.Label(root, text="")
output_label2.pack(pady=2)

# Get Tested Result Button with Icon
test_icon = tk.PhotoImage(file="image/test.png")
test_button = tk.Button(root, text="Get Tested Result", width=200, pady=10, font=2,command=test_result, bg="#D895DA", fg="black", image=test_icon, compound="right")
test_button.pack(pady=10)
output_label3 = tk.Label(root, text="", font=6)
output_label3.pack(pady=2)

root.mainloop()
