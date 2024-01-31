
import numpy as np
from tkinter import Tk, Label, Button, filedialog, StringVar  # Import StringVar
from PIL import Image, ImageTk
import tensorflow as tf

# Load the pre-trained model
model =tf.keras.models.load_model("C:/Users/Ayesha Shaikh/Desktop/Image Classificationfyp/cats-and-dogs/train/VGG_model.h5", compile=False)

class ImageClassifierGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Cat and Dog Image Classifier")
        self.root.geometry("400x400")

        self.upload_button = Button(root, text="Upload Image", command=self.upload_image)
        self.upload_button.pack()

        self.predict_button = Button(root, text="Predict", command=self.predict_image)
        self.predict_button.pack()

        self.image_label = Label(root)
        self.image_label.pack()

        # Create a StringVar to hold the text of the prediction label
        self.prediction_text = StringVar()  # Use StringVar from the tkinter module
        self.prediction_label = Label(root, textvariable=self.prediction_text, font=("Helvetica", 12))
        self.prediction_label.pack()

    def predict_image(self):
        if hasattr(self, "image_array"):
            # Add an extra dimension to the image array
            image_array = np.expand_dims(self.image_array, axis=0)

            # Make a prediction
            prediction = model.predict(image_array)

            # Determine the predicted class based on a threshold (e.g., 0.5)
            threshold = 0.5
            predicted_class = "dog" if prediction > threshold else "cat"

            # Update the text of the prediction label
            self.prediction_text.set(f"Prediction: It is a {predicted_class}")

    def upload_image(self):
        # Clear the previous prediction
        self.prediction_text.set("")

        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png;*.jpg;*.jpeg;*.gif;*.bmp")])
        if file_path:
            # Load and preprocess the selected image
            image = Image.open(file_path)
            image = image.resize((224, 224))  # Resize to (224, 224)
            self.image_array = np.array(image)

            # Display the image
            img_tk = ImageTk.PhotoImage(image)
            self.image_label.config(image=img_tk)
            self.image_label.image = img_tk

if __name__ == "__main__":
    root = Tk()
    app = ImageClassifierGUI(root)
    root.mainloop()