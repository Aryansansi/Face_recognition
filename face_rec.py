from kivy.app import App
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.image import Image
from kivy.uix.button import Button
from kivy.uix.label import Label
from kivy.clock import Clock
from kivy.graphics.texture import Texture
from kivy.logger import Logger
import cv2
import tensorflow as tf
from tensorflow.keras.layers import Layer # type: ignore
import os
import numpy as np

# Define the custom L1Dist layer
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super(L1Dist, self).__init__(**kwargs)

    def call(self, inputs):
        x, y = inputs
        return tf.abs(x - y)

# Build app and layout 
class CamApp(App):

    def build(self):
        # Main layout components 
        self.web_cam = Image(size_hint=(1, .8))
        self.button = Button(text="Verify", on_press=self.verify, size_hint=(1, .1))
        self.verification_label = Label(text="Verification Uninitiated", size_hint=(1, .1))

        # Add items to layout
        layout = BoxLayout(orientation='vertical')
        layout.add_widget(self.web_cam)
        layout.add_widget(self.button)
        layout.add_widget(self.verification_label)

        # Load tensorflow/keras model with custom_objects for L1Dist layer
        self.model = tf.keras.models.load_model('siamesemodel.h5', custom_objects={'L1Dist': L1Dist})

        # Setup video capture device
        self.capture = cv2.VideoCapture(0)  # Adjust to correct device index if needed
        Clock.schedule_interval(self.update, 1.0 / 33.0)
        
        return layout

    # Run continuously to get webcam feed
    def update(self, *args):
        # Read frame from opencv
        ret, frame = self.capture.read()
        if not ret:
            return

        frame = frame[120:120+250, 200:200+250, :]

        # Flip horizontally and convert image to texture
        buf = cv2.flip(frame, 0).tobytes()
        img_texture = Texture.create(size=(frame.shape[1], frame.shape[0]), colorfmt='bgr')
        img_texture.blit_buffer(buf, colorfmt='bgr', bufferfmt='ubyte')
        self.web_cam.texture = img_texture

    # Load image from file and convert to 100x100px
    def preprocess(self, file_path):
        # Read in image from file path
        byte_img = tf.io.read_file(file_path)
        # Load in the image 
        img = tf.io.decode_jpeg(byte_img)
        
        # Preprocessing steps - resizing the image to be 100x100x3
        img = tf.image.resize(img, (100, 100))
        # Scale image to be between 0 and 1 
        img = img / 255.0
        
        # Return image
        return img

    # Verification function to verify person
    def verify(self, *args):
        # Specify thresholds
        detection_threshold = 0.99
        verification_threshold = 0.8

        # Capture input image from our webcam
        SAVE_PATH = os.path.join('application_data', 'input_image', 'input_image.jpg')
        ret, frame = self.capture.read()
        if not ret:
            self.verification_label.text = "Capture Failed"
            return
        
        frame = frame[120:120+250, 200:200+250, :]
        cv2.imwrite(SAVE_PATH, frame)

        # Build results array
        results = []
        input_img = self.preprocess(SAVE_PATH)  # Preprocess the input image
        for image in os.listdir(os.path.join('application_data', 'verification_images')):
            validation_img_path = os.path.join('application_data', 'verification_images', image)
            validation_img = self.preprocess(validation_img_path)

            # Make Predictions 
            # Ensure to pass both images as a list
            result = self.model.predict([np.expand_dims(input_img, axis=0), np.expand_dims(validation_img, axis=0)])
            results.append(result)

        # Detection Threshold: Metric above which a prediction is considered positive 
        detection = np.sum(np.array(results) > detection_threshold)

        # Verification Threshold: Proportion of positive predictions / total positive samples 
        verification = detection / len(os.listdir(os.path.join('application_data', 'verification_images'))) 
        verified = verification > verification_threshold

        # Set verification text 
        self.verification_label.text = 'Verified' if verified else 'Unverified'

        # Log out details
        Logger.info(f"Results: {results}")
        Logger.info(f"Detection: {detection}")
        Logger.info(f"Verification: {verification}")
        Logger.info(f"Verified: {verified}")

        return results, verified

if __name__ == '__main__':
    CamApp().run()
