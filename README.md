# Real-Time Facial Verification System

This project is a real-time facial verification application that uses a Siamese neural network to verify an individual’s identity by comparing a live webcam image with a set of stored reference images. Built with TensorFlow, Kivy, and OpenCV, the system includes a custom L1 distance layer for image similarity calculation and a structured dataset for model training and verification.

## Project Structure

```
.
├── application_data
│   ├── input_image              # Stores captured webcam images for verification
│   └── verification_images       # Stores reference images for matching
├── data
│   ├── anchor                   # Images for the 'anchor' class (used in training)
│   ├── negative                 # Images for the 'negative' class (used in training)
│   └── positive                 # Images for the 'positive' class (used in training)
├── data_preprocessing.py         # Script to create image pairs for training
├── layers.py                     # Custom L1 distance layer definition
├── face_rec.py                   # Main application script (Kivy app with webcam)
├── siamese_model.py              # Model definition and training script
└── README.md                     # Project documentation
```

## Project Components

### 1. `face_rec.py`
This is the main Kivy application that serves as the user interface for the facial verification system. It captures images from the webcam, processes them, and passes them to the Siamese model for comparison with pre-stored verification images. The interface includes:
- **Webcam Feed**: Displays a live feed from the webcam.
- **Verify Button**: Captures an image from the feed and verifies it.
- **Verification Label**: Indicates whether the captured image matches any stored reference images.

The app uses `OpenCV` for webcam access and `TensorFlow` to load the pre-trained Siamese model, which is customized with an L1 distance layer for measuring image similarity.

### 2. `siamese_model.py`
This script defines the Siamese neural network model for facial verification. It includes:
- **CNN Branch**: A shared feature extractor that generates embeddings from input images.
- **L1 Distance Layer**: A custom layer that computes the L1 (absolute) distance between two embeddings to assess similarity.
- **Model Compilation**: The model is compiled using binary cross-entropy as the loss function, optimized for verifying matching and non-matching pairs.
- **Training Data**: Pairs of images (matching and non-matching) are generated to train the network.

The model is trained on pairs of images, learning to distinguish between similar and dissimilar pairs, and is saved in `siamesemodel.h5` for loading during verification.

### 3. `data_preprocessing.py`
This script prepares training data by creating pairs of images:
- **Anchor Images**: Images of the individual to be recognized.
- **Positive and Negative Pairs**: Matching (positive) and non-matching (negative) pairs for model training.
- **Preprocessing**: Images are resized and scaled to be suitable for training the network.

The function in this script organizes the data into anchor-positive-negative pairs, helping to structure the dataset for effective model learning.

### 4. `layers.py`
Defines the custom `L1Dist` layer for the model, which computes the absolute difference between two image embeddings. This layer enables the Siamese model to quantify similarity between pairs, which is essential for the verification process. The layer is defined as a subclass of `Layer` from Keras, ensuring compatibility with the TensorFlow/Keras framework and allowing the model to calculate a distance metric between embeddings.

### 5. `application_data`
This directory stores image data for verification:
- **input_image**: Stores the image captured from the webcam during a verification attempt.
- **verification_images**: Stores reference images that the model will compare with the captured image to verify identity.

## Getting Started

### Prerequisites
1. Python 3.x
2. TensorFlow
3. OpenCV
4. Kivy

Install the dependencies using:
```bash
pip install tensorflow opencv-python kivy
```

### Setup and Training

1. **Organize Data**:
   - Place verification images in `application_data/verification_images`.
   - Organize anchor, positive, and negative images in `data/` for training.

2. **Train the Model**:
   Run `siamese_model.py` to train the model on your dataset:
   ```bash
   python siamese_model.py
   ```
   This will generate `siamesemodel.h5`, which is saved and later loaded by the Kivy app.

3. **Run the Application**:
   Launch the Kivy application to perform real-time verification:
   ```bash
   python face_rec.py
   ```

## How It Works

1. **Image Capture**: The app captures an image from the webcam, which is preprocessed (resized, normalized).
2. **Embedding Comparison**: The captured image is compared with each image in `verification_images` using the Siamese network.
3. **Verification Decision**: The app computes a verification score based on the similarity of embeddings. If the score exceeds a predefined threshold, the individual is verified; otherwise, they are unverified.

## Usage

- **Verify Button**: Press to capture and verify an image.
- **Verification Thresholds**: Adjust `detection_threshold` and `verification_threshold` in `face_rec.py` for sensitivity.

## Troubleshooting

- **File Not Found Error**: Ensure that the `input_image` and `verification_images` folders exist within `application_data`.
- **Webcam Issues**: Adjust the device index (`self.capture = cv2.VideoCapture(0)`) if the webcam feed does not display.

## Future Improvements

- **Expand Training Data**: Add more diverse images to improve model generalization.
- **Optimize Performance**: Convert the model to TensorFlow Lite for faster inference on resource-constrained devices.
- **Enhanced UI**: Improve the app interface for better user experience.

## Contributing

Contributions are welcome! Please open issues for feature requests or bug reports.
