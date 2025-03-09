# write2digit - Handwritten Digit Classification Web App

Write2Digit is a web-based application that allows users to draw a digit (0-9) and get it classified using a deep learning model.

## Features
- Draw a digit on a canvas and get a prediction.
- Uses a trained Convolutional Neural Network (CNN) model.
- Flask-based backend with TensorFlow/Keras for inference.
- Interactive UI for an engaging user experience.

## Technologies Used
- **Frontend:** HTML, CSS, JavaScript (Canvas API)
- **Backend:** Flask, Python
- **Machine Learning:** TensorFlow/Keras
- **Deployment:** Flask server (can be extended to cloud platforms like AWS, GCP, or Heroku)

## Installation and Setup
### 1. Clone the Repository
```sh
git clone https://github.com/Sulojan2027/write2digit-Digit_classifier.git
cd write2digit-Digit_classifer
```

### 2. Create a Virtual Environment (Recommended)
```sh
python -m venv venv
source venv/bin/activate  # On Windows use: venv\Scripts\activate
```

### 3. Install Dependencies
```sh
pip install -r requirements.txt
```

### 4. Run the Flask Server
```sh
python app.py
```

### 5. Access the Application
After install all dependencies, run "app.py" and access web app:

## How It Works
1. The user draws a digit (0-9) on the web interface.
2. The canvas image is sent to the Flask backend.
3. The backend processes the image and feeds it to the trained CNN model.
4. The model predicts the digit and returns the result.
5. The prediction is displayed on the UI.

## Dataset and Model
- The model is trained using the **MNIST dataset**, which consists of 60,000 training images and 10,000 test images.
- A **CNN architecture** is used for classification, including convolutional layers, pooling layers, and fully connected layers.

## Folder Structure
```
write2digit/
├── static/              # Frontend assets (JS, CSS, images)
├── templates/           # HTML files (index.html)
├── model.h5               # Trained model files
├── app.py               # Flask application
├── requirements.txt     # Dependencies
├── README.md            # Project documentation
```

## Future Enhancements
- Improve UI/UX with better drawing tools.
- Add model retraining capabilities from user data.
- Deploy to a cloud platform.
- Extend support for multiple handwriting styles.

## Contributing
Feel free to contribute by opening an issue or submitting a pull request.

Happy Coding! 🚀
