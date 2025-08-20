🍎 Smart Sorting: A Fresh vs. Rotten Classifier
Project Overview
Welcome to Smart Sorting, an intelligent web application designed to help reduce food waste by automatically classifying fruits and vegetables as "Fresh" or "Rotten" based on an image.

This project is an end-to-end solution that demonstrates skills in:

Machine Learning: Training and deploying a deep learning model.

Computer Vision: Using a model to analyze and classify image data.

Full-Stack Development: Building a functional web application with a backend API and a responsive frontend.

Key Features
Image Upload: Users can easily upload an image of a fruit or vegetable.

Live Preview: The application provides a live preview of the uploaded image before prediction.

Real-time Prediction: The backend API processes the image and returns a classification result in real time.

Confidence Score: The system provides a confidence score for its prediction, indicating the certainty of the result.

User-Friendly Interface: A clean and intuitive front-end built with modern web technologies.

Technologies Used
Backend: Python with the Flask framework for the web server and API.

Machine Learning: TensorFlow and Keras for building and training the deep learning model. The model utilizes transfer learning with a pre-trained VGG16 architecture.

Frontend: HTML for structure, JavaScript for interactivity, and Tailwind CSS for a responsive and modern design.

IBM Certification: This project was developed as a part of the "IBM Machine Learning Certification," highlighting a structured approach to model development.

How to Run the Project
Follow these steps to get a copy of the project up and running on your local machine.

Prerequisites
You need to have Python installed on your system. You can install the required libraries using pip:

pip install -r requirements.txt

Installation
Clone the repository:

git clone https://github.com/manudoddikusuma/SmartSorting-project-ml.git
cd your-repository-name

Download the Model:

The pre-trained model (model_transfer_learning.h5) is required to run the application. (Note: Please download the model file and place it in the project's root directory.)

Run the Flask Application:

python app.py

The application will now be running at http://127.0.0.1:5000/.
Author
Manudoddi Kusuma - https://github.com/manudoddikusuma



