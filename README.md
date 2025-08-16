**❤️ Heart Disease Prediction Web Application**
This project is a web application built with Streamlit that predicts the likelihood of a patient having heart disease based on their clinical data. The prediction is powered by a Logistic Regression model trained on the "Heart Failure Prediction" dataset.

A live demonstration of the application in action.

📋 Overview
The primary goal of this project is to provide a user-friendly interface for predicting heart disease risk. It leverages a machine learning model trained on a comprehensive dataset to provide real-time predictions with an associated confidence score. After experimenting with several models, Logistic Regression was chosen for its superior performance, achieving an accuracy of 86.47% on the test set.

✨ Features
User-Friendly Interface: Simple and intuitive web interface built with Streamlit.

Real-Time Predictions: Enter patient data and get an instant prediction.

Confidence Score: Displays the model's confidence in its prediction as a percentage.

Data Preprocessing: The backend handles all necessary encoding and scaling to match the training process.

📂 Project Structure
.
├── Logistic_Regression_heart.pkl   # The trained logistic regression model file
├── scaler.pkl                        # The fitted StandardScaler file
├── app.py                            # The main Streamlit application script
├── heartdisease.ipynb                # Jupyter Notebook with the model training process
├── requirements.txt                  # Required Python libraries
└── README.md                         # This README file

🛠️ Technologies Used
Python: The core programming language.

Pandas: For data manipulation and creating the input DataFrame.

Scikit-learn: For building and training the machine learning model.

Streamlit: For creating and deploying the web application.

Jupyter Notebook: For the initial data analysis, model training, and evaluation.

⚙️ Setup and Installation
To run this application on your local machine, please follow these steps:

1. Clone the Repository:

git clone [https://github.com/KHarish66/Heart-Disease-Prediction-Application.git](https://github.com/KHarish66/Heart-Disease-Prediction-Application.git)
cd Heart-Disease-Prediction-Application

2. Create a Virtual Environment (Recommended):

python -m venv venv
source venv/bin/activate  # On Windows, use `venv\Scripts\activate`

3. Install Dependencies:

pip install -r requirements.txt

4. Run the Streamlit Application:

streamlit run app.py

The application should now be running and accessible in your web browser at http://localhost:8501.

🚀 Usage
Open the web application in your browser.

Enter the patient's clinical information into the input fields.

Click the "Predict Heart Disease" button.

The prediction result and the model's confidence score will be displayed.

🧠 Model Training
The model was trained in the heartdisease.ipynb Jupyter Notebook. The key steps included:

Data Loading and Exploration: The dataset was loaded and analyzed for missing values and data types.

Data Cleaning: '0' values in Cholesterol and RestingBP, which are physiologically improbable, were identified and replaced with the mean of the respective columns.

Feature Engineering: Categorical features were converted into a numerical format using one-hot encoding (pd.get_dummies).

Feature Scaling: Numerical features were scaled using StandardScaler to normalize their range.

Model Training and Evaluation: A comparative analysis was performed between several classification models. Logistic Regression was ultimately selected for its superior performance, achieving the highest accuracy (86.47%) and F1-score (88.05%).

Saving Artifacts: The final trained model and the scaler were saved as .pkl files for use in the Streamlit application.

🤝 Contributing
Contributions are welcome! If you have suggestions for improvements, please feel free to create a pull request or open an issue.

📄 License
This project is licensed under the MIT License.
