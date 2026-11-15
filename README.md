📰 Fake News Detection Using Machine Learning

📌 About the Project

Fake News Detection is a Machine Learning-based mini project developed using Python. The main aim of this project is to identify whether a given news article is likely to be Fake News or Real News.

This project uses Natural Language Processing (NLP) techniques and Machine Learning algorithms to analyse news text and make predictions.

🎯 Project Objective

- To detect fake and real news using Machine Learning.
- To understand how NLP works with text data.
- To convert text into numerical values using TF-IDF.
- To build a simple web application for news prediction.

🛠️ Technologies Used

- Python
- Pandas
- Scikit-learn
- Natural Language Processing (NLP)
- TF-IDF Vectorizer
- Logistic Regression
- Streamlit
- VS Code

⚙️ How the Project Works

1. Dataset Collection: Load fake and real news datasets.
2. Data Preprocessing: Clean the news text by removing unnecessary characters and spaces.
3. TF-IDF Vectorization: Convert text into numerical features.
4. Dataset Splitting: Divide the dataset into 80% training data and 20% testing data.
5. Model Training: Train the model using Logistic Regression.
6. Model Testing: Evaluate the model using accuracy and classification metrics.
7. Prediction: Enter news text into the web application to get a prediction.

✨ Features

- Simple and user-friendly interface.
- Accepts news text as input.
- Predicts Fake News or Real News.
- Uses Machine Learning for prediction.
- Displays results through a Streamlit web application.

📂 Project Structure

Fake-News-Detection/
│
├── app.py
├── fake_news_model.plk
├── Fake.csv
├── news_datset_cleaned.csv
├── news_dataset.csv
├── README.md
├── requirments.txt
├── tfidf_vectorizer.pkl
├── train.py
└── True.csv

▶️ How to Run the Project

Step 1: Install Required Libraries

pip install -r requirements.txt

Step 2: Train the Model

python train.py

Step 3: Run the Web Application

streamlit run app.py

Step 4: Make Predictions

- Enter a news article in the text box.
- Click the Check News button.
- View the prediction displayed on the screen.

📊 Machine Learning Algorithm

Logistic Regression

Logistic Regression is a supervised Machine Learning algorithm used for classification. In this project, it is used to classify news articles into two categories:

- 0 – Fake News
- 1 – Real News

🔮 Future Enhancements

- Improve prediction accuracy.
- Add more news datasets.
- Support multiple languages.
- Improve the user interface.
- Explore more Machine Learning algorithms.

⚠️ Disclaimer

This project is developed for educational purposes. The predictions are based on patterns learned from the training dataset and should not be considered as verified facts.

👩‍💻 Project Type

BTech CSE-AIML Mini Project

Project Name: Fake News Detection Using Machine Learning