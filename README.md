# 🏠 Rent-Price-Prediction-using-Machine-Learning



📘 Overview

This project aims to predict rental prices based on property features using multiple regression models.
The goal was to evaluate various machine learning algorithms and identify the best-performing model for accurate rent prediction.

🧠 Models Implemented

The following regression algorithms were implemented and compared:

Linear Regression

Lasso Regression

Ridge Regression

ElasticNet Regression

K-Nearest Neighbors (KNN) Regression

Random Forest Regression

Decision Tree Regression

⚙️ Workflow

Data Preprocessing

Handled categorical variables using pd.get_dummies()

Scaled features using StandardScaler

Split dataset into training and testing sets

Model Training & Evaluation

Used K-Fold Cross Validation (k=5) for all models

Evaluated using R² and Mean Squared Error (MSE)

Model Tuning

Tuned RandomForestRegressor using different n_estimators values (50–500)

Selected the configuration with the best average R² and lowest MSE

📊 Model Performance Comparison
Model	R²	MSE
Linear Regression	0.414	765,815

Lasso Regression	0.481	693,229

Ridge Regression	0.457	717,482

ElasticNet Regression	0.485	686,536

KNN Regression	0.348	802,094

Random Forest Regression	0.606	541,277

Decision Tree Regression	0.454	696,680


🔧 Model Tuning Results (Random Forest)

n_estimators	R² Mean	R² Std	MSE Mean

50	0.606	0.175	541,277

100	0.612	0.175	534,691

200	0.615	0.176	531,715

300	0.615	0.176	531,436

400	0.615	0.177	531,565

500	0.616	0.177	530,465


⚡ Key Insights

Random Forest outperformed all other models with an R² of 0.616 and the lowest MSE.

Feature scaling improved model performance for Lasso, Ridge, and ElasticNet.

Using n_jobs = -1 parallelized model training for faster computation.

🧩 Technologies Used

Python

Pandas, NumPy — Data manipulation

Scikit-learn — Model building & evaluation

Matplotlib, Seaborn — Visualization

Jupyter Notebook — Experimentation


🧾 Results & Conclusion

The Random Forest Regressor achieved the best overall performance.

Future improvements can include feature engineering, hyperparameter optimization, and advanced ensemble techniques to push R² beyond 0.65+.

👨‍💻 Author

Jay Panchal
📧 Email - panchaljay2711@gmail.com

LinkedIn - https://www.linkedin.com/in/jay-panchal-396443176

💼 Machine Learning
