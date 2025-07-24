# Logistic Regression – Machine Learning Project

This project implements Logistic Regression from the scikit-learn library to classify categorical data. The pipeline includes data preprocessing, model training, hyperparameter tuning using GridSearchCV, and performance evaluation on unseen test data.

Dataset Overview

The dataset consists of multiple numerical features and one target variable with categorical classes. Exploratory Data Analysis (EDA) was conducted using Seaborn’s pairplot to visually inspect patterns and class separability among features such as sepal length, petal length, etc.

Preprocessing

Imported and loaded the dataset using pandas.

Checked for missing values, feature distribution, and data types.

Split the dataset into training and testing sets to evaluate generalization.

No scaling or encoding was necessary due to the already clean and structured nature of the dataset.

Model: Logistic Regression

Logistic Regression was selected for its efficiency in binary and multiclass classification problems.

Model was initialized using LogisticRegression() from sklearn.linear_model.

Hyperparameters tuned using GridSearchCV:

penalty: ['l1', 'l2', 'elasticnet']

C: [1, 2, 3, 4, 5, 6, 10, 20, 30, 40, 50]

max_iter: [100, 200, 300]

cv=5 was used for cross-validation to ensure robust performance estimation.

Scoring metric: accuracy.

Training

The model was trained using the .fit(x_train, y_train) method.

GridSearchCV selected the optimal set of hyperparameters based on accuracy across cross-validation folds.

Evaluation

Predictions on the test set were made using .predict(x_test).

Accuracy was computed via accuracy_score(y_test, y_pred) to assess real-world performance.

A classification_report was generated to view precision, recall, and F1-score for each class.

Visualization

Used seaborn.pairplot(df, hue="species") to visualize how features distribute across different classes.

This helped in confirming the dataset’s suitability for logistic regression by observing linear separability.

Key Takeaways

Logistic Regression is highly interpretable and effective when the features are linearly separable.

GridSearchCV played a critical role in selecting the best hyperparameters, leading to improved accuracy.

The final model demonstrated strong performance on test data, confirming its ability to generalize.

Tech Stack

Python 3.x

pandas

seaborn

scikit-learn

pandas


License and Citation

This project is intended for educational and research purposes. Attribution is appreciated. You may cite this work or adapt its structure for academic or commercial ML applications.

