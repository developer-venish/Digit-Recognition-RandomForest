# Digit-Recognition-RandomForest
ML Python Project
---------------------------------------------------------------------------------------

# Demo
![](https://github.com/developer-venish/Digit-Recognition-RandomForest/blob/main/download.png)

---------------------------------------------------------------------------------------

Note :- All the code in this project has been tested and run successfully in Google Colab. I encourage you to try running it in Colab for the best experience and to ensure smooth execution. Happy coding!

---------------------------------------------------------------------------------------

The given code performs the following steps:

1. Installs the pandas library using `!pip install pandas`.
2. Imports the required libraries: pandas, numpy, and the drive module from google.colab.
3. Mounts the Google Drive using `drive.mount('/content/gdrive')` to access files from Google Drive.
4. Sets the `filename` variable to the path of the dataset file in Google Drive.
5. Reads the dataset file using `pd.read_csv()` and stores it in the `dataset` variable.
6. Prints the shape of the dataset using `dataset.shape` to display the number of rows and columns.
7. Prints the first 5 rows of the dataset using `dataset.head(5)` to display a sample of the data.
8. Extracts the features (input variables) by selecting all columns except the first one using `dataset.iloc[:,1:]` and stores them in the `X` variable.
9. Prints the `X` variable to display the extracted features.
10. Prints the shape of `X` using `X.shape` to show the dimensions of the feature matrix.
11. Extracts the target variable by selecting only the first column using `dataset.iloc[:,0]` and stores it in the `Y` variable.
12. Prints the `Y` variable to display the extracted target variable.
13. Prints the shape of `Y` using `Y.shape` to show the dimensions of the target variable.
14. Splits the dataset into training and testing sets using `train_test_split()` from `sklearn.model_selection`. The training set is assigned to `X_train` and `y_train`, while the testing set is assigned to `X_test` and `y_test`. The testing set size is set to 25% of the whole dataset.
15. Creates a Random Forest classifier model using `RandomForestClassifier()` from `sklearn.ensemble`.
16. Trains the model on the training data using `model.fit(X_train, y_train)`.
17. Predicts the target variable for the testing data using `model.predict(X_test)` and assigns it to `y_pred`.
18. Calculates the accuracy of the model using `accuracy_score()` from `sklearn.metrics` by comparing the predicted values with the actual values (`y_test`). The result is printed as a percentage.
19. Imports the `matplotlib.pyplot` library as `plt`.
20. Sets the `index` variable to 10.
21. Prints the predicted value at the specified index using `model.predict(X_test)[index]`.
22. Disables the axis using `plt.axis('off')`.
23. Displays the image corresponding to the specified index by reshaping the values from `X_test.iloc[index].values` into a 28x28 matrix and using `plt.imshow()` to show the image in grayscale.
24. The code execution completes.

Note: Some parts of the code may require additional setup or dependencies, such as mounting Google Drive or having the required dataset file in the specified location.

---------------------------------------------------------------------------------------

Random Forest is a machine learning algorithm that combines multiple decision trees to create a more accurate and robust predictive model. It is a supervised learning algorithm used for both classification and regression tasks.

In a Random Forest model, a collection of decision trees is built, where each tree is trained on a different subset of the training data. The trees are constructed using a random selection of features at each split, hence the term "random" in Random Forest. This randomness helps to reduce overfitting and increases the diversity among the individual trees.

During prediction, the Random Forest combines the predictions of all the individual trees and outputs the most frequent prediction for classification tasks or the average prediction for regression tasks. This ensemble approach of aggregating the predictions of multiple trees improves the overall accuracy and generalization of the model.

Random Forest has several advantages:

1. It handles both numerical and categorical features without requiring extensive data preprocessing.
2. It can handle large datasets with a high number of features.
3. It provides an estimate of feature importance, which helps in understanding the relevance of each feature.
4. It is less prone to overfitting compared to individual decision trees.
5. It can handle missing values and outliers effectively.

Random Forest is widely used in various domains, including finance, healthcare, marketing, and image classification, due to its robustness and ability to handle complex datasets.

---------------------------------------------------------------------------------------
