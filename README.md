# Project-House-Prices---Advanced-Regression-Techniques
Machine learning project to predict house prices using advanced regression techniques. Includes data preprocessing, model training, evaluation, and insights from feature importance analysis.
OBJECTIVE:

Introduction:
Real estate markets are complex ecosystems influenced by numerous factors ranging from economic trends to demographic shifts. Predicting house prices accurately is crucial for various stakeholders, including buyers, sellers, and real estate professionals. In this project, we aim to leverage advanced regression techniques to develop models that can predict house prices with high precision. By employing state-of-the-art methodologies, we seek to contribute to the optimization of pricing strategies and facilitate informed decision-making in the real estate domain.
Goal:

Our goal is to construct robust regression models that outperform traditional methods in predicting house prices. We aim to achieve superior generalization and provide valuable insights into housing market dynamics. We predict the sales price for each house. For each ID in the test set, we must predict the value of the SalePrice variable. 

Data Source:
The dataset is sourced from the Kaggle competition "House Prices: Advanced Regression Techniques." It includes comprehensive features describing residential properties and sale prices, serving as a rich resource for model development.

Importance:
Accurate prediction of house prices holds immense significance for multiple stakeholders within the real estate ecosystem. For prospective buyers, reliable price estimates enable informed decision-making, helping them assess affordability and negotiate favorable deals. Similarly, sellers can benefit from accurate pricing strategies to optimize their returns and minimize time on the market.
---------------------------------------------------------------------------------------------------------------------------
Approach to solving the problem:

Data Preprocessing:
1)	Handling Missing Values:
Here, we
a)	Replace missing values in numerical columns using the SimpleImputer object from the sklearn library.

b)	Filled missing values for certain columns with constants (e.g., 0 for   
              "BsmtFinSF1",  "MasVnrArea") and for others with the median.

2)	Categorical to numeral:
 
(I implemented the none_transform function which converts missing categorical values into specific strings from the none_conversion dictionary.)

 

3)	Feature Engineering:
As they are not contained in kaggle dataset I decided to create the following features  from other information:

1) "TotalSqrtFeet" - Total Live Area
2) "TotalBaths" - Total Area for Bathrooms
 

4)	Exploratory Data Analysis (EDA):

Here, we conducted EDA to understand data distribution, identified outliers, and checked for correlations. The prediction target is 'SalePrice'.
 
There are two outliers with prices more than 700000.
 
(here, we notice that it is right-skewed distribution with the pick around 160k and quite long tail with maximum about 800k.)

 

5)	Columns with NaN Values:
Here, we check the fraction of Nan values in each column and dropped columns with more than 90% NaN values.
 

6)	Removing outliers:

Here, we use two techniques: more and less rigorous for this data:

The first one was Z-score method. Z-scores are expressed in terms of standard deviations from their means. As a result, these z-scores have a distribution with a mean of 0 and a standard deviation of 1. 

I set threshold = 3 to identify outliers.
                                                               
 
Second, we use less rigorous method and make a plot for SalePrice and GrLivArea and removed those which seems to be outliers.
 
7)	GarageYrBlt Feature:

Here, we check if there are records that YearBuilt or GarageYrBlt have further year than 2017.
 

8)	LotFrontage feature:

LotFrontage is a linear feet of street connected to property. It is a high probability that these values are similar to houses in the same Neighborhood.Here, we imput missing values for "LotFrontage" based on the median of similar properties in the same neighborhood.
 
 

9)	Transformation of Numerical Variables:

Here, we convert some numerical variables (e.g., "MSSubClass", "OverallCond") that are actually categorical into string format.
 
Model Selection:

1)	Linear Regression:

Here,we use this Baseline model using a robust scaler for stability.
 
 
 
2)	LASSO Regression:

      This technique applies L1 regularization for feature selection and sparsity.
 
 
3)	Gradient Boosting Regressor (GBR):

Here, Gradient Boosting Regressor (GBR) is an ensemble technique that builds sequential trees, reducing bias and variance.
 
4)	XGBoost Regressor:

XGBoost Regressor is highly efficient implementation of gradient boosting with    regularization.
 
5)	ElasticNet:

ElasticNet combines L1 and L2 regularization to improve generalization and feature    selection.

 

6)	LightGBM:

LightGBM is a gradient boosting framework that is fast and efficient, particularly with large datasets.
 

7)	Bagging Regressor:

Bagging Regressor reduces variance by averaging predictions from multiple models trained on random subsets.
 
8)	Stacking:

Stacking leverages the strengths of multiple models by combining them to improve predictive accuracy.
 

Training Process:

1)	Splitting Data:
Divide the dataset into training and validation sets to evaluate model performance.
 

2)	Model Training:
In model training we train each model on the training set, applying hyperparameter tuning using techniques like grid search where applicable.We utilize GridSearchCV to perform an exhaustive search over specified parameter values for the estimator, optimizing hyperparameters during training.

3)	Cross-Validation:

Here we employ cross-validation (e.g., KFold) to ensure model generalization and prevent overfitting, using metrics like RMSE for evaluation.
example:

 
---------------------------------------------------------------------------------------------------------------------------
MODEL SUMMARY

Model Architecture:

Understanding the architecture helps in comprehending how the model captures patterns in the data and makes predictions.For this Project ,the types of models used, their components, and how they are interconnected :
•	 Linear Regression: Basic linear model with RobustScaler for preprocessing.
•	LASSO: Linear model with L1 regularization, optimized with LassoCV.
•	GradientBoostingRegressor: Ensemble model using boosting with decision trees.
•	XGBRegressor: Extreme Gradient Boosting model with tree-based learning.
•	ElasticNet: Linear model combining L1 and L2 regularization, optimized with ElasticNetCV.
•	LightGBM: Gradient boosting framework with tree-based learning.
•	Stacking Regressor: Ensemble model combining predictions from multiple models (Lasso, ElasticNet, XGBoost, LightGBM) using a meta-regressor.

Hyperparameters:

Proper selection and tuning of hyperparameters are essential for optimizing model performance and preventing overfitting or underfitting.For this Project,these are : 
•	LASSO: alphas: Controls the strength of L1 regularization, crucial for feature selection.
LASSO: alphas = [0.0004, 0.0005, 0.0006]
•	GradientBoostingRegressor: n_estimators, max_depth, learning_rate: Number of boosting stages, maximum depth of trees, and learning rate for shrinkage.
GradientBoostingRegressor: n_estimators=2500, max_depth=5, learning_rate=0.05
•	XGBRegressor: n_estimators, max_depth, learning_rate: Similar parameters to GradientBoostingRegressor, tailored for XGBoost's implementation.
XGBRegressor: n_estimators=2000, max_depth=3, learning_rate=0.05
•	ElasticNet: alphas, l1_ratio: Combination of L1 and L2 regularization strengths.
ElasticNet: alphas=[0.0001, 0.0003, 0.0004, 0.0006], l1_ratio=[0.9, 0.92]
•	LightGBM: num_leaves, learning_rate, n_estimators: Maximum number of leaves per tree, learning rate, and number of boosting iterations.
LightGBM: num_leaves=5, learning_rate=0.05, n_estimators=800

Training Performance:

Understanding training performance helps in assessing how well the model fits the training data and provides insights into potential areas for improvement.For this Project:

The metrics RMSE (Root Mean Squared Error) is used to evaluate model accuracy.
Performance Results:
•	Linear Regression: RMSE = 0.160
•	LASSO: RMSE = 0.135
•	GradientBoostingRegressor: RMSE = 0.118
•	XGBRegressor: RMSE = 0.119
•	ElasticNet: RMSE = 0.115
•	LightGBM: RMSE = 0.123
•	BaggingRegressor: RMSE = 0.172
---------------------------------------------------------------------------------------------------------------------------
RESULTS

Running Predictions:

•	This section demonstrates the process of making predictions using the trained models on the test dataset.
•	It includes code to generate predictions using each individual model (en_model, lasso_model, lgb_model) as well as the stacked model (stack_model).
•	A weighted average of the predictions from different models is calculated to generate the final predictions (stack_preds).
•	The predictions are then formatted into a DataFrame (predictions_df) with appropriate column names and index values.
 
 
Here,
Further we save the predictions to a CSV file (my_predictions.csv) for submission which gives us the sample_submission.csv .It includes code to export the predictions DataFrame to a CSV file with the required format for submission, including the appropriate headers and index labels like:
 
---------------------------------------------------------------------------------------------------------------------------
INFERENCES:

1)	Feature Importance:

Features like the size of the property, location (neighborhood), number of bedrooms and bathrooms, and other amenities such as the presence of a pool or garage tend to have a significant impact on house prices. This is inferred from the model's feature importance analysis, which highlights the most influential factors in predicting house prices.

2)	Model Interpretability:

The model makes predictions based on a combination of factors such as property size, location, and amenities. For example, it may assign higher prices to properties with larger square footage in desirable neighborhoods. Insights gained from interpreting the model include understanding how each feature contributes to the overall prediction and identifying patterns in the data that drive house prices.

3)	Limitations:

Limitations of the model may include potential biases in the dataset, such as underrepresentation of certain neighborhoods or property types. Additionally, the model may not account for external factors such as economic conditions or market trends, which could impact house prices but are not captured in the dataset. It's also important to acknowledge that the model's predictions are based on historical data and may not accurately reflect future market conditions.

4)	Future Directions:

Future research could focus on improving data quality by collecting more detailed information on property attributes and market trends. Additionally, exploring advanced modeling techniques such as deep learning or ensemble methods could help improve predictive accuracy. Incorporating external data sources such as economic indicators or demographic trends may also enhance the model's performance and provide more robust insights into housing market dynamics.
---------------------------------------------------------------------------------------------------------------------------

REFERENCES:

The references for this project include:

1)	Kaggle competition page for the "House Prices: Advanced Regression Techniques" challenge:
https://www.kaggle.com/c/house-prices-advanced-regression-techniques/overview
2)	Various articles and resources on feature engineering, one-hot encoding, and factors influencing home prices, sourced from:
•	"Discover Feature Engineering: How to Engineer Features and How to Get Good at It" by Machine Learning Mastery:
https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/
•	"Why One-Hot Encode Data in Machine Learning" by Machine Learning Mastery.
https://machinelearningmastery.com/why-one-hot-encode-data-in-machine-learning/
•	"6 Factors That Influence a Home's Value" by Inman.
https://www.inman.com/2017/08/07/6-factors-that-influence-a-homes-value/
•	"What Factors Influence the Sale Price of a Home" by Rochester Real Estate Blog.
https://www.rochesterrealestateblog.com/what-factors-influence-the-sale-price-of-a-home/
---------------------------------------------------------------------------------------------------------------------------
Other Comments:

•	Experimenting with different ensemble methods, such as stacking or blending, to combine the predictions of multiple models for improved performance.
•	Discussing the computational resources required to deploy the model, such as memory and processing power, and how to optimize them for efficiency.
•	Considering the need for real-time predictions and designing a system architecture that can handle high throughput and low latency requirements.
•	Acknowledging any collaborators or contributors who assisted with data collection, preprocessing, or model development.

Code link: 

https://jupyter.e.cloudxlab.com/user/mnpragnabi8929/notebooks/Project-House%20Prices%20-%20Advanced%20Regression%20Techniques/Project-House%20Prices%20-%20Advanced%20Regression%20Techniques/Project-House%20Prices%20-%20Advanced%20Regression%20Techniques.ipynb

Data Description:

https://jupyter.e.cloudxlab.com/user/mnpragnabi8929/edit/Project-House%20Prices%20-%20Advanced%20Regression%20Techniques/Project-House%20Prices%20-%20Advanced%20Regression%20Techniques/data_description.txt

Dataset:

https://jupyter.e.cloudxlab.com/user/mnpragnabi8929/edit/Project-House%20Prices%20-%20Advanced%20Regression%20Techniques/Project-House%20Prices%20-%20Advanced%20Regression%20Techniques/train.csv

https://jupyter.e.cloudxlab.com/user/mnpragnabi8929/edit/Project-House%20Prices%20-%20Advanced%20Regression%20Techniques/Project-House%20Prices%20-%20Advanced%20Regression%20Techniques/test.csv

Sample Submission:

https://jupyter.e.cloudxlab.com/user/mnpragnabi8929/edit/Project-House%20Prices%20-%20Advanced%20Regression%20Techniques/Project-House%20Prices%20-%20Advanced%20Regression%20Techniques/sample_submission.csv
