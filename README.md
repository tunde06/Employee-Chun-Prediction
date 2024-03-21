# Employee-Chun-Prediction
This project help detect employees at flight risk. The model build algorithms based on the attributes of employee who left voluntarily and extends to existing employees to identify those that are likely to leave. This project is a follow up to the Cost of Turnover project that shows the dollar implication of vountary terminations.

## Goal
The overall goal is to help priotize employees for intervention before retirement. Also, early detection of high performing employees and critical roles can help HR intervene with appropriate retention measures and thus abate productivity loss and save significant cost on hiring and training.

## Import Packages
***

```python
# Import required modules for creating a dataframe with random data
import pandas as pd
import numpy as np
import random
```

```python
# importing all neccessary libraries/ modules for data manipulation and visual representation
import matplotlib.pyplot as plt
import matplotlib as matplot
%matplotlib inline
import statsmodels.api as sm
import seaborn as sns
from sklearn.preprocessing import scale
from sklearn.model_selection import train_test_split
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, confusion_matrix
```

```python
# Set seed for productivty
np.random.seed(42)
```

## Creating A DataFrame with Random Numbers
***

```python
data = {
    'Leave': np.random.randint(0, 2, 10000), # 0 or 1 for binary data
    'Performance': np.random.randint(1, 6, 10000), # Random float between 1 and 6
    'Promotion': np.random.randint(0, 2, 10000), # 0 or 1 for binary data
    'Compensation': np.random.randint(50000, 250000, 10000), # Random Integer for compensation
    'Compensation_Range': np.random.randint(1, 6, 10000), # Random float for compensation range
    'Change': np.random.uniform(-1, 5, 10000), # Random float between -1 and 5 for percentage change in compensation
    'Service': np.random.randint(1, 20, 10000), # Random Integer for Years in Service
    'Tenure': np.random.randint(1, 6, 10000), # Random Integer for Tenure Range
    'Satisfaction': np.random.randint(1, 6, 10000), # Random Integer for Level of Satisfaction
}
df = pd.DataFrame(data)
```
```python
# Display the DataFrame
print(df.head(1000))
```

## Data Quality Check

```python
df.isnull().sum()
```
```python
# Observing the percentage of missing values in each column
df.isnull().sum()/df.shape[0]
```
```python
# Checking the format of the columns
df.info()
```
```python
## Data Exploration
df.describe()
```
```python
df.shape
```
```python
df.head(10)
```

## Data Preprocessing
***

### Outliers Treatment

```python
# Handling Outliers in 'Change' columns using Inter Quantile Range (IQR) Method
IQR = df.Change.quantile(0.75) - df.Change.quantile(0.25)
lower_bridge = df.Change.quantile(0.25) - (IQR*1.5)
upper_bridge = df.Change.quantile(0.75) + (IQR*1.5)
print (lower_bridge, upper_bridge)
```
```python
# Replacing Outliers in 'Change' column using Inter Quantile Range (IQR) Method
df.loc[df['Change']>= 8.035, 'Change'] = 8.035
df.loc[df['Change']<= -3.986, 'Change'] = -3.986
```
## Data Visualization
***

```python
# Target Variable; Leave
fig, ax= plt.subplots(1,1, figsize=(15,5))
```
```python
sns.distplot(df['Leave'], norm_hist=False, kde=True, color='blue')
ax.set_title('Employee at Flight Risk', fontsize=18)
ax.set_ylabel('Count', fontsize=16)
ax.set_xlabel('Leave', fontsize=16)
```
```python
# Checking the format of the columns
df.info()
```
```python
#Set up the metplotlib figure
f, axes = plt.subplots(ncols=4, figsize=(15,6))
```
```python
# Graph Employee performance
sns.histplot(df.Performance, kde=False, color="r", ax=axes[0]).set_title('Performance Rating Distribution')
axes[0].set_ylabel('Employee Count')
```
```python
# Graph Employee promotion
sns.histplot(df.Promotion, kde=False, color="b", ax=axes[1]).set_title('Promotion within Two Years Distribution')
axes[1].set_ylabel('Employee Count')
```
```python
# Graph Employee tenure
sns.histplot(df.Tenure, kde=False, color="g", ax=axes[2]).set_title('Tenure Distribution')
axes[2].set_ylabel('Employee Count')
```
```python
# Graph Employee satisfaction
sns.histplot(df.Satisfaction, kde=False, color="y", ax=axes[3]).set_title('Satisfaction Level Distribution')
axes[3].set_ylabel('Employee Count')
```
```python
#Set up the metplotlib figure
f, axes = plt.subplots(ncols=4, figsize=(15,6))
```
```python
# Graph Employee compensation
sns.histplot(df.Compensation, kde=False, color="y", ax=axes[0]).set_title('Compensation Distribution')
axes[0].set_ylabel('Employee Count')
```
```python
# Graph Employee compensation range
sns.histplot(df.Compensation_Range, kde=False, color="y", ax=axes[1]).set_title('Compensation Range Distribution')
axes[1].set_ylabel('Employee Count')
```
```python
# Graph Employee Percentage Base Change
sns.histplot(df.Change, kde=False, color="y", ax=axes[2]).set_title('Percentage Base Change Distribution')
axes[2].set_ylabel('Employee Count')
```
```python
# Graph Employee Service
sns.histplot(df.Service, kde=False, color="r", ax=axes[3]).set_title('Years in Service Distribution')
axes[3].set_ylabel('Employee Count')
```

### Correlation Matrix and Heatmap
***

```python
df.corr()
```
```python
correlation = df.corr(method='pearson')
fig, ax = plt.subplots()
ax.figure.set_size_inches(20,20)
sns.heatmap(correlation, annot= True)
plt.show
```
```python
df.describe()
```

## Modeling
***

```python
# Creating x and y variable for modeling
x = df.drop('Leave', axis=1)
y = df['Leave']
```

```python
# Checking dimensions of 'x' and 'y'
x.shape, y.shape
```

### Scaling the features
***

```python
# Scaling the features to make interpretation of regression coefficient easier
from sklearn.preprocessing import scale
```
```python
# Storing column names in cols, since column names are lost after scaling (df is converted to a numpy array)
cols=x.columns
x_scaled=pd.DataFrame(scale(x))
x_scaled.columns=cols
x_scaled.columns
```

### Splitting Data into Training and Testing Sets

```python
# Splitting Data into Train and Test Set with a ratio of 80:20 for Modelling
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size=0.2, random_state=10)
```

### Random Forest Model
***
```python
# Initiating the Random Forest Model
rf_model = RandomForestClassifier()
```
```python
# Fitting/Training the Random Forest Model
rf_model.fit(x_train, y_train)
```
```python
# Predicting for Test Dataset using the Random Forest Model
y_pred_rf = rf_model.predict(x_test)
```
```python
# Checking the Accuracy, Precision, Recall and F1-score of the Model on Test Dataset
print('Accuracy of Random Forest Model on Test Set: {:.2f}'.format(accuracy_score(y_test, y_pred_rf)))
print('Precision of Random Forest Model on Test Set: {:.2f}'.format(precision_score(y_test, y_pred_rf)))
print('Recall of Random Forest Model on Test Set: {:.2f}'.format(recall_score(y_test, y_pred_rf)))
print('F1-score of Random Forest Model on Test Set: {:.2f}'.format(f1_score(y_test, y_pred_rf)))
```
```python
# Visualizing Random Forest Model for Understanding
fn=features = list(df.columns[1:])
cn=[str(x) for x in fn]

fig, axes = plt.subplots(nrows = 1,ncols = 1,figsize = (4,4), dpi=800)
tree.plot_tree(rf_model.estimators_[0],
               feature_names = fn, 
               class_names=cn,
               filled = False);
fig.savefig('Flight Risk')
```
```python
df.head(10)
```
```python
# The model predict what employee will voluntarily terminate
rf_model.predict([[3,0,87962,4,-0.761185,6,3,1]])
```
### Create New Dataframe To Predict
***

```python
# Create New DataFrame with Random Data
data = {
    'Leave': np.random.randint(0, 2, 1000), # 0 or 1 for binary data
    'Performance': np.random.randint(1, 6, 1000), # Random float between 1 and 6
    'Promotion': np.random.randint(0, 2, 1000), # 0 or 1 for binary data
    'Compensation': np.random.randint(50000, 250000, 1000), # Random Integer for compensation
    'Compensation_Range': np.random.randint(1, 6, 1000), # Random float for compensation range
    'Change': np.random.uniform(-1, 5, 1000), # Random float between -1 and 5 for percentage change in compensation
    'Service': np.random.randint(1, 20, 1000), # Random Integer for Years in Service
    'Tenure': np.random.randint(1, 6, 1000), # Random Integer for Tenure Range
    'Satisfaction': np.random.randint(1, 6, 1000), # Random Integer for Level of Satisfaction
}
new_df = pd.DataFrame(data)
```
```python
# Assuming New Features contains the Feature columns for the New data
# Replace New features with Actual Feature Columns for the New data
new_features = new_df[['Performance', 'Promotion', 'Compensation', 'Compensation_Range', 'Change', 'Service', 'Tenure', 'Satisfaction']]
```
```python
# Use the trained model to make predictions on the New Data
new_predictions = rf_model.predict(new_features)
```
```python
# Employee Flight Risk Score
print("Predictions for the new data:")
print(new_predictions)
```

  
