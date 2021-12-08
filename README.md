# Classifying Whether or Not a Employee Promoted for Job Level

## Foreground

A growing e-commerce company that has 9 departments throughout its company organization, has 78,298 employees divided into 3 levels, namely Senior, Middle and Junior. One of the problems faced in the company is that the company does not have specific requirements and certain patterns to identify which employees are suitable for promotion at the right time. So the company is in need of help to identify the factors in determining the promotion of an employee by considering the company's promotion cycle.

## The explanation of every variables:

- employee_id : Unique ID for Employees.
- department : Department where Employees (categorical).
- region : Region of employment (categorical).
- education : Level of education of the Employees (categorical).
- gender : Gender of Employees (categorical).
- job_level : Job Level in current position (categorical).
- recruitment_channel : Employee recruitment sources (categorical).
- no_of_trainings : Number of trainings attended (numeric).
- age : Employee Age (numeric).
- previous_year_rating : Previous year's employee rating (numeric).
- length_of_service : Employee working time (numeric).
- awards_won? : Employee label have received an award or not (categorical).
- avg_training_score : Current training average (numeric).
- satisfaction_score : Value of employee satisfaction with the company (numeric).
- engagement_score : Average value of employees feeling attached to the Company (numeric).
- is_promoted : Recommended for promotion (categorical).

## Import the python module/library to be use and create variable for load dataset.

```sh
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

dataset=pd.read_csv('https://docs.google.com/spreadsheets/d/e/2PACX-1vQlM2tat2DbRFUyldK-ZBFE2BIUdEaQpnXShc6uRvzlOes2Sevze-KEPpFd3-I4aB-lp6G5eVrHzUXK/pub?gid=466965807&single=true&output=csv')
```


## Data Checking & Pre-Processing

Count data from each column with null values.
```sh
dataset.isnull().sum()
```

The creation of the 'cleaned_dataset' variable which contains the dataset has been cleaned by filling in the data in the column containing the null value and deleting the unused column.
```sh
cleaned_dataset=dataset.copy()
cleaned_dataset['education']=cleaned_dataset['education'].fillna('No Data')
cleaned_dataset['previous_year_rating']=cleaned_dataset['previous_year_rating'].fillna(cleaned_dataset['previous_year_rating'].median())
cleaned_dataset=cleaned_dataset.drop(['employee_id'], axis=1)
cleaned_dataset=cleaned_dataset.rename(columns={'is_promoted':'label'})
cleaned_dataset.isnull().sum()
```

## Exploratory Data Analysis (EDA)
Visualizing Percentage is promoted.
```sh
fig = plt.figure()
ax = fig.add_axes([0,0,1,1])
ax.axis('equal')
labels = ['No', 'Yes']
label = cleaned_dataset.label.value_counts()
ax.pie(label, labels=labels, autopct='%.0f%%', colors=('#ED676F', '#1FAFD2'))
plt.title('Percentage of employees who have been promoted')
plt.show()
```

Exploratory Data Analysis (EDA) Numerical Variables.
```sh
numerical_features = ['no_of_trainings', 'age', 'previous_year_rating', 'length_of_service', 'avg_training_score', 'satisfaction_score', 'engagement_score']
fig, ax = plt.subplots(1, 7, figsize=(30, 6))
numerical_features, use a color of blue and red, respectively
cleaned_dataset[cleaned_dataset.label == 0][numerical_features].hist(bins=20, color='#ED676F', alpha=0.5, ax=ax)
cleaned_dataset[cleaned_dataset.label == 1][numerical_features].hist(bins=20, color='#1FAFD2', alpha=0.5, ax=ax)
plt.show()
```

Exploratory Data Analysis (EDA) Categorical Variables.
```sh
fig, ax = plt.subplots(1, 1, figsize=(12, 6))
sns.countplot(data=cleaned_dataset,x='one of the categorical variable names',hue='label',palette=('#ED676F', '#1FAFD2'))
plt.title('Total Promoted per Departement')
legend=plt.legend(bbox_to_anchor=(1,1), shadow=True, title='Promoted')
legend.get_texts()[0].set_text('No')
legend.get_texts()[1].set_text('Yes')
plt.tight_layout()
plt.show()
```

## Bulding Machine Learning Model
Data preparation by encoding data.
```sh
for column in cleaned_dataset.columns:
    if cleaned_dataset[column].dtype == np.number: continue
    cleaned_dataset[column] = LabelEncoder().fit_transform(cleaned_dataset[column])
```
Splitting dataset for train and test data.
```sh
X = cleaned_dataset.drop('label', axis=1)
y = cleaned_dataset['label']
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
```

SMOTE oversampling train data
```sh
from imblearn.over_sampling import SMOTE
smote=SMOTE()
x_train, y_train=smote.fit_resample(x_train, y_train)
```
Print according to the expected result.
```sh
print('The number of rows and columns of x_train is:', x_train.shape, ', while the Number of rows and columns of y_train is:', y_train.shape)
print('The percentage of Promoted in the Training data is:')
print(y_train.value_counts(normalize=True))
print('The number of rows and columns of x_test is:', x_test.shape, ', while the Number of rows and columns of y_test is:', y_test.shape)
print('The percentage of Promoted in the Testing data is:')
print(y_test.value_counts(normalize=True))
```

Modelling using Logistic Regression algorithm.
```sh
log_model=LogisticRegression().fit(x_train, y_train)
```

Modelling using Random Forest Classifier algorithm.
```sh
rdf_model=RandomForestClassifier().fit(x_train, y_train)
```

Modelling using Gradient Boosting Classifier algorithm.
```sh
gbt_model=GradientBoostingClassifier().fit(x_train, y_train)
```

Testing the performance of the model based on classification report.

```sh
y_train_pred = '#name_of_variable_model'.predict(x_train)
print('Classification Report #type_of_data_used (#name_of_algorithm_used):')
print(classification_report(y_train, y_train_pred))
print(accuracy_score(y_train, y_train_pred))
```

Displaying plots of heatmap confusion matrix.

```sh
confusion_matrix_df = pd.DataFrame((confusion_matrix(y_train, y_train_pred)), ('No promoted', 'Promoted'), ('No promoted', 'Promoted'))

plt.figure()
heatmap = sns.heatmap(confusion_matrix_df, annot=True, annot_kws={'size': 14}, fmt='d', cmap='YlGnBu')
heatmap.yaxis.set_ticklabels(heatmap.yaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)
heatmap.xaxis.set_ticklabels(heatmap.xaxis.get_ticklabels(), rotation=0, ha='right', fontsize=14)

plt.title('Confusion Matrix for #type_of_data_used\n(#name_of_algorithm_used)', fontsize=18, color='darkblue')
plt.ylabel('True label', fontsize=14)
plt.xlabel('Predicted label', fontsize=14)
plt.show()
```

Displaying plots of predictors/feature importance.
```sh
importance = '#name_of_variable_model'.feature_importances_
fig, ax = plt.subplots(figsize=(12, 6))
ax.barh(X.columns, importance, align='center')
ax.invert_yaxis()  
plt.show()
```


