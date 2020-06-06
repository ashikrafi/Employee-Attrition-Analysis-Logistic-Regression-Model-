import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from IPython.core.display import display
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

'exec(%matplotlib inline)'
sns.set()

data = pd.read_csv("Data/logistic_regression_data.csv", sep=",")
# # data study
# display(data.info())
# display(data.shape)
# display(data.describe())
# display(data.columns)

# # data Cleaning
# display(data.isnull().any())
# display(data.isnull().sum())

# display(len(data))
# display(len(data[data['Attrition'] == 'Yes']))
# display(len(data[data['Attrition'] == 'No']))
# display("percentage of yes Attrition is:", (len(data[data['Attrition'] == 'Yes']) / len(data)) * 100, "%")
# display("percentage of no Attrition is:", (len(data[data['Attrition'] == 'No']) / len(data)) * 100, "%")
#

data.fillna(0, inplace=True)
data.drop(['EmployeeCount', 'EmployeeID', 'StandardHours'], axis=1, inplace=True)

# # Data Visualization
corr_cols = data[['Age', 'Attrition', 'BusinessTravel', 'DistanceFromHome', 'Education',
                  'EducationField', 'Gender', 'JobLevel', 'JobRole', 'MaritalStatus', 'MonthlyIncome',
                  'NumCompaniesWorked',
                  'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears', 'TrainingTimesLastYear',
                  'YearsAtCompany',
                  'YearsSinceLastPromotion', 'YearsWithCurrManager']]

corr = corr_cols.corr()
plt.figure(figsize=(16, 10))
sns.heatmap(corr, annot=True)
# plt.show()

sns.countplot(x="Attrition", data=data)
# plt.show()

sns.countplot(x="Attrition", data=data, hue="Gender")
# plt.show()

sns.countplot(x="Attrition", data=data, hue="JobLevel")


# plt.show()

# function to create group of ages, this helps because we have 78 different values here
def Age(dataframe):
    dataframe.loc[dataframe['Age'] <= 30, 'Age'] = 1
    dataframe.loc[(dataframe['Age'] > 30) & (dataframe['Age'] <= 40), 'Age'] = 2
    dataframe.loc[(dataframe['Age'] > 40) & (dataframe['Age'] <= 50), 'Age'] = 3
    dataframe.loc[(dataframe['Age'] > 50) & (dataframe['Age'] <= 60), 'Age'] = 4
    return dataframe


Age(data)

sns.countplot(x="Attrition", data=data, hue="Age")
# plt.show()

# Convert all the Categorical data into numerical data
# display(data['BusinessTravel'].unique())
# display(data['EducationField'].unique())
# display(data['Gender'].unique())
# display(data['Department'].unique())
# display(data['JobRole'].unique())
# display(data['MaritalStatus'].unique())
# display(data['Over18'].unique())

labelEncoder_X = LabelEncoder()
data['BusinessTravel'] = labelEncoder_X.fit_transform(data['BusinessTravel'])
data['Department'] = labelEncoder_X.fit_transform(data['Department'])
data['EducationField'] = labelEncoder_X.fit_transform(data['EducationField'])
data['Gender'] = labelEncoder_X.fit_transform(data['Gender'])
data['JobRole'] = labelEncoder_X.fit_transform(data['JobRole'])
data['MaritalStatus'] = labelEncoder_X.fit_transform(data['MaritalStatus'])
data['Over18'] = labelEncoder_X.fit_transform(data['Over18'])

label_encoder_y = LabelEncoder()
data['Attrition'] = label_encoder_y.fit_transform(data['Attrition'])
# display(data.head())

corr_cols = data[
    ['Age', 'Attrition', 'BusinessTravel', 'DistanceFromHome', 'Education', 'EducationField', 'Gender', 'JobLevel',
     'JobRole',
     'MaritalStatus', 'MonthlyIncome', 'NumCompaniesWorked',
     'PercentSalaryHike', 'StockOptionLevel', 'TotalWorkingYears',
     'TrainingTimesLastYear', 'YearsAtCompany', 'YearsSinceLastPromotion',
     'YearsWithCurrManager']]

corr = corr_cols.corr()
plt.figure(figsize=(18, 10))
sns.heatmap(corr, annot=True)
# plt.show()

# Split data into training and Testing set:
# Choose dependent and independent var:¶
# here dependent var is Attrition and rest of the var are independent var.
y = data['Attrition']
x = data.drop('Attrition', axis=1)

X_train, X_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=42)
Scaler_X = StandardScaler()
X_train = Scaler_X.fit_transform(X_train)
X_test = Scaler_X.transform(X_test)

lr = LogisticRegression()
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)

display(accuracy_score(y_test, y_pred))
display(confusion_matrix(y_test, y_pred))
display(classification_report(y_test, y_pred))
