
# Importing Necessary Libraries
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
# %matplotlib inline

train_df = pd.read_csv("/content/Train_data.csv")
test_df = pd.read_csv("/content/Test_data.csv")

#Display more rows and columns
pd.set_option('display.max_rows', None)  #Show all rows
pd.set_option('display.max_columns', None)  #Show all columns

#Display the first 5 rows of the DataFrame 'train_df'
train_df.head()

#Display the first 5 rows of the DataFrame 'test_df'
test_df.head()

train_df.info()

test_df.info()

#Check of null values in training set
train_df.isnull().sum()

#Check of null values in test set
test_df.isnull().sum()

#Check if there are any duplicate rows in train_df
train_df.duplicated().values.any()

#Generate summary statistics for numerical columns in train_df
train_df.describe()

#Select columns with categorical data , columns with object data type
cat_features = train_df.select_dtypes(include='object')

#Display column categorical features 'cat_features'
cat_features.columns

#Visualizing the Target Variable Class
sns.countplot(data = train_df, x = 'class')

#Count number of unique value in the 'class' column in 'train_df'
train_df['class'].value_counts()

#Plot to visualize distribution of 'protocol_type' for 'class'
sns.countplot(data=train_df, x='protocol_type', hue='class')

plt.figure(figsize = (12,6))
#Plot to visualize distribution of 'service' for 'class'
sns.countplot(data = train_df, x = 'service', hue = 'class')

#Plot to visualize distribution of 'flag' for 'class'
sns.countplot(data = train_df, x = 'flag', hue = 'class')

#Map the 'class' column to numerical , 0 for 'normal' , 1 for 'anomaly'
labels = {'normal': 0, 'anomaly': 1}

#Create a new 'labels' column with the mapped values , habdlo fel dataframe bta3ty y3ny
train_df['labels'] = train_df['class'].map(labels)

#To ensure the mapping done and in my dataframe display 0 & 1
train_df[['class','labels']].head()

#Select columns with numerical data types
num_features = train_df.select_dtypes(include = ['int64','float64'])
#Display column names of selected numerical features
num_features.columns

#Calculate correlation of all numerical features with the 'labels' column, and sort the values
num_features.corr()['labels'].sort_values()
#Correlation measures the strength and direction of the relationship between two variables.

#Positive correlation: Both variables increase together
#Negative correlation: One variable increases while the other decreases
#No correlation: No relationship between the variables

#I do drop correlation of column is always 1, useful for analysis m4 h3rf analyz w hykom sa3b any arsmo f 3ayza ykon less than 1
#Plot sorted correlation values as a bar chart
num_features.corr()['labels'].drop('labels').sort_values().plot(kind ='bar')

plt.figure(figsize=(12, 6))
#Create a heatmap to visualize correlation matrix
sns.heatmap(data=num_features.corr(), cmap='viridis')

#Calculating the correlation matrix
correlation_matrix = num_features.corr()

#threshold for strong correlation
threshold = 0.90

#Initialize empty array to store pairs of strongly correlated features
strong_correlations = []
#Iterate over correlation matrix to find strongly correlated feature pairs
for i in range(len(correlation_matrix.columns)): #Loop rows
    for j in range(i): #Loop columns
    # Check if the absolute value of correlation is greater than the threshold > 0.90
        if abs(correlation_matrix.iloc[i, j]) > threshold:
            strong_correlations.append((correlation_matrix.columns[i], correlation_matrix.columns[j], correlation_matrix.iloc[i, j]))

#Convert strongly correlated for better readability
strong_correlations_df = pd.DataFrame(strong_correlations, columns=['Feature 1', 'Feature 2', 'Correlation Coefficient'])
#Display the DataFrame with strongly correlated features
strong_correlations_df

from matplotlib import pyplot as plt
strong_correlations_df['Correlation Coefficient'].plot(kind='line', figsize=(8, 4), title='Correlation Coefficient')
plt.gca().spines[['top', 'right']].set_visible(False)

strong_correlations_list = list(strong_correlations_df['Feature 1'])
strong_correlations_list

#shows the number of rows and columns 'train_df'
train_df.shape

#Drop 'strong_correlations_list' from the 'train_df'
#The 'axis=1' we are dropping columns, not rows
train_df = train_df.drop(strong_correlations_list, axis = 1)
#axis=1: we are dropping columns , axis=0, it would drop rows.
#removes strong_correlations_list 3l4an ht3ml reduce the accuracy of the model w hyba2 bais

#Get the number of unique values
train_df['service'].nunique()

#column has alot of unique values fh7awl a4el ally malho4 lazma w w a5ly al 3add ya2l 4oya

#prevent multicollinearity
#Drop the 'service' column ,axis=1 dropping a column
train_df = train_df.drop('service', axis = 1)
#The column may not be useful for the model.

#To ensure col is dropped
train_df.shape

test_df = test_df.drop(strong_correlations_list, axis = 1)
test_df = test_df.drop('service', axis = 1)

test_df.shape

#Ensure columns in 'train_df' that are missing in 'test_df' wala kolha mogoda 3ady

# '~' condition to find the columns that are not in 'test_df' w by5ly al true false w al 3aks
missing_columns = train_df.columns[~train_df.columns.isin(test_df.columns)]

# Convert the missing columns to a list for better readability
missing_columns_list = missing_columns.tolist()

# Print the list of missing columns
print(missing_columns_list)

# h3ml ll col kol drop mn 'train_df'
train_df = train_df.drop('class', axis = 1)
cat_features.columns #cols al object

#Converting categorical data into numerical, binary columns to make data usable for training
train_dummies = pd.get_dummies(train_df[['protocol_type','flag']], drop_first=True )
train_df.drop(['protocol_type','flag'], axis =1, inplace = True) # Avoiding multicollinearity: By using drop_first=True
#Concatenate the dummy variables
train_df = pd.concat([train_df, train_dummies], axis =1)

train_df.shape

#same steps of training
test_dummies = pd.get_dummies(test_df[['protocol_type','flag']], drop_first=True )
test_df.drop(['protocol_type','flag'], axis =1, inplace = True)

test_df = pd.concat([test_df, test_dummies], axis = 1)

# Baseline Model
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import classification_report, confusion_matrix

log_model = LogisticRegression()
scaler = MinMaxScaler()

X = train_df.drop('labels', axis = 1)
y = train_df['labels']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=100)
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Fit logistic regression model on the training data
log_model.fit(X_train, y_train)  # trains on X_train features and y_train labels

# Predict  labels
predictions = log_model.predict(X_test)  # predictions for X_test using the trained model

# This prints a report with precision, recall, f1-score, and support for each class
print(classification_report(y_test, predictions))

import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Calculate the confusion matrix
cm = confusion_matrix(y_test, predictions)

# Plot the confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", xticklabels=["Predicted Negative", "Predicted Positive"], yticklabels=["Actual Negative", "Actual Positive"])
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.title('Confusion Matrix')
plt.show()

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense,Dropout
from tensorflow.keras.callbacks import EarlyStopping

#Initialize the Sequential model
ann_model = Sequential()
#Add the first Dense layer with 64 units and ReLU activation function
#input shape bt7km number of features in the training data (X_train.shape[1])
ann_model.add(Dense(64, activation = 'relu',input_shape=(X_train.shape[1], ))) # 1 3l4an bya5d al col ,input shape byt3aml m3 tuple

#Add a Dropout layer, to prevent overfitting
#This means 50% of the neurons in this layer will be randomly ignored during each training iteration
ann_model.add(Dropout(0.5))

#Add another Dense layer with 32 units and ReLU activation function
ann_model.add(Dense(32, activation = 'relu')) #ReLU (Rectified Linear Unit) activation function to introduce non-linearity

#Add another Dropout layer with 50% dropout rate for the second Dense layer
ann_model.add(Dropout(0.5))

#Add the final Dense layer with 1 unit for binary classification and a sigmoid activation function
#Sigmoid will output a value between 0 and 1
ann_model.add(Dense(1, activation = 'sigmoid')) # if the output > 0.5, classify as class 1; otherwise, class 0

#Compile the model by specifying the loss function, optimizer, and evaluation metric
#'binary_crossentropy' is commonly used for binary classification problems
#'adam' is an optimizer that adapts the learning rate during training
ann_model.compile(loss = 'binary_crossentropy', optimizer ='adam')

# Early stopping callback to prevent overfitting and stop training when the validation loss stops improving
early_stop = EarlyStopping(monitor='val_loss', mode='min', verbose=1, patience=3)

# Train the neural network model with early stopping
ann_model.fit(x=X_train, y=y_train, validation_data=(X_test, y_test), epochs=50, callbacks=[early_stop])

#visualize the loss over epochs
loss = pd.DataFrame(ann_model.history.history)

#Plot the loss values training and validation loss over epochs
loss.plot()
#By looking at the plot of loss and val_loss, we see no signs of overfitting hence, our model is actually performing well.

# Predict the probabilities for the test data
y_pred_proba = ann_model.predict(X_test)

# probabilities to class labels based on a threshold of 0.8
#threshold has been changed to 0.8, model must be "more confident" before classifying
y_pred = (y_pred_proba > 0.8).astype(int)

# This prints a report with precision, recall, f1-score, and support for each class
print(classification_report(y_test, y_pred))

#evaluation metrics and results for both models ANN and Logistic Regression
metrics = ['Precision (Class 0)', 'Precision (Class 1)', 'Recall (Class 0)', 'Recall (Class 1)', 'F1-Score (Class 0)', 'F1-Score (Class 1)', 'Accuracy']
ann_result = [0.97, 1.00, 1.00, 0.97, 0.99, 0.98, 0.98]  # ANN model scores
log_result = [0.95, 0.96, 0.97, 0.94, 0.96, 0.95, 0.96]  # Logistic Regression model scores

bar_width = 0.35
index = np.arange(len(metrics))

fig, ax = plt.subplots(figsize =(12,6))

#Plot for ANN model
bars1 = ax.bar(index, ann_result, bar_width, label='ANN', color='b')

#Plot for Logistic Regression model
bars2 = ax.bar(index + bar_width, log_result, bar_width, label='Log', color='g')

ax.set_xlabel('Metrics')
ax.set_ylabel('Scores')
ax.set_title('Comparison of Model Results')

ax.set_xticks(index + bar_width / 2)
ax.set_xticklabels(metrics, rotation=90)

#Display differentiate between ANN and Logistic Regression
ax.legend()

#visualization and prevent overlapping bars
plt.ylim(0, 1.2)
plt.tight_layout()

X_train = train_df.drop('labels', axis = 1).values
y_train = train_df['labels'].values

X_test = test_df.values
X_train =scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = Sequential()
model.add(Dense(64, activation='relu', input_shape=(X_train.shape[1],)))  # Add input shape
model.add(Dropout(0.5))
model.add(Dense(32, activation='relu'))
model.add(Dropout(0.5))

model.add(Dense(1, activation='sigmoid'))  # For binary classification

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=50, batch_size=32, validation_split=0.2, callbacks = [early_stop])

test_proba = ann_model.predict(X_test)
test_predictions = (test_proba > 0.8).astype(int)

test_predictions

predictions_df = pd.DataFrame(test_predictions, columns = ['Predicted_Labels'])
predictions_df.to_csv('predictions.csv', index = False)
predictions_df.head()

model.summary()