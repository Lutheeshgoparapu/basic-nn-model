# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY



"Design and implement a neural network regression model to accurately predict a continuous target variable based on a set of input features within the provided dataset. The objective is to develop a robust and reliable predictive model that can capture complex relationships in the data, ultimately yielding accurate and precise predictions of the target variable. The model should be trained, validated, and tested to ensure its generalization capabilities on unseen data, with an emphasis on optimizing performance metrics such as mean squared error or mean absolute error. This regression model aims to provide valuable insights into the underlying patterns and trends within the dataset, facilitating enhanced decision-making and understanding of the target variable's behavior."

## Neural Network Model

![image](https://github.com/Lutheeshgoparapu/basic-nn-model/assets/94154531/0b22e299-c219-481d-a794-2dccbc9d32a0)


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: G.Lutheesh
### Register Number: 212221230029
```python

from google.colab import auth
import gspread
from google.auth import default
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense

auth.authenticate_user()
creds,_ = default()
gc = gspread.authorize(creds)
worksheet = gc.open('exp 1').sheet1
rows = worksheet.get_all_values()

df = pd.DataFrame(rows[1:], columns=rows[0])
df = df.astype({'INPUT':'float'})
df = df.astype({'OUTPUT':'float'})

X = df[['INPUT']].values
y = df[['OUTPUT']].values

X_train,X_test,y_train,y_test = train_test_split(X,y,test_size = 0.33,random_state = 33)
Scaler = MinMaxScaler()
Scaler.fit(X_train)
X_train1 = Scaler.transform(X_train)

AI_Brain = Sequential([
    Dense(units = 6, activation = 'relu', input_shape=[1]),
    Dense(units = 5, activation = 'relu'),
    Dense(units = 1)
])

AI_Brain.compile(optimizer= 'rmsprop', loss="mse")
AI_Brain.fit(X_train1,y_train,epochs=5000)
AI_Brain.summary()


loss_df = pd.DataFrame(AI_Brain.history.history)

loss_df.plot()

X_test1 = Scaler.transform(X_test)

AI_Brain.evaluate(X_test1,y_test)

X_n1 = [[20]]
X_n1_1 = Scaler.transform(X_n1)
AI_Brain.predict(X_n1_1)




```
## Dataset Information
![image](https://github.com/Lutheeshgoparapu/basic-nn-model/assets/94154531/b4c59116-7d39-42c0-8da1-d150db71d341)

## OUTPUT

### Training Loss Vs Iteration Plot

![image](https://github.com/Lutheeshgoparapu/basic-nn-model/assets/94154531/c0c4caf1-3f1c-49e9-a06e-5e124f69214c)


### Test Data Root Mean Squared Error

![image](https://github.com/Lutheeshgoparapu/basic-nn-model/assets/94154531/a14ad08b-28f7-4ffb-bbf4-87410dcf9f8e)

### New Sample Data Prediction
![image](https://github.com/Lutheeshgoparapu/basic-nn-model/assets/94154531/2d473831-e60e-4f60-a7f9-358ff6ac14c9)


## RESULT

Include your result here
