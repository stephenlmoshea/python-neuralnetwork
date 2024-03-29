# Neural network
Artificial neural network for Python. Features online backpropagtion learning using gradient descent, momentum, the sigmoid and hyperbolic tangent activation function.

## About
The library allows you to build and train multi-layer neural networks. You first define the structure for the network. The number of input, output, layers and hidden nodes. The network is then constructed. Interconnection strengths are represented using an adjacency matrix and initialised to small random values.  Traning data is then presented to the network incrementally. The neural network uses an online backpropagation training algorithm that uses gradient descent to descend the error curve to adjust interconnection strengths. The aim of the training algorithm is to adjust the interconnection strengths in order to reduce the global error. The global error for the network is calculated using the mean sqaured error. 

You can provide a learning rate and momentum parameter.  The learning rate will affect the speed at which the neural network converges to an optimal solution. The momentum parameter will help gradient descent to avoid converging to a non optimal solution on the error curve called local minima.  The correct size for the momentum parameter will help to find the global minima but too large a value will prevent the neural network from ever converging to a solution.

Trained neural networks can be saved to file and loaded back for later activation.

## Installation
```bash
$  pip install neuralnetwork
```

## Logging
### To log training to the file ./training.log create ./.env file with following parameters
```py
LOG_TRAINING=true
LOG_LEVEL=1
```
#### LOG_LEVEL=2 logs more details including network outputs and network error after each training example is presented.

## Examples
### Training XOR function on three layer neural network with two inputs and one output
```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

sigmoid = Sigmoid()

networkLayer = [2,2,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.3)

trainingSet = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

while True:
    backpropagation.initialise()
    result = backpropagation.train(trainingSet)

    if(result):
        break

feedForward.activate([0,0])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([0,1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,0])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,1])
outputs = feedForward.getOutputs()
print(outputs[0])
```

### Training XOR function on three layer neural network with two inputs and two output
```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

sigmoid = Sigmoid()

networkLayer = [2,3,2]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.3,0.002)

trainingSet = [
    [0,0,0,0],
    [0,1,0,1],
    [1,0,1,0],
    [1,1,0,0]
]

while True:
    backpropagation.initialise()
    result = backpropagation.train(trainingSet)

    if(result):
        break

feedForward.activate([0,0])
outputs = feedForward.getOutputs()
print(outputs)

feedForward.activate([0,1])
outputs = feedForward.getOutputs()
print(outputs)

feedForward.activate([1,0])
outputs = feedForward.getOutputs()
print(outputs)

feedForward.activate([1,1])
outputs = feedForward.getOutputs()
print(outputs)
```

### Training XOR function on three layer neural network using Hyperbolic Tangent
```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.HyperbolicTangent import HyperbolicTanget
from neuralnetwork.Backpropagation import Backpropagation

hyperbolicTangent = HyperbolicTangent()

networkLayer = [2,2,1]

feedForward = FeedForward(networkLayer, hyperbolicTangent)

backpropagation = Backpropagation(feedForward,0.7,0.3,0.001)

trainingSet = [
                    [-1,-1,-1],
                    [-1,1,1],
                    [1,-1,1],
                    [1,1,-1]
                ];

while True:
    backpropagation.initialise()
    result = backpropagation.train(trainingSet)

    if(result):
        break

feedForward.activate([-1,-1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([-1,1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,-1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,1])
outputs = feedForward.getOutputs()
print(outputs[0])
```

### Saving trained neural network to file
```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

sigmoid = Sigmoid()

networkLayer = [2,2,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.3)

trainingSet = [
    [0,0,0],
    [0,1,1],
    [1,0,1],
    [1,1,0]
]

while True:
    backpropagation.initialise()
    result = backpropagation.train(trainingSet)

    if(result):
        break

feedForward.save('./network.txt')
```

### Load trained neural network from file
```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

feedForward = FeedForward.load('./network.txt')

feedForward.activate([0,0])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([0,1])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,0])
outputs = feedForward.getOutputs()
print(outputs[0])

feedForward.activate([1,1])
outputs = feedForward.getOutputs()
print(outputs[0])
```

## Training Neural Network to Predict Diabetes
For this example we will train a neural network to predict whether a patient will develop diabetes within the next five years given various health measurements such as number of times pregnant, glucose plasma, blood pressure diastolic, skin fold thickness, serum insulin, body mass index, diabetes pedigree function, and age.

The dataset for this project is hosted by Kaggle. To download the necessary dataset for this example, please follow the instructions below.

1. Go to https://www.kaggle.com/uciml/pima-indians-diabetes-database

2. Click on the 'Download All' button

3. Kaggle will prompt you to sign in or to register. If you do not have a Kaggle account, you can register for one.

4. Upon signing in, the download will start automatically.

5. After the download is complete, unzip the zip file and move the file 'diabetes.csv' into your project folder.

For this example we first preprocess the data to ensure we have no missing or zero values.  We then scale or normalise the inputs before passing into the neural network.

We then split the data into a training and validation set which we use to test the accuracy of the trained neural network.

We then construct a neural network with 8 inputs, 32 nodes in the first hidden layer, 16 nodes in the second hidden layer, and finally one node in the output layer.

The neural network will output a 1 if the patient will develop diabetes and a 0 otherwise.

We then train the neural network using the training set.

Finally we test the trained neural network using the validation set and output and acurracy percetange.

```py
import matplotlib
matplotlib.use("TkAgg")
import pandas as pd
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
import numpy as np
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn import preprocessing
np.random.seed(16)

def preprocess(df):
    print('----------------------------------------------')
    print("Before preprocessing")
    print("Number of rows with 0 values for each variable")
    for col in df.columns:
        missing_rows = df.loc[df[col]==0].shape[0]
        print(col + ": " + str(missing_rows))
    print('----------------------------------------------')

    # Replace 0 values with the mean of the existing values
    df['Glucose'] = df['Glucose'].replace(0, np.nan)
    df['BloodPressure'] = df['BloodPressure'].replace(0, np.nan)
    df['SkinThickness'] = df['SkinThickness'].replace(0, np.nan)
    df['Insulin'] = df['Insulin'].replace(0, np.nan)
    df['BMI'] = df['BMI'].replace(0, np.nan)
    df['Glucose'] = df['Glucose'].fillna(df['Glucose'].mean())
    df['BloodPressure'] = df['BloodPressure'].fillna(df['BloodPressure'].mean())
    df['SkinThickness'] = df['SkinThickness'].fillna(df['SkinThickness'].mean())
    df['Insulin'] = df['Insulin'].fillna(df['Insulin'].mean())
    df['BMI'] = df['BMI'].fillna(df['BMI'].mean())

    print('----------------------------------------------')
    print("After preprocessing")
    print("Number of rows with 0 values for each variable")
    for col in df.columns:
        missing_rows = df.loc[df[col]==0].shape[0]
        print(col + ": " + str(missing_rows))
    print('----------------------------------------------')

    # Standardization
    df_scaled = preprocessing.scale(df)
    df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
    df_scaled['Outcome'] = df['Outcome']
    df = df_scaled
    

    return df



try:
    df = pd.read_csv('diabetes.csv')
except:
    print("""
      Dataset not found in your computer.
      Please follow the instructions in the link below to download the dataset:
      hhttps://github.com/stephenlmoshea/neural-network-to-predict-diabetes/blob/master/how_to_download_the_dataset.txt
      """)
    quit()

# Perform preprocessing and feature engineering
df = preprocess(df)

trainingSet = df.loc[:]

train, test = train_test_split(trainingSet, test_size=0.2)

sigmoid = Sigmoid()

networkLayer = [8,32,16,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.8, 0.05, 2000)

backpropagation.initialise()
result = backpropagation.train(train.values)

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

totalCorrect = 0
for num,row in enumerate(test.values):
    feedForward.activate(row[:8])
    outputs = feedForward.getOutputs()
    print("Expected: {}, Actual: {}".format(int(row[8]),round(outputs[0])))
    if(int(row[8]) == int(round(outputs[0]))):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))

```
## Training Neural Network to Detect Heart Arrhythmia
The dataset for this project is hosted by Kaggle. To download the necessary dataset for this project, please follow the instructions below.

1. Go to https://www.kaggle.com/shayanfazeli/heartbeat

2. Click on the 'Download All' button

3. Kaggle will prompt you to sign in or to register. If you do not have a Kaggle account, you can register for one.

4. Upon signing in, the download will start automatically.

5. After the download is complete, unzip the zip file and move the file 'mitbih_train.csv' and 'mitbih_test.csv' into your project folder.

```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import OneHotEncoder
from datetime import datetime

mit_test_data = pd.read_csv('mitbih_test.csv', header=None)

label_names = {0 : 'N',
              1: 'S',
              2: 'V',
              3: 'F',
              4 : 'Q'}

inputs = mit_test_data.iloc[:, :187]

targets = mit_test_data.iloc[:, 187:]

label_encoder = LabelEncoder()
integer_encoded = label_encoder.fit_transform(targets.values.reshape(-1,))

onehot_encoder = OneHotEncoder(sparse=False)
integer_encoded = integer_encoded.reshape(len(targets), 1)
onehot_encoded = onehot_encoder.fit_transform(targets)

outputs_df = pd.DataFrame.from_records(onehot_encoded)

trainingSet = pd.concat([inputs, outputs_df], axis=1)

train, test = train_test_split(trainingSet, test_size=0.2)

sigmoid = Sigmoid()

networkLayer = [187,50,50,5]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.8, 0.05, 1)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

backpropagation.initialise()
result = backpropagation.train(train.to_numpy().tolist())

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

print("Training time {} minutes".format(backpropagation.getTrainingTime()/60))

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

totalCorrect = 0
for num,row in enumerate(test.values):
    feedForward.activate(row[0:187])
    outputs = feedForward.getOutputs()
    actualClass = label_encoder.inverse_transform([argmax(outputs)])
    expectedClass = label_encoder.inverse_transform([argmax(row[187:])])

    print("Expected: {}, Predicted: {}".format(label_names[int(expectedClass)],label_names[int(actualClass)]))
    if(expectedClass == actualClass):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))
```
## Training Neural Network to Predict Breast Cancer
The dataset for this project is hosted by Kaggle. To download the necessary dataset for this project, please follow the instructions below.

1. Go to https://www.kaggle.com/uciml/breast-cancer-wisconsin-data

2. Click on the 'Download All' button

3. Kaggle will prompt you to sign in or to register. If you do not have a Kaggle account, you can register for one.

4. Upon signing in, the download will start automatically.

5. After the download is complete, unzip the zip file and move the file 'data.csv' into your project folder.

```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split

from datetime import datetime

data = pd.DataFrame()
data = pd.read_csv("./data.csv")

data.dropna(axis=1, inplace=True)

data.drop(labels='id',axis=1,inplace=True)

values = data.drop(labels='diagnosis',axis=1)
targets = data.diagnosis

encoder = LabelEncoder()
encoder.fit(targets)
encoded_targets = encoder.fit_transform(targets)

column_maxes = values.max()
df_max = column_maxes.max()
column_mins = values.min()
df_min = column_mins.min()
normalized_df = (values - df_min) / (df_max - df_min)

# print(encoded_targets)

normalized_df["encoded_targets"] = encoded_targets

train, test = train_test_split(normalized_df, test_size=0.2)

sigmoid = Sigmoid()

networkLayer = [30,60,30,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.7,0.8, 0.05, 2000)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

backpropagation.initialise()
result = backpropagation.train(train.values)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

print("Training time {} minutes".format(backpropagation.getTrainingTime()/60))

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

totalCorrect = 0
for num,row in enumerate(test.values):
    feedForward.activate(row[:30])
    outputs = feedForward.getOutputs()
    print("Expected: {}, Actual: {}".format(int(row[30]),round(outputs[0])))
    if(int(row[30]) == int(round(outputs[0]))):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))
```
## Training Neural Network to Predict Heart Disease
The dataset for this project is hosted by Kaggle. To download the necessary dataset for this project, please follow the instructions below.

The dataset for this project is hosted by Kaggle. To download the necessary dataset for this project, please follow the instructions below.

1. Go to https://www.kaggle.com/ronitf/heart-disease-uci

2. Click on the 'Download All' button

3. Kaggle will prompt you to sign in or to register. If you do not have a Kaggle account, you can register for one.

4. Upon signing in, the download will start automatically.

5. After the download is complete, unzip the zip file and move the file 'heart.csv' into your project folder.

```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

from sklearn.preprocessing import LabelEncoder

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split

from datetime import datetime

from sklearn.preprocessing import MinMaxScaler

data = pd.DataFrame()
data = pd.read_csv('./heart.csv')

data.dropna(axis=1, inplace=True)

min_max_scaler = MinMaxScaler()

data[["age", "cp", "trestbps", "chol", "thalach", "oldpeak", "slope", "thal"]] = min_max_scaler.fit_transform(data[["age", "cp", "trestbps", "chol", "thalach", "oldpeak", "slope", "thal"]])

train, test = train_test_split(data, test_size=0.2)

sigmoid = Sigmoid()

networkLayer = [13,26,13,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.3,0.5, 0.01, 2000)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

backpropagation.initialise()
result = backpropagation.train(train.values)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

print("Training time {} minutes".format(backpropagation.getTrainingTime()/60))

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

totalCorrect = 0
for num,row in enumerate(test.values):
    feedForward.activate(row[:13])
    outputs = feedForward.getOutputs()
    print("Expected: {}, Actual: {}".format(int(row[13]),round(outputs[0])))
    if(int(row[13]) == int(round(outputs[0]))):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))

```

## Training Neural Network to Predict Kidney Disease
The dataset for this project is hosted by Kaggle. To download the necessary dataset for this project, please follow the instructions below.

1. Go to https://www.kaggle.com/mansoordaku/ckdisease

2. Click on the 'Download All' button

3. Kaggle will prompt you to sign in or to register. If you do not have a Kaggle account, you can register for one.

4. Upon signing in, the download will start automatically.

5. After the download is complete, unzip the zip file and move the file 'kidney_disease.csv' into your project folder.

```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

from sklearn.impute import SimpleImputer

from numpy import nan
from numpy import isnan

ckd_df = pd.read_csv('./kidney_disease.csv')

col_dict={"bp":"blood_pressure",
          "sg":"specific_gravity",
          "al":"albumin",
          "su":"sugar",
          "rbc":"red_blood_cells",
          "pc":"pus_cell",
          "pcc":"pus_cell_clumps",
          "ba":"bacteria",
          "bgr":"blood_glucose_random",
          "bu":"blood_urea",
          "sc":"serum_creatinine",
          "sod":"sodium",
          "pot":"potassium",
          "hemo":"hemoglobin",
          "pcv":"packed_cell_volume",
          "wc":"white_blood_cell_count",
          "rc":"red_blood_cell_count",
          "htn":"hypertension",
          "dm":"diabetes_mellitus",
          "cad":"coronary_artery_disease",
          "appet":"appetite",
          "pe":"pedal_edema",
          "ane":"anemia"}

ckd_df.rename(columns=col_dict, inplace=True)

ckd_df['diabetes_mellitus'] =ckd_df['diabetes_mellitus'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})
ckd_df['coronary_artery_disease'] = ckd_df['coronary_artery_disease'].replace(to_replace='\tno',value='no')
ckd_df['white_blood_cell_count'] = ckd_df['white_blood_cell_count'].replace(to_replace='\t8400',value='8400')

ckd_df["classification"]=ckd_df["classification"].replace("ckd\t", "ckd")

ckd_df["white_blood_cell_count"]=ckd_df["white_blood_cell_count"].replace("\t?", np.nan)
ckd_df["red_blood_cell_count"]=ckd_df["red_blood_cell_count"].replace("\t?", np.nan)
ckd_df['diabetes_mellitus'] = ckd_df['diabetes_mellitus'].replace(to_replace={'\tno':'no','\tyes':'yes',' yes':'yes'})
ckd_df['coronary_artery_disease'] = ckd_df['coronary_artery_disease'].replace(to_replace='\tno',value='no')
ckd_df['white_blood_cell_count'] = ckd_df['white_blood_cell_count'].replace(to_replace='\t8400',value='8400')
ckd_df["packed_cell_volume"]= ckd_df["packed_cell_volume"].replace("\t?", np.nan)

for string_column in ["red_blood_cells","pus_cell","pus_cell_clumps","bacteria","hypertension","diabetes_mellitus","coronary_artery_disease","pedal_edema","anemia","appetite"]:
  ckd_df[string_column]=ckd_df[string_column].astype(str)

ckd_df['red_blood_cells']=ckd_df['red_blood_cells'].replace({'normal':1,'abnormal':0})
ckd_df['pus_cell']=ckd_df['pus_cell'].replace({'normal':1,'abnormal':0})
ckd_df['pus_cell_clumps']=ckd_df['pus_cell_clumps'].replace({'notpresent':0,'present':1})
ckd_df['bacteria']=ckd_df['bacteria'].replace({'notpresent':0,'present':1})
ckd_df['hypertension']=ckd_df['hypertension'].replace({'no':0,'yes':1})
ckd_df['diabetes_mellitus']=ckd_df['diabetes_mellitus'].replace({'no':0,'yes':1})
ckd_df['coronary_artery_disease']=ckd_df['coronary_artery_disease'].replace({'no':0,'yes':1})
ckd_df['pedal_edema']=ckd_df['pedal_edema'].replace({'no':0,'yes':1})
ckd_df['anemia']=ckd_df['anemia'].replace({'no':0,'yes':1})
ckd_df['appetite']=ckd_df['appetite'].replace({'poor':0,'good':1})
ckd_df['classification']=ckd_df['classification'].replace({'ckd':1,'notckd':0})

ckd_df.drop('id',axis=1,inplace=True)

values = ckd_df.values

imputer = SimpleImputer(missing_values=nan, strategy='mean')
# transform the dataset
transformed_values = imputer.fit_transform(values)

min_max_scaler = MinMaxScaler()

transformed_values = min_max_scaler.fit_transform(transformed_values)

train, test = train_test_split(transformed_values, test_size=0.2, stratify=transformed_values[:,24])

sigmoid = Sigmoid()

networkLayer = [24,48,24,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.3,0.6, 0.05)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

backpropagation.initialise()
result = backpropagation.train(train)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

print("Training time {} minutes".format(backpropagation.getTrainingTime()/60))

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

totalCorrect = 0
for num,row in enumerate(test):  
    feedForward.activate(row[0:24])
    outputs = feedForward.getOutputs()
    print("Expected: {}, Actual: {}".format(int(row[24]),round(outputs[0])))
    if(int(row[24]) == int(round(outputs[0]))):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))
```
## Training Neural Network to Predict Liver Disease
The dataset for this project is hosted by Kaggle. To download the necessary dataset for this project, please follow the instructions below.

1. Go to https://www.kaggle.com/uciml/indian-liver-patient-records

2. Click on the 'Download All' button

3. Kaggle will prompt you to sign in or to register. If you do not have a Kaggle account, you can register for one.

4. Upon signing in, the download will start automatically.

5. After the download is complete, unzip the zip file and move the file 'indian-liver-patient.csv' into your project folder.

```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

liver_df = pd.read_csv('./indian_liver_patient.csv')

liver_df = liver_df.dropna()

liver_df_normalised = liver_df.copy()

cleanup_nums = {"Gender":     {"Female": 1, "Male": 2}}

liver_df_normalised = liver_df_normalised.replace(cleanup_nums)

min_max_scaler = MinMaxScaler()

liver_df_normalised[liver_df_normalised.columns] = min_max_scaler.fit_transform(liver_df_normalised[liver_df_normalised.columns])

train, test = train_test_split(liver_df_normalised, test_size=0.2)

sigmoid = Sigmoid()

networkLayer = [10,20,10,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.3,0.6, 0.05,5000)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

backpropagation.initialise()
result = backpropagation.train(train.values)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

print("Training time {} minutes".format(backpropagation.getTrainingTime()/60))

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

totalCorrect = 0
for num,row in enumerate(test.values):    
    feedForward.activate(row[:10])
    outputs = feedForward.getOutputs()
    print("Expected: {}, Actual: {}".format(int(row[10]),round(outputs[0])))
    if(int(row[10]) == int(round(outputs[0]))):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))
```
## Training Neural Network to Predict Parkinsons Disease
The dataset for this project is hosted by Kaggle. To download the necessary dataset for this project, please follow the instructions below.

1. Go to https://www.kaggle.com/dipayanbiswas/parkinsons-disease-speech-signal-features

2. Click on the 'Download All' button

3. Kaggle will prompt you to sign in or to register. If you do not have a Kaggle account, you can register for one.

4. Upon signing in, the download will start automatically.

5. After the download is complete, unzip the zip file and move the file 'pd_speech_features.csv' into your project folder.

```py
from neuralnetwork.FeedForward import FeedForward
from neuralnetwork.Sigmoid import Sigmoid
from neuralnetwork.Backpropagation import Backpropagation

import pandas as pd
import numpy as np

from numpy import argmax

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from datetime import datetime

pd_speech_df = pd.read_csv('./pd_speech_features.csv')

pd_speech_df = pd_speech_df.dropna()

pd_speech_df_normalised = pd_speech_df.copy()

min_max_scaler = MinMaxScaler()

pd_speech_df_normalised[pd_speech_df_normalised.loc[:, pd_speech_df_normalised.columns != 'class'].columns] = min_max_scaler.fit_transform(pd_speech_df_normalised[pd_speech_df_normalised.loc[:, pd_speech_df_normalised.columns != 'class'].columns])

train, test = train_test_split(pd_speech_df_normalised, test_size=0.2, stratify=pd_speech_df_normalised['class'])

sigmoid = Sigmoid()

networkLayer = [754,70,40,1]

feedForward = FeedForward(networkLayer, sigmoid)

backpropagation = Backpropagation(feedForward,0.3,0.6, 0.05,50)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("Start Time =", current_time)

backpropagation.initialise()
result = backpropagation.train(train.values)

now = datetime.now()
current_time = now.strftime("%H:%M:%S")
print("End Time =", current_time)

print("Training time {} minutes".format(backpropagation.getTrainingTime()/60))

feedForward.save('./network.txt')

feedForward = FeedForward.load('./network.txt')

totalCorrect = 0
for num,row in enumerate(test.values):   
    feedForward.activate(row[:754])
    outputs = feedForward.getOutputs()
    print("Expected: {}, Actual: {}".format(int(row[754]),round(outputs[0])))
    if(int(row[754]) == int(round(outputs[0]))):
        totalCorrect = totalCorrect +1
    

percentageCorrect = totalCorrect/len(test.values) * 100

print(totalCorrect)

print("Percentage correct: {}%".format(percentageCorrect))
```