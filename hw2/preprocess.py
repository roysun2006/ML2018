import os, csv
import pandas as pd
import numpy as np

dir = os.path.abspath(os.path.dirname(__file__))
trainpath = os.path.join(dir, 'data\\train.csv')
testpath = os.path.join(dir, 'data\\test.csv')


def process(Data):
  Data['sex'] = (Data['sex'] == ' Male').astype("int64")
  ObjectColumn = [col for col in Data.columns if Data[col].dtypes=="object"]
  NonObjectColumn = [col for col in Data.columns if Data[col].dtypes != "object"]
  NonObjectData = Data[NonObjectColumn]
  ObjectData = Data[ObjectColumn]
  ObjectData = pd.get_dummies(ObjectData);
  Data = NonObjectData.join(ObjectData).astype("int64")
  return Data



Train = pd.read_csv(trainpath)
Test = pd.read_csv(testpath)

#print(Train.head())
#print(Test.head())
Train_y = pd.DataFrame((Train['income'] == ' >50K').astype("int64"), columns = ['income'])
Train = Train.drop(['income'], axis = 1)
Train_data = process(Train)
Test_data = process(Test)

Train_data.to_csv(os.path.join(dir, "data\\traindata.csv"), index=False)
Test_data.to_csv(os.path.join(dir, "data\\testdata.csv"), index=False)
Train_y.to_csv(os.path.join(dir, "data\\trainlabel.csv"), index=False)


