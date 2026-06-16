import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import os,sys
sys.path.append(os.path.abspath("C:/Users/hp/Desktop/Projet_fin_etude/core"))
from preprocessing_vmd import load_data_vmd

train,test,scaler_y=load_data_vmd()
'''
print("train shape",train.shape)
print("test shape",test.shape)
print("les colonnes",train.columns)
print("description:",train.describe())

print("les nulles:",train.isnull().sum())

plt.figure()
sns.histplot(train["energy"],bins=50,kde=True)
plt.title("distribution d'energy")
plt.show()

plt.figure(figsize=(15,5))
plt.plot(train["energy"][:500])
plt.title("time series")
plt.show()
'''
day=train[train["global_radiation"]>0]
night = train[train["global_radiation"] < 2]  # seuil à ajuster
print("combien de valeur de jour:",len(day))
print("combien de valeur de nuit",len(night))
plt.figure()
sns.histplot(day["energy"],color="orange",label="Day",kde=True)
sns.histplot(night["energy"],color="Blue",label="night",kde=True)
plt.legend()
plt.title("night vs day")
plt.show