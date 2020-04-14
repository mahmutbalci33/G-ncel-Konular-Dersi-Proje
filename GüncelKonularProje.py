# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:44:47 2020

@author: Lenovo
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
#veri yükleme
veriler=pd.read_csv('güncelkonular.csv')
print(veriler)

#eksik verileri tamamlamama



from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)

tamamlanmisveriler=veriler.iloc[:,1:6].values
imputer=imputer.fit(tamamlanmisveriler[:,1:6])

tamamlanmisveriler[:,1:6]=imputer.transform(tamamlanmisveriler[:,1:6])


tamamlanmisveriler= pd.DataFrame(data=tamamlanmisveriler,  columns=['PM10','SO2','CO','NO2','O3'])

tarih=veriler.iloc[:,0:1].values

tarih=pd.DataFrame(data=tarih,columns=['Tarih'])

dataframe=pd.concat([tarih,tamamlanmisveriler],axis=1)

plt.figure(figsize=(25,15))
plt.plot(dataframe.index, dataframe.PM10, color="red", linestyle="--", alpha=0.9)
plt.plot(dataframe.index, dataframe.SO2, color="green", linestyle="--", alpha=0.9)
plt.plot(dataframe.index, dataframe.CO, color="purple", linestyle="--", alpha=0.9)
plt.plot(dataframe.index, dataframe.NO2, color="pink", linestyle="--", alpha=0.9)
plt.plot(dataframe.index, dataframe.O3, color="black",linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri")
plt.show()


plt.figure(figsize=(25,15))
plt.plot(dataframe.index, dataframe.PM10, color="red", linestyle="--", alpha=0.9)
plt.plot(dataframe.index, dataframe.SO2, color="green", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri")
plt.show()


plt.figure(figsize=(25,15))
plt.plot(dataframe.index, dataframe.PM10, color="red", linestyle="--", alpha=0.9)
plt.plot(dataframe.index, dataframe.NO2, color="green", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri")
plt.show()

plt.figure(figsize=(25,15))
plt.plot(dataframe.index, dataframe.PM10, color="red", linestyle="--", alpha=0.9)
plt.plot(dataframe.index, dataframe.O3, color="green", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri")
plt.show()


plt.figure(figsize=(25,15))
plt.plot(dataframe.index, dataframe.NO2, color="red", linestyle="--", alpha=0.9)
plt.plot(dataframe.index, dataframe.SO2, color="green", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri")
plt.show()


plt.figure(figsize=(25,15))
plt.plot(dataframe.index, dataframe.NO2, color="red", linestyle="--", alpha=0.9)
plt.plot(dataframe.index, dataframe.O3, color="green", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri")
plt.show()

plt.figure(figsize=(25,15))
plt.plot(dataframe.index, dataframe.CO, color="red", linestyle="--", alpha=0.9)
plt.plot(dataframe.index, dataframe.O3, color="green", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri")
plt.show()

