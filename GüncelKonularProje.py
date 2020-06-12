# -*- coding: utf-8 -*-
"""
Created on Wed Apr  8 22:44:47 2020

@author: Lenovo
"""

import pandas as pd
from sklearn.metrics import r2_score
#veri yükleme
veriler=pd.read_csv('veriyeni.csv')

#eksik verileri tamamlamama


from sklearn.preprocessing import Imputer
imputer=Imputer(missing_values='NaN', strategy='mean', axis=0)

tamamlanmisveriler=veriler.iloc[:,1:].values

imputer=imputer.fit(tamamlanmisveriler[:,:])

tamamlanmisveriler[:,:]=imputer.transform(tamamlanmisveriler[:,:])


tamamlanmisveriler= pd.DataFrame(data=tamamlanmisveriler,  columns=['PM10','SO2','CO','NO2','O3'])

tarih=veriler.iloc[:,0:1].values

tarih=pd.DataFrame(data=tarih,columns=['Tarih'])

dataframe=pd.concat([tarih,tamamlanmisveriler],axis=1)

tamamlanmisveriler= pd.DataFrame(data=tamamlanmisveriler,  columns=['PM10','SO2','CO','NO2','O3'])


'''
plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.PM10, color="black", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.SO2, color="green", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:PM10, SO2")
plt.show()


plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.PM10, color="red", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.NO2, color="grey", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:PM10, NO2")
plt.show()

plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.PM10, color="brown", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.O3, color="yellow", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:PM10, O3")
plt.show()

plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.PM10, color="red", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.CO, color="black", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:PM10, CO")
plt.show()


plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.NO2, color="orange", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.SO2, color="green", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:NO2, SO2")
plt.show()


plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.NO2, color="red", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.O3, color="green", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:NO2, O3")
plt.show()

plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.NO2, color="black", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.CO, color="red", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:NO2, CO")
plt.show()


plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.CO, color="blue", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.SO2, color="grey", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:CO,SO2")
plt.show()

plt.figure(figsize=(25,15))
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.CO, color="yellow", linestyle="--", alpha=0.9)
plt.plot(tamamlanmisveriler.index, tamamlanmisveriler.O3, color="black", linestyle="--", alpha=0.9)
plt.xlabel("Zaman")
plt.ylabel("Ölçüm Parametreleri:CO, O3")
plt.show()
'''

PM10=tamamlanmisveriler[['PM10']]
SO2=tamamlanmisveriler[['SO2']]
CO=tamamlanmisveriler[['CO']]
NO2=tamamlanmisveriler[['NO2']]
O3=tamamlanmisveriler[['O3']]
#Linear Regression

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test=train_test_split(NO2, PM10, test_size=0.3, random_state=0)

from sklearn.linear_model import LinearRegression
lr=LinearRegression()
lr.fit(x_train, y_train)

PM10_lrreg=lr.predict(x_test)  

#Çoklu Doğrusal Regression
from sklearn.linear_model import LinearRegression

ml_veri=pd.concat([NO2, SO2], axis=1)
x_train, x_test, y_train, y_test=train_test_split(ml_veri, PM10, test_size=0.3, random_state=0)
ml_reg=LinearRegression()
ml_reg.fit(x_train, y_train)
PM10_ml_reg=ml_reg.predict(x_test)

#Polinomal Regression
from sklearn.preprocessing import PolynomialFeatures
poly_reg=PolynomialFeatures(degree=2)
NO2_poly=poly_reg.fit_transform(NO2)
lin_reg2=LinearRegression()
lin_reg2.fit(NO2_poly,PM10)
PM10_poly=lin_reg2.predict(poly_reg.fit_transform(NO2))

#Support Vektör
from sklearn.preprocessing import StandardScaler
sc1=StandardScaler()
NO2_olcek=sc1.fit_transform(NO2)
sc2=StandardScaler()
PM10_olcek=sc2.fit_transform(PM10)

from sklearn.svm import SVR
svr_reg=SVR(kernel='rbf')
svr_reg.fit(NO2_olcek, PM10_olcek)
PM10_svr=svr_reg.predict(PM10_olcek)

#Karar Ağacı
from sklearn.tree import DecisionTreeRegressor
detree=DecisionTreeRegressor(random_state=0)
detree.fit(NO2,PM10)
PM10_detree=detree.predict(PM10)


#Random Forest
from sklearn.ensemble import RandomForestRegressor
rf_reg=RandomForestRegressor(n_estimators=10, random_state=0)
rf_reg.fit(NO2, PM10)
PM10_rf=rf_reg.predict(PM10)

PM10Normal=PM10

PolyBasari= (r2_score(SO2, PM10_poly ))

LinearBasari=(r2_score(y_test, PM10_lrreg ))

MlRegBasari=(r2_score(y_test, PM10_ml_reg ))

SVRBasari=(r2_score(PM10, PM10_svr) )

DeTreeBasari=(r2_score(PM10, PM10_detree ))

RFBasari=(r2_score(PM10, PM10_rf))





