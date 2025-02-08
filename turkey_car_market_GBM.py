import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV
from sklearn.metrics import mean_absolute_percentage_error, mean_squared_error, r2_score
import datetime
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Download Dataset
data=pd.read_csv("turkey_car_market.csv")
df=data.copy()

# look at first 5 row
print(df.head())

#Data set shape
print(df.shape)

#Statisric value of  Data set
print(df.describe().T)

# Look at variable types in a data set
print(df.dtypes)

# Check for missing data in the data set
print("eksik veri:\n",df.isnull().sum())

#Separating Categorical variables in a data set
df_kat=df.select_dtypes(include=["object"])

#Separating numeric variables in a data set
df["Model Yıl"]=df["Model Yıl"].astype("int64")
df_num=df.select_dtypes(include=["int64"])

#Look at the relationship between Categorical Variables and Price variable ANOVA Test
# p < 0.05 → Strong association (variable may be significant)
# p > 0.05 → No strong association (variable may be insignificant for the model)
from scipy.stats import f_oneway
for col in df.select_dtypes(include=["object"]).columns:
    groups = [df[df[col] == val]["Fiyat"].dropna() for val in df[col].unique()]
    anova_p_value = f_oneway(*groups)[1]
    print(f"{col} ve fiyat arasındaki ANOVA p-değeri: {anova_p_value}")
    
    
# -----------------------DATA CLEANING --------------
df["Kontrol"]=(df["Km"]<0)|(df["Km"]>1000000)|(df["Model Yıl"]<1950)|(df["Model Yıl"]>2025)
df=df.loc[df["Kontrol"]==False].reset_index(drop=True)
df.drop(["Kontrol"],axis=1, inplace=True)

#Data Containing “Bilmiyorum”
for i in df_kat.columns:
    print(f"{i} sütununda 'Bilmiyorum' değeri sayısı:", (df_kat[i] == "Bilmiyorum").sum())

#Number of 'Bilmiyorum' values in the Beygir Gucu column: 5549, so we remove this column from the data set
df.drop(["Beygir Gucu"],axis=1, inplace=True)

# Number of 'Bilmiyorum' values in the CCM column: 108, so we remove these rows from the data set
ccm_drop=df.loc[df["CCM"]=="Bilmiyorum"]
df.drop(ccm_drop.index, inplace=True)

#Data with “-” in it
for i in df_kat.columns:
    print(f"{i} sütununda '-' değeri sayısı:", (df_kat[i] == "-").sum())
    
# of '-' values in the Arac Tipi column: 55, so we remove these rows from the data set
arac_tip=df.loc[df["Arac Tip"]=="-"]
df.drop(arac_tip.index, inplace=True)

# Number of '-' values in the CCM column: 1, so we remove these rows from the data set
ccm=df.loc[df["CCM"]=="-"]
df.drop(ccm.index, inplace=True)

#Outlier data analysis for price variable
q1=df["Fiyat"].quantile(0.25)
q3=df["Fiyat"].quantile(0.75)
IOC=q3-q1
alt_sinir=q1-1.5*IOC
ust_sinir=q3+1.5*IOC

#remove outlier observations from the data set
df["aykiri"]=(df["Fiyat"]<alt_sinir)|(df["Fiyat"]>ust_sinir)
df.drop((df.loc[df["aykiri"]==True]).index, inplace=True)

#Veri setinin son hali
df.drop(["aykiri", "İlan Tarihi"], axis=1, inplace=True)
df.index=np.arange(0,len(df))
# ----------------------- EXPLORATORY DATA ANALYSIS --------------
#Observing data distribution and outliers
plt.figure(figsize=(20,6))
sns.histplot(x="Km", data=df, kde=True, color="red")
plt.title("Km Değişkeni Dağılımları")
plt.figure(figsize=(20,6))
sns.histplot(x="Model Yıl", data=df, kde=True, color="red")
plt.title("Model Yılı Değişkeni Dağılımları")
plt.figure(figsize=(20,6))
sns.histplot(x="Fiyat", data=df, kde=True, color="red")
plt.title("Yiyat Değişkeni Dağılımları")

# looking at outlier data
plt.figure(figsize=(5,26))
sns.boxplot(df)

#Kasa Tipi Variable's relationship with Price
plt.figure(figsize=(20,6))
sns.barplot(x=df["Kasa Tipi"], y=df["Fiyat"], data=df)
plt.title("Kasa Tipine Göre Ortalama Fiyatlar")
#Yakıt Türü Variable's relationship with Price
plt.figure(figsize=(20,6))
sns.barplot(x=df["Yakıt Turu"], y=df["Fiyat"], data=df)
plt.title("Yakıt Turune Göre Ortalama Fiyatlar")
#Vites Variable's relationship with Price
plt.figure(figsize=(20,6))
sns.barplot(x=df["Vites"], y=df["Fiyat"], data=df)
plt.title("Vites Turune Göre Ortalama Fiyatlar")
#Renk Variable's relationship with Price
plt.figure(figsize=(20,6))
sns.barplot(x=df["Renk"], y=df["Fiyat"], data=df)
plt.title("Renklere Göre Ortalama Fiyatlar")
plt.xticks(rotation=90)
#Km Variable's relationship with Price
plt.figure(figsize=(20,6))
sns.scatterplot(x=df["Km"], y=df["Fiyat"], data=df)
plt.title("Km'ye Göre Fiyat Dağılımı")
#Model Yılı Variable's relationship with Price
plt.figure(figsize=(20,6))
sns.scatterplot(x=df["Model Yıl"], y=df["Fiyat"], data=df)
plt.title("Model Yılı'na Göre Fiyat Dağılımı")


#---------------TRANSFORMATION OF CATEGORICAL VARIABLES--------------
df=pd.get_dummies(df, columns=["Vites"], dtype=int)
df=pd.get_dummies(df, columns=["Kimden"], dtype=int)
df=pd.get_dummies(df, columns=["Durum"], dtype=int)
df=pd.get_dummies(df, columns=["Yakıt Turu"], dtype=int)
lb=LabelEncoder()

df["Arac Tip Grubu"]=lb.fit_transform(df["Arac Tip Grubu"])
df["Arac Tip"]=lb.fit_transform(df["Arac Tip"])
df["CCM"]=lb.fit_transform(df["CCM"])
df["Renk"]=lb.fit_transform(df["Renk"])
df["Kasa Tipi"]=lb.fit_transform(df["Kasa Tipi"])
df["Marka"]=lb.fit_transform(df["Marka"])

##Correlation Map
plt.figure(figsize=(20,20))
sns.heatmap(df.corr(), annot=True)

#Bütün kolonlara ile fiyat kolonu ilişkisi
fig = plt.figure(figsize = (15,20))
ax = fig.gca()
df.hist(ax = ax)

#-----------------SEPARATION OF TRAINING AND TEST DATA-------------
X=df.drop(["Fiyat"], axis=1)
y=df["Fiyat"].values
y=y.flatten()
y=y.reshape(-1,1)


#------------------------STANDARDIZATION OF DATA--------------------
scaler_X=MinMaxScaler()
scaler_y=MinMaxScaler()
X=scaler_X.fit_transform(X)
y=scaler_y.fit_transform(y)

#------------------------TRAİN AND TEST SPLIT--------------------
X_train, X_test, y_train, y_test=train_test_split(X,y, test_size=0.2, random_state=42)

#-------------------------------MODEL----------------------------
gbm_model=GradientBoostingRegressor()
gbm_model.fit(X_train, y_train)
y_pred=gbm_model.predict(X_test)

mse=mean_squared_error(y_test, y_pred)
r2_skor=r2_score(y_test, y_pred)
print(f"Modelin MSE Hata Oranı:{mse:4f}\nModelin R^2 Oranı:{r2_skor:.4f}")


#----------------MODEL OPTIMIZATION WITH GRAYD SEARCH---------------
params={"learning_rate":[0.001,0.01,0.1,0.2],
        "max_depth":[3,5,8,15,20,50],
        "n_estimators":[200,500,1000,2000],
        "subsample":[1,0.5,0.75]}

gbm_model_grid=GridSearchCV(gbm_model, params, cv=10, n_jobs=-1)
gbm_model_grid.fit(X_train, y_train)

#Retraining the model with the best parameters
gbm_model_tuned=GradientBoostingRegressor(min_samples_split=gbm_model_grid.best_params_["min_samples_split"], max_leaf_nodes=gbm_model_grid.best_params_["max_leaf_nodes"] )
gbm_model_tuned.fit(X_train, y_train)
y_pred_tuned=gbm_model_tuned.predict(X_test)

mse=mean_squared_error(y_test, y_pred_tuned)
r2_skor=r2_score(y_test, y_pred_tuned)
print(f"Grid Search ile Optimize Edilmiş Modelin MSE Hata Oranı:{mse:.4f}\nGrid Search ile Optimize Edilmiş Modelin R^2 Oranı:{r2_skor:.4f}")




