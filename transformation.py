import json

import pandas as pd
import os
import requests
import sklearn
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score




def nettoyage_file(filecsv):
        fichier1 = pd.read_csv(filecsv)
        fichier1 = fichier1.iloc[5:]
        fichier1.columns =["date","CO","PM10","PM25","NO2","NO","NOX","O3","OBJECTID"]
        fichier1.drop(["CO","O3","NO2","NO","NOX","OBJECTID"],axis=1,inplace=True)
        if fichier1["date"].str.contains('|'.join(['2020','2O21'])).any() :
                fichier1["date"] = fichier1["date"].str.replace("2020","2020").str.replace("2021","2021").str.replace("/","-")
        elif fichier1["date"].str.contains('|'.join(['2021','2O22'])).any():
                fichier1["date"] = fichier1["date"].str.replace("2021","2021").str.replace("2022","2022").str.replace("/","-")
        elif fichier1["date"].str.contains('|'.join(['2022','2O23'])).any():
                fichier1["date"] = fichier1["date"].str.replace("2022","2022").str.replace("2023","2023").str.replace("/","-")
        fichier1[["date","heure"]] = fichier1["date"].str.split(" ",expand=True)
        fichier1["heure"]=fichier1["heure"].str.replace(":00:00+00","")
        fichier1 = fichier1.astype({"PM10": "float","PM25":"float"})
        position_col = ["date","PM10","PM25"]
        fichier1 = fichier1[position_col]
        resultat = fichier1.groupby(['date'],group_keys=False)[['PM10','PM25']].mean().reset_index()
        position_col = ["date","PM10","PM25"]
        resultat = resultat[position_col]
        resultat["PM10"] = resultat["PM10"].round(2)
        resultat["PM25"] = resultat["PM25"].round(2)
        resultat.to_csv("output"+"/"+filecsv[0:5]+"sortie"+".csv",index=False)

#nettoyage_file("2022_PA01H.csv")

def fusioner_les_data(dossier):
        directory_path = dossier
        directory = os.fsencode(directory_path)
        list_files = []
        for file in os.listdir(directory):
                filename = os.fsdecode(file)
                list_files.append(directory_path + "/" + filename)
        dict = {}
        for i, value in enumerate(list_files):
                dict[i] = value
        dict_dataframe = {}
        list_dataframe = []
        for fichier in dict.keys():
                datafame = pd.read_csv(dict[fichier])
                dict_dataframe[fichier] = datafame
        for datafame in dict_dataframe.values():
                list_dataframe.append(datafame)
                nouveau_dataframe = pd.concat(list_dataframe)
        return nouveau_dataframe

le_data_frame = fusioner_les_data("output")


le_data_frame.to_csv("dataframe.csv",index=False)
"""
date_manque = le_data_frame[(le_data_frame['PM10'].isna() == True) | (le_data_frame["PM25"].isna()==True)]
date_manque["date"].nunique()



reponse = requests.get("https://api.zippopotam.us/us/10012")
json_file = reponse.json()
json_file["country"]
"""

def créer_airaprif_2023(PM10,PM25):

        pm10 = pd.read_csv(PM10)
        pm25 = pd.read_csv(PM25)
        pm10 = pm10[["Unnamed: 0","PA01H:PM10"]]
        pm10.columns = ["date", "PM10"]
        pm10 = pm10.iloc[5:]
        pm10 = pm10.astype({"PM10": "float"})
        pm10[["date", "heure"]] = pm10["date"].str.split(" ", expand=True)
        pm10 = pm10.drop("heure", axis=1)
        pm10 = pm10.groupby(['date'], group_keys=False)[['PM10']].mean().reset_index()

        pm25 = pd.read_csv(PM25)
        pm25 = pm25[["Unnamed: 0", "PA01H:PM25"]]
        pm25.columns = ["date", "PM25"]
        pm25 = pm25.iloc[5:]
        pm25 = pm25.astype({"PM25": "float"})
        pm25[["date", "heure"]] = pm25["date"].str.split(" ", expand=True)
        pm25 = pm25.drop("heure", axis=1)
        pm25 = pm25.groupby(['date'], group_keys=False)[['PM25']].mean().reset_index()

        airparif2023 =  pd.merge(pm10,pm25,how="inner")
        airparif2023 =airparif2023[airparif2023["date"].str.contains("2023/12")==False]
        airparif2023["date"] = airparif2023["date"].str.replace("/","-")
        airparif2023["PM10"] = airparif2023["PM10"].round(2)
        airparif2023["PM25"] = airparif2023["PM25"].round(2)
        return airparif2023.to_csv("output/2023_sortie.csv",index=False)



créer_airaprif_2023("2023_PM10.csv","2023_PM25.csv")

#moyenne datset final
mon_dataframe = pd.read_csv("df_compl_new (1).csv",index_col=False)
mon_dataframe[["Date", "heure"]] = mon_dataframe["Date"].str.split(" ", expand=True)
mon_dataframe = mon_dataframe.drop("heure",axis=1)
mon_dataframe= mon_dataframe.groupby(['Date'], group_keys=False).mean().reset_index()
mon_dataframe.to_csv("final_means.csv",index=False)


X_train, X_test, y_train, y_test = train_test_split(mon_dataframe["AirParif_pm2p5"].values.reshape(-1,1),
                                                    mon_dataframe["Copernecus_pm2p5"], test_size=0.2, random_state=42)
model = LinearRegression()
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)


print(f'Mean Squared Error (MSE): {mse}')
print(f'R-squared (R2): {r2}')
