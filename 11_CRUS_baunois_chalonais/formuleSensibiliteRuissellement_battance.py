# -*- coding: utf-8 -*-
"""
Created on Mon Jan 13 15:57:39 2020
Mis à jour le 18/03/2022

@author: manuel.collongues
Cerema / Laboratoire de Nancy / ERTD

Ce script permet de calculer la sensibilité à la production de ruissellement (selon la méthode CRUS développée par le Cerema de 2018 à 2020) d'un territoire à partir des données d'entrée suivantes :
    - pentes
    - permeabilite
    - occupation du sol
    - battance
    
Ce script suppose de fournir au format gtif chacun de ces raster, avec exactement les mêmes bornes et résolution spatiale
Il calcule 3 rasters :
    - un issu de la formule CRUS sans prendre en compte la battance
    - un issu de la formule CRUS en prenant en compte la battance
    - un faisant la différence booléenne entre les deux premiers
    
Ce script se révèle significativement plus rapide que la même méthode exécutée sous Qgis (via la calculatrice raster) car il tire parti du calcul hautement parallélisé sous GPU. 
"""

from __future__ import division
import numpy as np
from numba import cuda
import math
import datetime
from osgeo import gdal


# CUDA kernel
@cuda.jit
def my_kernel(dataPentes,dataPermea,dataOccSol,dataBattance,dataResult,data_Ssbatt,dataDiff,tabCorCod,tabCoefBat):
    pos = cuda.grid(1)
    for i in range(tabCorCod.shape[0]):
        if dataPentes[pos] == tabCorCod[i,0]:
            dataPentes[pos] = tabCorCod[i,1]
    for i in range(tabCoefBat.shape[0]):
        if dataBattance[pos] == tabCoefBat[i,0]:
            dataResult[pos] = tabCoefBat[i,1] * (0 * dataPentes[pos] + 0.6 * dataPermea[pos] + 0.4 * dataOccSol[pos]) / 100
    data_Ssbatt[pos] = 0 * dataPentes[pos] + 0.6 * dataPermea[pos] + 0.4 * dataOccSol[pos]
                 
    if dataResult[pos] < 0:
        dataResult[pos] = -1
    elif dataResult[pos] <= 32:
        dataResult[pos] = 0
    elif dataResult[pos] <= 42:
        dataResult[pos] = 1
    elif dataResult[pos] <=55:
        dataResult[pos] = 2
    else:
        dataResult[pos] = 3
        
    if data_Ssbatt[pos] < 0:
        data_Ssbatt[pos] = -1
    elif data_Ssbatt[pos] <= 32:
        data_Ssbatt[pos] = 0
    elif data_Ssbatt[pos] <= 42:
        data_Ssbatt[pos] = 1
    elif data_Ssbatt[pos] <=55:
        data_Ssbatt[pos] = 2
    else:
        data_Ssbatt[pos] = 3
    
    if data_Ssbatt[pos] == dataResult[pos]:
        dataDiff[pos] = 1
    else:
        dataDiff[pos] = -1

fichierRasterPentes = "classe pente.tif"
fichierRasterPermea = "permeabilite.tif"
fichierRasterOccSol = "occupation_territoire.tif"
fichierRasterBattance = "battance.tif"

fichierSortieCRUS_battance = "SensibiliteProdRuiss_Bat2_sansPente.tif"
fichierSortieCRUS_DiffsansBattance = "Difference_Bat_SsBat2_sansPente.tif"


# Lecture tableau de correspondance du reclass
t0 = datetime.datetime.now()
tableauCorrespondanceCodes=[]
debut = datetime.datetime.now()
with open ("code_reclassPentes.txt", "r", encoding = "utf-8") as mat:
    for ligne in mat:
        lignee = ligne.split("\n")[0].split("=")
        code = int(lignee[0])
        codeMat = int(lignee[1])
        tableauCorrespondanceCodes.append([code, codeMat])
    mat.close()
tabCorCod = np.array(tableauCorrespondanceCodes)
print("...Lecture tableau de correspondance Terminée en : ", datetime.datetime.now() - debut)

# Lecture tableau des coefficients multiplicateurs du reclass
t0 = datetime.datetime.now()
tableauCoefBat=[]
debut = datetime.datetime.now()
with open ("codes_battance.txt", "r", encoding = "utf-8") as mat:
    for ligne in mat:
        lignee = ligne.split("\n")[0].split("=")
        code = int(lignee[0])
        codeMat = int(lignee[1])
        tableauCoefBat.append([code, codeMat])
    mat.close()
tabCoefBat = np.array(tableauCoefBat)
print("...Lecture tableau de correspondance Terminée en : ", datetime.datetime.now() - debut)

# Lecture du raster des valeurs CRUS de pentes
debut = datetime.datetime.now()
#raster = lecture(fichier, dim1, dim2)
ds = gdal.Open(fichierRasterPentes)
rasterPentes = np.array(ds.GetRasterBand(1).ReadAsArray())
(dim1,dim2) = rasterPentes.shape
print("\nPente - dim1 : ",dim1,"\ndim 2 : ",dim2)
print("...Lecture rasterPentes terminée en : ", datetime.datetime.now() - debut)

# Lecture du raster des valeurs CRUS de permea
debut = datetime.datetime.now()
#raster = lecture(fichier, dim1, dim2)
ds = gdal.Open(fichierRasterPermea)
rasterPermea = np.array(ds.GetRasterBand(1).ReadAsArray())
#(dim1,dim2) = rasterPermea.shape
print("\nPente - dim1 : ",dim1,"\ndim 2 : ",dim2)
print("...Lecture rasterPermea terminée en : ", datetime.datetime.now() - debut)

# Lecture du raster des valeurs CRUS d'occupation du sol
debut = datetime.datetime.now()
#raster = lecture(fichier, dim1, dim2)
ds = gdal.Open(fichierRasterOccSol)
rasterOccSol = np.array(ds.GetRasterBand(1).ReadAsArray())
#(dim1,dim2) = rasterOccSol.shape
print("\nPente - dim1 : ",dim1,"\ndim 2 : ",dim2)
print("...Lecture rasterOccSol terminée en : ", datetime.datetime.now() - debut)

# Lecture du raster des valeurs CRUS de battance
debut = datetime.datetime.now()
#raster = lecture(fichier, dim1, dim2)
ds = gdal.Open(fichierRasterBattance)
rasterBattance = np.array(ds.GetRasterBand(1).ReadAsArray())
#(dim1,dim2) = rasterOccSol.shape
print("\nPente - dim1 : ",dim1,"\ndim 2 : ",dim2)
print("...Lecture rasterBattance terminée en : ", datetime.datetime.now() - debut)


# Host code   
dataPentes = rasterPentes.reshape(dim1*dim2)
dataPermea = rasterPermea.reshape(dim1*dim2)
dataOccSol = rasterOccSol.reshape(dim1*dim2)
dataBattance = rasterBattance.reshape(dim1*dim2)
dataResult = dataOccSol.copy()
dataDiff = dataOccSol.copy()
data_Ssbatt = dataResult.copy()
debut = datetime.datetime.now()
threadsperblock = 1024
blockspergrid = math.ceil(dataResult.shape[0] / threadsperblock)
my_kernel[blockspergrid, threadsperblock](dataPentes,dataPermea,dataOccSol,dataBattance,dataResult,data_Ssbatt,dataDiff,tabCorCod,tabCoefBat)
print("...Traitement GPU terminé en : ", datetime.datetime.now() - debut)
raster_result = dataResult.reshape(dim1,dim2)
raster_diff = dataDiff.reshape(dim1,dim2)


# Ecriture du fichier de sortie
debut = datetime.datetime.now()
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(fichierSortieCRUS_battance, dim2, dim1, 1, gdal.GDT_Int16)
outdata.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
outdata.SetProjection(ds.GetProjection())##sets same projection as input
outdata.GetRasterBand(1).WriteArray(raster_result)
outdata.GetRasterBand(1).SetNoDataValue(-1)##if you want these values transparent
outdata.FlushCache() ##saves to disk!!

print("...Ecriture fichier sortie terminée en : ", datetime.datetime.now() - debut)
print("...Temps total de traitement : ", datetime.datetime.now() - t0)

# Ecriture du fichier de sortie
debut = datetime.datetime.now()
driver = gdal.GetDriverByName("GTiff")
outdiff = driver.Create(fichierSortieCRUS_DiffsansBattance, dim2, dim1, 1, gdal.GDT_Int16)
outdiff.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
outdiff.SetProjection(ds.GetProjection())##sets same projection as input
outdiff.GetRasterBand(1).WriteArray(raster_diff)
outdiff.GetRasterBand(1).SetNoDataValue(-1)##if you want these values transparent
outdiff.FlushCache() ##saves to disk!!
outdiff = None
ds=None

outdata = None
ds=None
print("...Ecriture fichier sortie terminée en : ", datetime.datetime.now() - debut)
print("...Temps total de traitement : ", datetime.datetime.now() - t0)

















