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
import threading

def warp_raster(fichierRaster, raster_warped, outputBounds, resX, resY):
    ds = gdal.Open(fichierRaster)
    print("début warp")
    raster = gdal.Warp(raster_warped, ds, outputBounds=outputBounds, xRes=resX, yRes=resY, resampleAlg='bilinear', outputType=gdal.GDT_Int16, multithread=True)
    ds = None
    print("fin warp")
    
    

def lire_raster(fichier, designation):
    debut = datetime.datetime.now()
    ds = gdal.Open(fichierRasterBattance)
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = geotransform[0] + (geotransform[1] * ds.RasterXSize)
    ymin = geotransform[3] + (geotransform[5] * ds.RasterYSize)
    resX = geotransform[1]
    resY = -geotransform[5]
    print(f"...Lecture {designation} terminée en : ", datetime.datetime.now() - debut)
    return {"ds":ds, "geotransform":geotransform, "projection":projection, "xmin":xmin, "ymax":ymax, "xmax":xmax, "ymin":ymin, "resX":resX, "resY":resY}

# CUDA kernel
@cuda.jit
def my_kernel(dataPentes,dataPermea,dataOccSol,dataBattance,dataResult,data_Ssbatt,dataDiff,tabCorCod,tabCoefBat):
    pos = cuda.grid(1)
    for i in range(tabCorCod.shape[0]):
        if dataPentes[pos] == tabCorCod[i,0]:
            dataPentes[pos] = tabCorCod[i,1]
    for i in range(tabCoefBat.shape[0]):
        if dataBattance[pos] == tabCoefBat[i,0]:
            dataResult[pos] = tabCoefBat[i,1] * (0.4 * dataPentes[pos] + 0.35 * dataPermea[pos] + 0.25 * dataOccSol[pos]) / 100
    data_Ssbatt[pos] = 0.4 * dataPentes[pos] + 0.35 * dataPermea[pos] + 0.25 * dataOccSol[pos]
                 
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

fichierRasterPentes = "pentes.tif"
fichierRasterPermea = "permea.tif"
fichierRasterOccSol_5a = "os5ans-vf.tif"
fichierRasterOccSol_2021 = "os-2021-vf.tif"
fichierRasterOccSol_modif = "os_modif-vf.tif"
fichierRasterBattance = "battance.tif"

fichierSortieCRUS_battance = "SensibiliteProdRuiss_Bat2.tif"
fichierSortieCRUS_DiffsansBattance = "Difference_Bat_SsBat2.tif"


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



# Découpage selon l'emprise minimale

dico = {"pente": lire_raster(fichierRasterPentes, "pente"), 
        "permea": lire_raster(fichierRasterPermea, "permea"), 
        "occsol": lire_raster(fichierRasterOccSol_5a, "occsol"), 
        "occsol": lire_raster(fichierRasterOccSol_2021, "occsol"), 
        "occsol": lire_raster(fichierRasterOccSol_modif, "occsol"), 
        "batt": lire_raster(fichierRasterBattance, "batt")}

# Définir les dimensions de l'emprise minimale pour tous les rasters
xmin = max([ds["xmin"] for ds in dico.values()])
ymin = max([ds["ymin"] for ds in dico.values()])
xmax = min([ds["xmax"] for ds in dico.values()])
ymax = min([ds["ymax"] for ds in dico.values()])

# Définir la résolution souhaitée pour tous les rasters
resX = max([ds["resX"] for ds in dico.values()])
resY = max([ds["resY"] for ds in dico.values()])

# Découper chaque raster pour qu'il corresponde à l'emprise et la résolution souhaitées

t1 = threading.Thread(target=warp_raster, args=(fichierRasterPermea, "permearepr", [xmin, ymin, xmax, ymax], resX, resY, ))
t2a = threading.Thread(target=warp_raster, args=(fichierRasterOccSol_5a, "occsolrepr_5a", [xmin, ymin, xmax, ymax], resX, resY, ))
t2b = threading.Thread(target=warp_raster, args=(fichierRasterOccSol_2021, "occsolrepr_2021", [xmin, ymin, xmax, ymax], resX, resY, ))
t2c = threading.Thread(target=warp_raster, args=(fichierRasterOccSol_modif, "occsolrepr_modif", [xmin, ymin, xmax, ymax], resX, resY, ))
t3 = threading.Thread(target=warp_raster, args=(fichierRasterBattance, "battrepr", [xmin, ymin, xmax, ymax], resX, resY, ))
t4 = threading.Thread(target=warp_raster, args=(fichierRasterPentes, "penterepr", [xmin, ymin, xmax, ymax], resX, resY, ))

t1.start()
t2a.start()
t2b.start()
t2c.start()
t3.start()
t4.start()

t1.join()
t2a.join()
t2b.join()
t2c.join()
t3.join()
t4.join()


rasterPermea = gdal.Open("permearepr").ReadAsArray()
rasterOccSol_5a = gdal.Open("occsolrepr_5a").ReadAsArray()
rasterOccSol_2021 = gdal.Open("occsolrepr_2021").ReadAsArray()
rasterOccSol_modif = gdal.Open("occsolrepr_modif").ReadAsArray()
rasterBattance = gdal.Open("battrepr").ReadAsArray()
rasterPentes = gdal.Open("penterepr").ReadAsArray()

'''
ds = gdal.Open(fichierRasterPermea)
print("début warp permea")
rasterPermea = gdal.Warp('permearepr', ds, outputBounds=[xmin, ymin, xmax, ymax], xRes=resX, yRes=resY, resampleAlg='bilinear', outputType=gdal.GDT_Int16, multithread=True).ReadAsArray()
ds = None
print("fin warp permea" + str(rasterPermea.shape))

ds = gdal.Open(fichierRasterOccSol)
print("début warp occsol")
rasterOccSol = gdal.Warp('occsolrepr', ds, outputBounds=[xmin, ymin, xmax, ymax], xRes=resX, yRes=resY, resampleAlg='bilinear', outputType=gdal.GDT_Int16, multithread=True).ReadAsArray()
ds = None
print("fin warp occsol")

ds = gdal.Open(fichierRasterBattance)
print("début warp battance")
rasterBattance = gdal.Warp('battrepr', ds, outputBounds=[xmin, ymin, xmax, ymax], xRes=resX, yRes=resY, resampleAlg='bilinear', outputType=gdal.GDT_Int16, multithread=True).ReadAsArray()
ds = None
print("fin warp battance")

ds = gdal.Open(fichierRasterPentes)
print("début warp pentes")
rasterPentes = gdal.Warp('penterepr', ds, outputBounds=[xmin, ymin, xmax, ymax], xRes=resX, yRes=resY, resampleAlg='bilinear', outputType=gdal.GDT_Int16, multithread=True).ReadAsArray()
ds = None
print("fin warp pentes")
'''


rasterOccSol_5a = gdal.Open("occsolrepr_5a").ReadAsArray()
rasterOccSol_2021 = gdal.Open("occsolrepr_2021").ReadAsArray()
rasterOccSol_modif = gdal.Open("occsolrepr_modif").ReadAsArray()


(dim1,dim2) = rasterPentes.shape


# Host code   
dataPentes = rasterPentes.reshape(dim1*dim2)
dataPermea = rasterPermea.reshape(dim1*dim2)
dataOccSol_5a = rasterOccSol_5a.reshape(dim1*dim2)
dataOccSol_2021 = rasterOccSol_2021.reshape(dim1*dim2)
dataOccSol_modif = rasterOccSol_modif.reshape(dim1*dim2)
dataBattance = rasterBattance.reshape(dim1*dim2)
dimtot = dim1*dim2

# _5a ###########################################################################
if dimtot > 1000000000:
    nb = dimtot // 1000000000
    dataPentes_lst = np.array_split(dataPentes, nb)
    dataPermea_lst = np.array_split(dataPermea, nb)
    dataOccSol_lst = np.array_split(dataOccSol_5a, nb)
    dataBattance_lst = np.array_split(dataBattance, nb)
    raster_result_lst = []
    raster_diff_lst = []
    for i in range(nb):
        dataPentes_p = dataPentes_lst[i]
        dataPermea_p = dataPermea_lst[i]
        dataOccSol_p = dataOccSol_lst[i]
        dataBattance_p = dataBattance_lst[i]
        
        dataResult_p = dataOccSol_p.copy()
        dataDiff_p = dataOccSol_p.copy()
        data_Ssbatt_p = dataResult_p.copy()
        debut = datetime.datetime.now()
        threadsperblock = 1024
        blockspergrid = math.ceil(dataResult_p.shape[0] / threadsperblock)
        my_kernel[blockspergrid, threadsperblock](dataPentes_p,dataPermea_p,dataOccSol_p,dataBattance_p,dataResult_p,data_Ssbatt_p,dataDiff_p,tabCorCod,tabCoefBat)
        print("...Traitement GPU terminé en : ", datetime.datetime.now() - debut)
        
        raster_result_lst.append(dataResult_p)
        raster_diff_lst.append(dataDiff_p)
    raster_result_p = np.concatenate(tuple(raster_result_lst))
    raster_diff_p = np.concatenate(tuple(raster_diff_lst))
else:
    dataResult = dataOccSol_5a.copy()
    dataDiff = dataOccSol_5a.copy()
    data_Ssbatt = dataResult.copy()
    debut = datetime.datetime.now()
    threadsperblock = 1024
    blockspergrid = math.ceil(dataResult.shape[0] / threadsperblock)
    my_kernel[blockspergrid, threadsperblock](dataPentes,dataPermea,dataOccSol_5a,dataBattance,dataResult,data_Ssbatt,dataDiff,tabCorCod,tabCoefBat)
    print("...Traitement GPU terminé en : ", datetime.datetime.now() - debut)
    raster_result_p = dataResult.reshape(dim1,dim2)
    raster_diff_p = dataDiff.reshape(dim1,dim2)
    
raster_result = raster_result_p.reshape(dim1,dim2)
raster_diff = raster_diff_p.reshape(dim1,dim2)


# Ecriture du fichier de sortie
debut = datetime.datetime.now()
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(fichierSortieCRUS_battance + "5a.tif", dim2, dim1, 1, gdal.GDT_Int16)
ds = gdal.Open("permearepr")
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
outdiff = driver.Create(fichierSortieCRUS_DiffsansBattance + "5a.tif", dim2, dim1, 1, gdal.GDT_Int16)
outdiff.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
outdiff.SetProjection(ds.GetProjection())##sets same projection as input
outdiff.GetRasterBand(1).WriteArray(raster_diff)
outdiff.GetRasterBand(1).SetNoDataValue(-1)##if you want these values transparent
outdiff.FlushCache() ##saves to disk!!
outdiff = None
ds=None

outdata = None
ds=None

# _2021 ###########################################################################
if dimtot > 1000000000:
    nb = dimtot // 1000000000
    dataPentes_lst = np.array_split(dataPentes, nb)
    dataPermea_lst = np.array_split(dataPermea, nb)
    dataOccSol_lst = np.array_split(dataOccSol_2021, nb)
    dataBattance_lst = np.array_split(dataBattance, nb)
    raster_result_lst = []
    raster_diff_lst = []
    for i in range(nb):
        dataPentes_p = dataPentes_lst[i]
        dataPermea_p = dataPermea_lst[i]
        dataOccSol_p = dataOccSol_lst[i]
        dataBattance_p = dataBattance_lst[i]
        
        dataResult_p = dataOccSol_p.copy()
        dataDiff_p = dataOccSol_p.copy()
        data_Ssbatt_p = dataResult_p.copy()
        debut = datetime.datetime.now()
        threadsperblock = 1024
        blockspergrid = math.ceil(dataResult_p.shape[0] / threadsperblock)
        my_kernel[blockspergrid, threadsperblock](dataPentes_p,dataPermea_p,dataOccSol_p,dataBattance_p,dataResult_p,data_Ssbatt_p,dataDiff_p,tabCorCod,tabCoefBat)
        print("...Traitement GPU terminé en : ", datetime.datetime.now() - debut)
        
        raster_result_lst.append(dataResult_p)
        raster_diff_lst.append(dataDiff_p)
    raster_result_p = np.concatenate(tuple(raster_result_lst))
    raster_diff_p = np.concatenate(tuple(raster_diff_lst))
else:
    dataResult = dataOccSol_2021.copy()
    dataDiff = dataOccSol_2021.copy()
    data_Ssbatt = dataResult.copy()
    debut = datetime.datetime.now()
    threadsperblock = 1024
    blockspergrid = math.ceil(dataResult.shape[0] / threadsperblock)
    my_kernel[blockspergrid, threadsperblock](dataPentes,dataPermea,dataOccSol_2021,dataBattance,dataResult,data_Ssbatt,dataDiff,tabCorCod,tabCoefBat)
    print("...Traitement GPU terminé en : ", datetime.datetime.now() - debut)
    raster_result_p = dataResult.reshape(dim1,dim2)
    raster_diff_p = dataDiff.reshape(dim1,dim2)
    
raster_result = raster_result_p.reshape(dim1,dim2)
raster_diff = raster_diff_p.reshape(dim1,dim2)


# Ecriture du fichier de sortie
debut = datetime.datetime.now()
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(fichierSortieCRUS_battance + "2021.tif", dim2, dim1, 1, gdal.GDT_Int16)
ds = gdal.Open("permearepr")
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
outdiff = driver.Create(fichierSortieCRUS_DiffsansBattance + "2021.tif", dim2, dim1, 1, gdal.GDT_Int16)
outdiff.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
outdiff.SetProjection(ds.GetProjection())##sets same projection as input
outdiff.GetRasterBand(1).WriteArray(raster_diff)
outdiff.GetRasterBand(1).SetNoDataValue(-1)##if you want these values transparent
outdiff.FlushCache() ##saves to disk!!
outdiff = None
ds=None

outdata = None
ds=None

# _modif ###########################################################################
if dimtot > 1000000000:
    nb = dimtot // 1000000000
    dataPentes_lst = np.array_split(dataPentes, nb)
    dataPermea_lst = np.array_split(dataPermea, nb)
    dataOccSol_lst = np.array_split(dataOccSol_modif, nb)
    dataBattance_lst = np.array_split(dataBattance, nb)
    raster_result_lst = []
    raster_diff_lst = []
    for i in range(nb):
        dataPentes_p = dataPentes_lst[i]
        dataPermea_p = dataPermea_lst[i]
        dataOccSol_p = dataOccSol_lst[i]
        dataBattance_p = dataBattance_lst[i]
        
        dataResult_p = dataOccSol_p.copy()
        dataDiff_p = dataOccSol_p.copy()
        data_Ssbatt_p = dataResult_p.copy()
        debut = datetime.datetime.now()
        threadsperblock = 1024
        blockspergrid = math.ceil(dataResult_p.shape[0] / threadsperblock)
        my_kernel[blockspergrid, threadsperblock](dataPentes_p,dataPermea_p,dataOccSol_p,dataBattance_p,dataResult_p,data_Ssbatt_p,dataDiff_p,tabCorCod,tabCoefBat)
        print("...Traitement GPU terminé en : ", datetime.datetime.now() - debut)
        
        raster_result_lst.append(dataResult_p)
        raster_diff_lst.append(dataDiff_p)
    raster_result_p = np.concatenate(tuple(raster_result_lst))
    raster_diff_p = np.concatenate(tuple(raster_diff_lst))
else:
    dataResult = dataOccSol_modif.copy()
    dataDiff = dataOccSol_modif.copy()
    data_Ssbatt = dataResult.copy()
    debut = datetime.datetime.now()
    threadsperblock = 1024
    blockspergrid = math.ceil(dataResult.shape[0] / threadsperblock)
    my_kernel[blockspergrid, threadsperblock](dataPentes,dataPermea,dataOccSol_modif,dataBattance,dataResult,data_Ssbatt,dataDiff,tabCorCod,tabCoefBat)
    print("...Traitement GPU terminé en : ", datetime.datetime.now() - debut)
    raster_result_p = dataResult.reshape(dim1,dim2)
    raster_diff_p = dataDiff.reshape(dim1,dim2)
    
raster_result = raster_result_p.reshape(dim1,dim2)
raster_diff = raster_diff_p.reshape(dim1,dim2)


# Ecriture du fichier de sortie
debut = datetime.datetime.now()
driver = gdal.GetDriverByName("GTiff")
outdata = driver.Create(fichierSortieCRUS_battance + "modif.tif", dim2, dim1, 1, gdal.GDT_Int16)
ds = gdal.Open("permearepr")
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
outdiff = driver.Create(fichierSortieCRUS_DiffsansBattance + "modif.tif", dim2, dim1, 1, gdal.GDT_Int16)
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

















