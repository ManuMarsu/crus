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
    print("début warp", raster_warped, "\n")
    raster = gdal.Warp(raster_warped, ds, outputBounds=outputBounds, xRes=resX, yRes=resY, resampleAlg='bilinear', outputType=gdal.GDT_Int16, multithread=True)
    ds = None
    print("fin warp", raster_warped, "\n")
    
    

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
fichierRasterOccSol = "os_modif-vf.tif"
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
        "occsol": lire_raster(fichierRasterOccSol, "occsol"), 
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

t1a = threading.Thread(target=warp_raster, args=(fichierRasterPermea, "permearepr_a", [xmin, ymin, xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2], resX * 2, resY * 2, ))
t1b = threading.Thread(target=warp_raster, args=(fichierRasterPermea, "permearepr_b", [xmin + (xmax - xmin) / 2, ymin, xmax, (ymax - ymin) / 2], resX * 2, resY * 2, ))
t1c = threading.Thread(target=warp_raster, args=(fichierRasterPermea, "permearepr_c", [xmin, ymin + (ymax - ymin) / 2, xmin + (xmax - xmin) / 2, ymax], resX * 2, resY * 2, ))
t1d = threading.Thread(target=warp_raster, args=(fichierRasterPermea, "permearepr_d", [xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2, xmax, ymax], resX * 2, resY * 2, ))

t2a = threading.Thread(target=warp_raster, args=(fichierRasterOccSol, "occsolrepr_a", [xmin, ymin, xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2], resX * 2, resY * 2, ))
t2b = threading.Thread(target=warp_raster, args=(fichierRasterOccSol, "occsolrepr_b", [xmin + (xmax - xmin) / 2, ymin, xmax, (ymax - ymin) / 2], resX * 2, resY * 2, ))
t2c = threading.Thread(target=warp_raster, args=(fichierRasterOccSol, "occsolrepr_c", [xmin, ymin + (ymax - ymin) / 2, xmin + (xmax - xmin) / 2, ymax], resX * 2, resY * 2, ))
t2d = threading.Thread(target=warp_raster, args=(fichierRasterOccSol, "occsolrepr_d", [xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2, xmax, ymax], resX * 2, resY * 2, ))

t3a = threading.Thread(target=warp_raster, args=(fichierRasterBattance, "battrepr_a", [xmin, ymin, xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2], resX * 2, resY * 2, ))
t3b = threading.Thread(target=warp_raster, args=(fichierRasterBattance, "battrepr_b", [xmin + (xmax - xmin) / 2, ymin, xmax, (ymax - ymin) / 2], resX * 2, resY * 2, ))
t3c = threading.Thread(target=warp_raster, args=(fichierRasterBattance, "battrepr_c", [xmin, ymin + (ymax - ymin) / 2, xmin + (xmax - xmin) / 2, ymax], resX * 2, resY * 2, ))
t3d = threading.Thread(target=warp_raster, args=(fichierRasterBattance, "battrepr_d", [xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2, xmax, ymax], resX * 2, resY * 2, ))

t4a = threading.Thread(target=warp_raster, args=(fichierRasterPentes, "penterepr_a", [xmin, ymin, xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2], resX * 2, resY * 2, ))
t4b = threading.Thread(target=warp_raster, args=(fichierRasterPentes, "penterepr_b", [xmin + (xmax - xmin) / 2, ymin, xmax, (ymax - ymin) / 2], resX * 2, resY * 2, ))
t4c = threading.Thread(target=warp_raster, args=(fichierRasterPentes, "penterepr_c", [xmin, ymin + (ymax - ymin) / 2, xmin + (xmax - xmin) / 2, ymax], resX * 2, resY * 2, ))
t4d = threading.Thread(target=warp_raster, args=(fichierRasterPentes, "penterepr_d", [xmin + (xmax - xmin) / 2, ymin + (ymax - ymin) / 2, xmax, ymax], resX * 2, resY * 2, ))

t1a.start()
t1b.start()
t1c.start()
t1d.start()

t2a.start()
t2b.start()
t2c.start()
t2d.start()

t3a.start()
t3b.start()
t3c.start()
t3d.start()

t4a.start()
t4b.start()
t4c.start()
t4d.start()

t1a.join()
t1b.join()
t1c.join()
t1d.join()
	
t2a.join()
t2b.join()
t2c.join()
t2d.join()
	
t3a.join()
t3b.join()
t3c.join()
t3d.join()
	
t4a.join()
t4b.join()
t4c.join()
t4d.join()


for it in ["a", "b", "c", "d"]:
        
    
    rasterPermea = gdal.Open(f"permearepr_{it}").ReadAsArray()
    rasterOccSol = gdal.Open(f"occsolrepr_{it}").ReadAsArray()
    rasterBattance = gdal.Open(f"battrepr_{it}").ReadAsArray()
    rasterPentes = gdal.Open(f"penterepr_{it}").ReadAsArray()
    
    
    (dim1,dim2) = rasterPentes.shape
    
    
    # Host code   
    dataPentes = rasterPentes.reshape(dim1*dim2)
    dataPermea = rasterPermea.reshape(dim1*dim2)
    dataOccSol = rasterOccSol.reshape(dim1*dim2)
    dataBattance = rasterBattance.reshape(dim1*dim2)
    dimtot = dim1*dim2

    '''
    if dimtot > 1000000000:
        nb = dimtot // 1000000000
        dataPentes_lst = np.array_split(dataPentes, nb)
        dataPermea_lst = np.array_split(dataPermea, nb)
        dataOccSol_lst = np.array_split(dataOccSol, nb)
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
        '''
    dataResult = dataOccSol.copy()
    dataDiff = dataOccSol.copy()
    data_Ssbatt = dataResult.copy()
    debut = datetime.datetime.now()
    threadsperblock = 1024
    blockspergrid = math.ceil(dataResult.shape[0] / threadsperblock)
    my_kernel[blockspergrid, threadsperblock](dataPentes,dataPermea,dataOccSol,dataBattance,dataResult,data_Ssbatt,dataDiff,tabCorCod,tabCoefBat)
    print("...Traitement GPU terminé en : ", datetime.datetime.now() - debut)
    raster_result_p = dataResult.reshape(dim1,dim2)
    raster_diff_p = dataDiff.reshape(dim1,dim2)
    
    #Retour de l'indentation
    #--------------------
    
    raster_result = raster_result_p.reshape(dim1,dim2)
    raster_diff = raster_diff_p.reshape(dim1,dim2)


    # Ecriture du fichier de sortie
    debut = datetime.datetime.now()
    driver = gdal.GetDriverByName("GTiff")
    outdata = driver.Create(fichierSortieCRUS_battance + f"_{it}.tif", dim2, dim1, 1, gdal.GDT_Int16)
    ds = gdal.Open(f"permearepr_{it}")
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
    outdiff = driver.Create(fichierSortieCRUS_DiffsansBattance + f"_{it}.tif", dim2, dim1, 1, gdal.GDT_Int16)
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

















