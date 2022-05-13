# -*- coding: utf-8 -*-
"""
Created on Thu May  5 11:31:11 2022

@author: manuel.collongues
Cerema / Laboratoire de Nancy / ERTD

Prolongation de la méthode CRUS du Cerema pour déterminer la partie "solide"
des écoulements, réalisé notamment grâce aux études suivantes :
    - perte de sol Canada : http://www.omafra.gov.on.ca/french/engineer/facts/12-052.htm#:~:text=A%20%3D%20R%20x%20K%20x,%C2%AB%20pertes%20en%20terre%20tol%C3%A9rables%20%C2%BB
    - Estimation du risque d'érosion en Italie (cf pdf biblio/eritalie.pdf'

Ce script permet de calculer la sensibilité à l'érodibilité d'un territoire à partir des données d'entrée suivantes :
    - LS.tif            : combinaison de le facteur de longueur de pente (S) et de l'indice de pente (L)
    - pente.tif         : en degrés
    - R.tif             : facteur d'érosivité des précipitations'
    - taux_argile.tif   : taux d'argile calculé à partir des données pédologiques
    - taux_limon.tif    : taux de limon calculé à partir des données pédologiques
    - taux_sable.tif    : taux de sable calculé à partir des données pédologiques
    
Ce script suppose de fournir au format gtif chacun de ces raster, avec exactement les mêmes bornes et résolution spatiale
Il calcule 2 rasters :
    - un (A) issu de la formule perte de terre (A = R * K * LS * C) (résultats en tonnes / ha)
    - un qui convertit ce résultat (A) pour des rasters au pas de 5 mètres (résultats en kg / pixel)
    - un qui convertit le résultat (A) en épaisseur de sol érodée (résultats en mètre)
    
ATTENTION : dans le calcul de LS, l'hypothèse est prise que les pixels des rasters sont de taille 5m * 5m
"""


from __future__ import division
import numpy as np
from numba import cuda
import math
import datetime
from osgeo import gdal


# CUDA kernel
@cuda.jit
def my_kernel(R, K, tx_argile, tx_limon, tx_sable, pente, L, S, C, occ_sol, result_A, result_kg_px, result_epaisseur, occ_sol_coeff_c):
    pos = cuda.grid(1)
    # Correspondance entre coefficient C et occupation du sol
    for i in range(occ_sol_coeff_c.shape[0]):
        if occ_sol[pos] == occ_sol_coeff_c[i, 0]:
            C[pos] = occ_sol_coeff_c[i, 1]
            
    # Calcul des coefficients L, S
    L[pos] = 1.4 * math.pow((5/22.13), 0.4)
    S[pos] = math.pow((math.sin(pente[pos]) / 0.0896), 1.3)
    
    # Correspondance entre coefficient K et taux d'argile, limon et sable
    if tx_argile[pos] < 18 and tx_sable[pos] > 65:
        K[pos] = 0.0115
    elif tx_argile[pos] > 18 and tx_argile[pos] < 35 and tx_sable[pos] > 65:
        K[pos] = 0.0311
    elif tx_argile[pos] < 35 and tx_sable[pos] < 15:
        K[pos] = 0.0438
    elif tx_argile[pos] > 35 and tx_argile[pos] < 60:
        K[pos] = 0.0339
    elif tx_argile[pos] > 60:
        K[pos] = 0.0170
        
    result_A[pos] = R[pos] * K[pos] * C[pos] * L[pos] * S[pos]
    result_kg_px[pos] = result_A[pos] * 0.0025 * 1000
    result_epaisseur[pos] = (result_kg_px[pos] / 1250) / 25
    



#####################################################################################################################
# Fonctions
#####################################################################################################################

def lecture_raster_tif(nom_fichier):
    message = ""
    debut = datetime.datetime.now()
    ds = gdal.Open(nom_fichier)
    raster = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype=np.float32)
    (dim1,dim2) = raster.shape
    data_raster = raster.reshape(dim1 * dim2)
    message += "...Lecture " + nom_fichier + " terminée en : " + str(datetime.datetime.now() - debut) + " - Dimensions : " + str(dim1) + ", " + str(dim2)
    return ds, dim1, dim2, data_raster, message

def ecriture_raster_sortie(nom_fichier, donnee, type_donnee, ds, dim1, dim2):
    driver = gdal.GetDriverByName("GTiff")
    outdiff = driver.Create(nom_fichier, dim2, dim1, 1, type_donnee)
    outdiff.SetGeoTransform(ds.GetGeoTransform())##sets same geotransform as input
    outdiff.SetProjection(ds.GetProjection())##sets same projection as input
    outdiff.GetRasterBand(1).WriteArray(donnee)
    outdiff.GetRasterBand(1).SetNoDataValue(-1)##if you want these values transparent
    outdiff.FlushCache() ##saves to disk!!
    outdiff = None
    return None


#####################################################################################################################
# Execution principale
#####################################################################################################################
def main():
    
    #####################################################################################################################
    # Paramètres en entrée et sortie
    #####################################################################################################################
    # pente.tif
            
    fichier_R = "R.tif"
    fichier_pente = "pente degres.tif"
    fichier_tx_argile = "argile.tif"
    fichier_tx_limon = "limon.tif"
    fichier_tx_sable = "sable.tif"
    fichier_occ_sol = "occup terri.tif"
    
    sortie_A = "erodibilite_tonne_ha.tif"
    sortie_P = "erodibilite_kg_pixel.tif"
    sortie_E = "erodibilite_epaisseur.tif"
    
    # Lecture tableau de correspondance du coefficient C (constitué à partir de la note CRUS d'occupation du sol)
    t0 = datetime.datetime.now()
    tableauCorrespondanceCodes=[]
    debut = datetime.datetime.now()
    with open ("correspondance_coefC_occ_sol.txt", "r", encoding = "utf-8") as mat:
        for ligne in mat:
            lignee = ligne.split("\n")[0].split("=")
            code_occ_sol = float(lignee[0])
            coef_c = float(lignee[1])
            tableauCorrespondanceCodes.append([code_occ_sol, coef_c])
        mat.close()
    occ_sol_coeff_c = cuda.to_device(np.ascontiguousarray(tableauCorrespondanceCodes, dtype=np.float64))
    print("...Lecture tableau de correspondance coefficient C / Occupation du sol (CRUS) terminée en : ", datetime.datetime.now() - debut)
    
    # Lecture des rasters d'entrée
    ds, dim1, dim2, h_R, message = lecture_raster_tif(fichier_R)
    print(message)
    ds, dim1, dim2, h_tx_argile, message = lecture_raster_tif(fichier_tx_argile)
    print(message)
    ds, dim1, dim2, h_tx_limon, message = lecture_raster_tif(fichier_tx_limon)
    print(message)
    ds, dim1, dim2, h_tx_sable, message = lecture_raster_tif(fichier_tx_sable)
    print(message)
    ds, dim1, dim2, h_occ_sol, message = lecture_raster_tif(fichier_occ_sol)
    print(message)
    ds, dim1, dim2, h_pente, message = lecture_raster_tif(fichier_pente)
    print(message)
    
    # Host code  
    
    pente = cuda.to_device(np.ascontiguousarray(h_pente, dtype = np.float64))
    R = cuda.to_device(np.ascontiguousarray(h_R, dtype = np.int16))
    tx_argile = cuda.to_device(np.ascontiguousarray(h_tx_argile, dtype = np.float32))
    tx_limon = cuda.to_device(np.ascontiguousarray(h_tx_limon, dtype = np.float32))
    tx_sable = cuda.to_device(np.ascontiguousarray(h_tx_sable, dtype = np.float32))
    occ_sol = cuda.to_device(np.ascontiguousarray(h_occ_sol, dtype = np.float32))
    
    C = cuda.device_array_like(occ_sol)
    K = cuda.device_array_like(occ_sol)
    L = cuda.device_array_like(occ_sol)
    S = cuda.device_array_like(occ_sol)
    result_A = cuda.device_array_like(occ_sol)
    result_epaisseur = cuda.device_array_like(occ_sol)
    result_kg_px = cuda.device_array_like(occ_sol)
    
    #####################################################################################################################
    # Lancement du traitement GPU
    #####################################################################################################################
    debut = datetime.datetime.now()
    threadsperblock = 1024
    blockspergrid = math.ceil(result_A.shape[0] / threadsperblock)
    my_kernel[blockspergrid, threadsperblock](R, K, tx_argile, tx_limon, tx_sable, pente, L, S, C, occ_sol, result_A, result_kg_px, result_epaisseur, occ_sol_coeff_c)
    print("...Traitement GPU terminé en : ", datetime.datetime.now() - debut)
    
    h_result_A = np.empty(shape=result_A.shape, dtype=result_A.dtype)
    h_result_kg_px = np.empty(shape=result_kg_px.shape, dtype=result_kg_px.dtype)
    h_result_epaisseur = np.empty(shape=result_epaisseur.shape, dtype=result_epaisseur.dtype)
    result_A.copy_to_host(h_result_A)
    result_kg_px.copy_to_host(h_result_kg_px)
    result_epaisseur.copy_to_host(h_result_epaisseur)
    
    raster_result_A = h_result_A.reshape(dim1,dim2)
    raster_result_kg_px = h_result_kg_px.reshape(dim1,dim2)
    raster_result_epaisseur = h_result_epaisseur.reshape(dim1,dim2)
    
    #####################################################################################################################
    # Ecriture des fichiers de sortie
    #####################################################################################################################
    debut = datetime.datetime.now()
    res = ecriture_raster_sortie(sortie_A, raster_result_A, gdal.GDT_Float64, ds, dim1, dim2)
    res = ecriture_raster_sortie(sortie_P, raster_result_kg_px, gdal.GDT_Float64, ds, dim1, dim2)
    res = ecriture_raster_sortie(sortie_E, raster_result_epaisseur, gdal.GDT_Float64, ds, dim1, dim2)
    ds = None
    
    
    print("...Ecriture fichiers sortie terminée en : ", datetime.datetime.now() - debut)
    print("...Temps total de traitement : ", datetime.datetime.now() - t0)


if __name__ == '__main__':
    main()


















