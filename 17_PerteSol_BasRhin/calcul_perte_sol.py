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
import threading
from osgeo import gdal

def warp_raster(fichierRaster, raster_warped, outputBounds, resX, resY, fichier_log):
    ds = gdal.Open(fichierRaster)
    log(fichier_log, f"{fichierRaster} - Début warp")
    raster = gdal.Warp(raster_warped, ds, outputBounds=outputBounds, xRes=resX, yRes=resY, resampleAlg='bilinear', outputType=gdal.GDT_Int16, multithread=True)
    ds = None
    log(fichier_log, f"{fichierRaster} - Fin warp")

def lire_raster(fichier, fichier_log):
    debut = datetime.datetime.now()
    ds = gdal.Open(fichier)
    geotransform = ds.GetGeoTransform()
    projection = ds.GetProjection()
    xmin = geotransform[0]
    ymax = geotransform[3]
    xmax = geotransform[0] + (geotransform[1] * ds.RasterXSize)
    ymin = geotransform[3] + (geotransform[5] * ds.RasterYSize)
    resX = geotransform[1]
    resY = -geotransform[5]
    log(fichier_log, f"...Lecture {fichier} terminée en : {datetime.datetime.now() - debut}")
    return {"ds":ds, "geotransform":geotransform, "projection":projection, "xmin":xmin, "ymax":ymax, "xmax":xmax, "ymin":ymin, "resX":resX, "resY":resY}

def log(fichier, texte):
    with open(fichier, "a", encoding="utf-8") as f:
        f.write(texte + "\n")
    print(texte)

# CUDA kernel
@cuda.jit
def my_kernel(R, K, tx_argile, tx_limon, tx_sable, pente, LS, C, occ_sol, result_A, result_kg_px, result_epaisseur, occ_sol_coeff_c, ndv):
    pos = cuda.grid(1)

    # Calcul du coefficient C à partir de l'occupation du sol
    for i in range(occ_sol_coeff_c.shape[0]):
        if occ_sol[pos] == occ_sol_coeff_c[i, 0]:
            C[pos] = occ_sol_coeff_c[i, 1]
            
    # Calcul des coefficients L, S
    # LS[pos] = 1.4 * math.pow((5/22.13), 0.4) (ancienne formule)
    # S[pos] = math.pow((math.sin(pente[pos]) / 0.0896), 1.3) (ancienne formule)
    LS[pos] = (0.065 + 0.0456 * pente[pos] + 0.006541 * pente[pos] * pente[pos]) * 1.063589 # 1,063589 = (25/22,1)^0,5
    
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
        
    # Calcul des résultats
    result_A[pos] = R[pos] * K[pos] * C[pos] * LS[pos]
    result_kg_px[pos] = result_A[pos] * 0.0025 * 1000
    result_epaisseur[pos] = pente[pos]

    # Si le pixel est un NoDataValue, on le remplace par un NoDataValue
    if pente[pos] == ndv:
        LS[pos] = -1
        pente[pos] = -1
        result_A[pos] = -1
        result_kg_px[pos] = -1
        result_epaisseur[pos] = -1
    
def lance_calcul(h_pente, h_R, h_tx_argile, h_tx_limon, h_tx_sable, h_occ_sol, occ_sol_coeff_c, ndv):
    # Host code  
    pente = cuda.to_device(np.ascontiguousarray(h_pente, dtype = np.float32))
    R = cuda.to_device(np.ascontiguousarray(h_R, dtype = np.int16))
    tx_argile = cuda.to_device(np.ascontiguousarray(h_tx_argile, dtype = np.float32))
    tx_limon = cuda.to_device(np.ascontiguousarray(h_tx_limon, dtype = np.float32))
    tx_sable = cuda.to_device(np.ascontiguousarray(h_tx_sable, dtype = np.float32))
    occ_sol = cuda.to_device(np.ascontiguousarray(h_occ_sol, dtype = np.float32))
    
    C = cuda.device_array_like(occ_sol)
    K = cuda.device_array_like(occ_sol)
    LS = cuda.device_array_like(occ_sol)
    result_A = cuda.device_array_like(occ_sol)
    result_epaisseur = cuda.device_array_like(occ_sol)
    result_kg_px = cuda.device_array_like(occ_sol)
    
    #####################################################################################################################
    # Lancement du traitement GPU
    #####################################################################################################################
    debut = datetime.datetime.now()
    threadsperblock = 1024
    blockspergrid = math.ceil(result_A.shape[0] / threadsperblock)
    my_kernel[blockspergrid, threadsperblock](R, K, tx_argile, tx_limon, tx_sable, pente, LS, C, occ_sol, result_A, result_kg_px, result_epaisseur, occ_sol_coeff_c, ndv)
    print("...Traitement GPU terminé en : ", datetime.datetime.now() - debut)
    
    h_result_A = np.empty(shape=result_A.shape, dtype=result_A.dtype)
    h_result_kg_px = np.empty(shape=result_kg_px.shape, dtype=result_kg_px.dtype)
    h_result_epaisseur = np.empty(shape=result_epaisseur.shape, dtype=result_epaisseur.dtype)
    result_A.copy_to_host(h_result_A)
    result_kg_px.copy_to_host(h_result_kg_px)
    result_epaisseur.copy_to_host(h_result_epaisseur)
    return h_result_A, h_result_kg_px, h_result_epaisseur

#####################################################################################################################
# Fonctions
#####################################################################################################################

def lecture_raster_tif(nom_fichier):
    message = ""
    debut = datetime.datetime.now()
    ds = gdal.Open(nom_fichier)
    raster = np.array(ds.GetRasterBand(1).ReadAsArray(), dtype=np.float32)
    ndv = ds.GetRasterBand(1).GetNoDataValue()
    (dim1,dim2) = raster.shape
    data_raster = raster.reshape(dim1 * dim2)
    message += "...Lecture " + nom_fichier + " terminée en : " + str(datetime.datetime.now() - debut) + " - Dimensions : " + str(dim1) + ", " + str(dim2)
    return ds, dim1, dim2, ndv, data_raster, message

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
    # Création d'un fichier log horodaté pour chaque nouvelle exécution
    fichier_log = "log_" + datetime.datetime.now().strftime("%Y%m%d_%H%M%S") + ".txt"
    with open(fichier_log, "w", encoding="utf-8") as f:
        f.write("**************************************************")
        f.write(f"********* Début du traitement le {datetime.datetime.now()} *********")
        f.write("**************************************************")
        f.write("")
    
    #####################################################################################################################
    # Paramètres en entrée et sortie
    #####################################################################################################################
    fichier_R = "R.tif"
    fichier_pente = "pente degres.tif"
    fichier_tx_argile = "argile.tif"
    fichier_tx_limon = "limon.tif"
    fichier_tx_sable = "sable.tif"
    fichier_occ_sol = "occupation_sol.tif"
    
    sortie_A = "erodibilite_tonne_ha.tif"
    sortie_P = "erodibilite_kg_pixel.tif"
    sortie_E = "erodibilite_epaisseur.tif"

    #####################################################################################################################
    # Lecture fichiers sources
    #####################################################################################################################
    
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


    #####################################################################################################################
    # Découpage selon l'emprise minimale
    #####################################################################################################################
    if True:
        dico = {"R": lire_raster(fichier_R, fichier_log),
                "occsol": lire_raster(fichier_pente, fichier_log), 
                "permea": lire_raster(fichier_tx_argile, fichier_log), 
                "occsol": lire_raster(fichier_tx_limon, fichier_log), 
                "occsol": lire_raster(fichier_tx_sable, fichier_log), 
                "occsol": lire_raster(fichier_occ_sol, fichier_log)}

        # Définir les dimensions de l'emprise minimale pour tous les rasters
        xmin = max([ds["xmin"] for ds in dico.values()])
        ymin = max([ds["ymin"] for ds in dico.values()])
        xmax = min([ds["xmax"] for ds in dico.values()])
        ymax = min([ds["ymax"] for ds in dico.values()])

        # Définir la résolution souhaitée pour tous les rasters
        resX = max([ds["resX"] for ds in dico.values()])
        resY = max([ds["resY"] for ds in dico.values()])

        condition_non_decoupage = (xmin == min([ds["xmin"] for ds in dico.values()]) 
                                and ymin == min([ds["ymin"] for ds in dico.values()]) 
                                and xmax == max([ds["xmax"] for ds in dico.values()]) 
                                and ymax == max([ds["ymax"] for ds in dico.values()]) 
                                and resX == max([ds["resX"] for ds in dico.values()]) 
                                and resY == max([ds["resY"] for ds in dico.values()]))

        if not condition_non_decoupage:
            # Découper chaque raster pour qu'il corresponde à l'emprise et la résolution souhaitées
            t1 = threading.Thread(target=warp_raster, args=(fichier_R, "fichier_raster_R.tif", [xmin, ymin, xmax, ymax], resX, resY, ))
            t2a = threading.Thread(target=warp_raster, args=(fichier_pente, "fichier_raster_pente.tif", [xmin, ymin, xmax, ymax], resX, resY, ))
            t2b = threading.Thread(target=warp_raster, args=(fichier_tx_argile, "fichier_raster_tx_argile.tif", [xmin, ymin, xmax, ymax], resX, resY, ))
            t2c = threading.Thread(target=warp_raster, args=(fichier_tx_limon, "fichier_raster_tx_limon.tif" [xmin, ymin, xmax, ymax], resX, resY, ))
            t3 = threading.Thread(target=warp_raster, args=(fichier_tx_sable, "fichier_raster_tx_sable.tif", [xmin, ymin, xmax, ymax], resX, resY, ))
            t4 = threading.Thread(target=warp_raster, args=(fichier_occ_sol, "fichier_raster_occ_sol.tif", [xmin, ymin, xmax, ymax], resX, resY, ))

            t1.start()
            t2a.start()
            t2b.start()
            t2c.start()
            t3.start()
            t4.start()

            t1.join()
            log(fichier_log, f"...Fin du découpage R - temps total = {datetime.datetime.now() - t0}")
            t2a.join()
            log(fichier_log, f"...Fin du découpage pente - temps total = {datetime.datetime.now() - t0}")
            t2b.join()
            log(fichier_log, f"...Fin du découpage tx_argile - temps total = {datetime.datetime.now() - t0}")
            t2c.join()
            log(fichier_log, f"...Fin du découpage tx_limon - temps total = {datetime.datetime.now() - t0}")
            t3.join()
            log(fichier_log, f"...Fin du découpage tx_sable - temps total = {datetime.datetime.now() - t0}")
            t4.join()
            log(fichier_log, f"...Fin du découpage occ_sol - temps total = {datetime.datetime.now() - t0}")
        
            fichier_R = "fichier_raster_R.tif"
            fichier_pente = "fichier_raster_pente.tif"
            fichier_tx_argile = "fichier_raster_tx_argile.tif"
            fichier_tx_limon = "fichier_raster_tx_limon.tif"
            fichier_tx_sable = "fichier_raster_tx_sable.tif"
            fichier_occ_sol = "fichier_raster_occ_sol.tif"

    #####################################################################################################################
    # Préparation des données et Host code
    #####################################################################################################################
    # Lecture des rasters d'entrée et reshape intégré à la fonction lecture_raster_tif
    ds, dim1, dim2, ndv, h_R, message = lecture_raster_tif(fichier_R)
    print(message)
    ds, dim1, dim2, ndv, h_pente, message = lecture_raster_tif(fichier_pente)
    print(message)
    ds, dim1, dim2, ndv, h_tx_argile, message = lecture_raster_tif(fichier_tx_argile)
    print(message)
    ds, dim1, dim2, ndv, h_tx_limon, message = lecture_raster_tif(fichier_tx_limon)
    print(message)
    ds, dim1, dim2, ndv, h_tx_sable, message = lecture_raster_tif(fichier_tx_sable)
    print(message)
    ds, dim1, dim2, ndv, h_occ_sol, message = lecture_raster_tif(fichier_occ_sol)
    print(message)

    # Test de la taille des rasters pour partitionner le traitement si nécessaire
    dimtot = dim1*dim2
    dimmax = 10000000
    if dimtot > dimmax:
        nb = dimtot // dimmax
        dataR_lst = np.array_split(h_R, nb)
        dataPente_lst = np.array_split(h_pente, nb)
        dataTxArgile_lst = np.array_split(h_tx_argile, nb)
        dataTxLimon_lst = np.array_split(h_tx_limon, nb)
        dataTxSable_lst = np.array_split(h_tx_sable, nb)
        dataOccSol_lst = np.array_split(h_occ_sol, nb)
        
        # Fichiers résultats
        result_A_lst = []
        result_kg_px_lst = []
        result_epaisseur_lst = []


        for i in range(nb):
            in_h_R = dataR_lst[i]
            in_h_pente = dataPente_lst[i]
            in_h_tx_argile = dataTxArgile_lst[i]
            in_h_tx_limon = dataTxLimon_lst[i]
            in_h_tx_sable = dataTxSable_lst[i]
            in_h_occ_sol = dataOccSol_lst[i]

            # Lancement du calcul
            h_result_A, h_result_kg_px, h_result_epaisseur = lance_calcul(in_h_pente, in_h_R, in_h_tx_argile, in_h_tx_limon, in_h_tx_sable, in_h_occ_sol, occ_sol_coeff_c, ndv)

            # Ajout des résultats dans les listes
            result_A_lst.append(h_result_A)
            result_kg_px_lst.append(h_result_kg_px)
            result_epaisseur_lst.append(h_result_epaisseur)
        
        # Concaténation des résultats
        h_result_A = np.concatenate(tuple(result_A_lst))
        h_result_kg_px = np.concatenate(tuple(result_kg_px_lst))
        h_result_epaisseur = np.concatenate(tuple(result_epaisseur_lst))
    else:
        # Lancement du calcul
        h_result_A, h_result_kg_px, h_result_epaisseur = lance_calcul(h_pente, h_R, h_tx_argile, h_tx_limon, h_tx_sable, h_occ_sol, occ_sol_coeff_c, ndv)

    # Sauvegarde des résultats
    raster_result_A = h_result_A.reshape(dim1,dim2)
    raster_result_kg_px = h_result_kg_px.reshape(dim1,dim2)
    raster_result_epaisseur = h_result_epaisseur.reshape(dim1,dim2)
    
    #####################################################################################################################
    # Ecriture des fichiers de sortie
    #####################################################################################################################
    debut = datetime.datetime.now()
    res = ecriture_raster_sortie(sortie_A, raster_result_A, gdal.GDT_Float32, ds, dim1, dim2)
    res = ecriture_raster_sortie(sortie_P, raster_result_kg_px, gdal.GDT_Float32, ds, dim1, dim2)
    res = ecriture_raster_sortie(sortie_E, raster_result_epaisseur, gdal.GDT_Float32, ds, dim1, dim2)
    ds = None
    print("...Ecriture fichiers sortie terminée en : ", datetime.datetime.now() - debut)
    print("...Temps total de traitement : ", datetime.datetime.now() - t0)


if __name__ == '__main__':
    main()


















