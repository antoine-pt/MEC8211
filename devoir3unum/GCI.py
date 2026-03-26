import numpy as np
import matplotlib.pyplot as plt
import csv

with open('devoir3unum/results_k.csv', 'r') as file:
    csvr = csv.reader(file)
    next(csvr)

    seed = []
    i = -1
    data = [row for row in csvr]
    array = np.zeros((3,len(data)//3,3))
    for row in data:
        if row[0] not in seed:
            seed.append(row[0])
            i+=1
            j=0
        array[i,j,:] = row[1:]
        j+=1

    avgArray = np.mean(array, axis = 0)

    # Calcul des de l'écart-type pour IC sur k
    std_k = np.std(array[:,:,2], axis = 0)
    erreur_type_k = std_k/np.sqrt(array.shape[0])
    t_95_2 = 2.920 # unilatéral, 95% de confiance
    IC = t_95_2 * erreur_type_k
    yerr = [np.zeros_like(IC),IC]

    # Calcul de l'IC sur l'erreur relative de k
    IC_relative = IC / avgArray[-1,2]
    yerr = [np.zeros_like(IC_relative[:-1]), IC_relative[:-1]]

    # ---- PLOT de l'erreur relative de k en fonction de dx ----


    k_fine = avgArray[-1,2]
    relative_error = np.abs(k_fine - avgArray[:,2]) / k_fine

    plt.figure(figsize=(8,6))
    plt.loglog(avgArray[:-1,1], relative_error[:-1], marker='o', linestyle='-', color='b')
    plt.errorbar(avgArray[:-1,1], relative_error[:-1], yerr=yerr, fmt='o', color='b', ecolor='r', capsize=5, label="IC à 95% pour l'erreur relative de la perméabilité (k)")
    
    # Ajout de la régréssion linéaire sur les points (en excluant le dernier point)
    p = np.polyfit(np.log(avgArray[:-2,1]), np.log(relative_error[:-2]), 1)
    plt.loglog(avgArray[:-1,1], np.exp(p[1]) * avgArray[:-1,1]**p[0], linestyle='--', color='g', label=f'Fit: slope={p[0]:.2f}')

    # Paramétrisation du graphique
    plt.xlabel('dx (µm)', fontsize=14)
    plt.ylabel('(k_fine-k_dx)/k_fine (-)', fontsize=14)
    plt.title('Erreur relative de la perméabilité en fonction de la taille des cellules dx (µm)', fontsize=16)
    plt.legend()
    plt.grid(True, which="both", ls="--", linewidth=0.5)

    # Ajustement de l'échelle du grpahique
    y_min = np.min(relative_error[:-1])
    y_max = np.max(relative_error[:-1] + IC_relative[:-1])
    plt.ylim(y_min * 0.9, y_max * 1.1)

    plt.show()
    plt.close()

    # -------------------------------------------------

    # ----------------- Calcul du GCI -----------------

    p_f = 2 # ordre attendu
    p_hat = p[0] # ordre calculé
    rel = np.abs(p_hat - p_f) / p_f 
    if rel < 0.1:
        Fs = 1.25
        p = p_f
    else:
        Fs = 3.0
        p = min(max(0.5,p_hat),p_f)
    
    r = 2 # prendre des points avec raffinement de 2
    GCI = (Fs/(r**p-1)) * np.abs(k_fine - avgArray[-2,2])
    print(f"GCI = {GCI:.6e} µm²")
    print(f"u_num = {GCI/2:.6e} µm²")

    # -------------------------------------------------



    # ---- PLOT de k en fonction de dx ----
    # ---- Non utilisé ----

    # plt.figure(figsize=(8,6))
    # plt.semilogy(avgArray[:,1], avgArray[:,2], marker='o', linestyle='-', color='b')
    # plt.errorbar(avgArray[:,1], avgArray[:,2], yerr=yerr, fmt='o', color='b', ecolor='r', capsize=5, label='IC à 95% pour k')
    # plt.xlabel('dx (µm)', fontsize=14)
    # plt.ylabel('k (µm²)', fontsize=14)
    # plt.title('Permeabilité en fonction de la taille des cellules dx (µm)', fontsize=16)
    # plt.legend()
    # plt.grid(True, which="both", ls="--", linewidth=0.5)

    # # Ajustement de l'échelle du grpahique
    # y_min = np.min(avgArray[:,2])
    # y_max = np.max(avgArray[:,2] + IC)
    # plt.ylim(y_min * 0.9, y_max * 1.1)
    
    # plt.show()
    # plt.close()

