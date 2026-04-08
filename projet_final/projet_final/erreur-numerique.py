import numpy as np
import matplotlib.pyplot as plt
import sys
try:
    from fonctions import *
except ImportError:
    print("Error: Could not import the 'fonctions' module. Please ensure it is in the same directory as this script.")
    exit(1)


def normeL2(ana, sim, params):
    """ Calcule la norme L2 entre une solution MMS analytique et une solution
    numérique. Norme calculée selon R, z et t. 

    Args:
        ana (dico(np.array)): solution MMS
        sim (dico(np.array)): solution numérique
        params: paramètres de la simulation

    Returns:
        tuple: (L2SpatioTemporel, L2Final) - normes L2 spatio-temporelle et au dernier pas de temps
    """
    L2 = 0
    L2f=0
    for time in range(len(ana)):
        for r in range(ana(time).shape[0]):
            error = (ana[r,:] - sim[r,:])
            error_sq=error**2
            L2 +=np.sum(error_sq)
            if time==params.t_fin:
                L2f+=np.sum(error_sq)

    # Dernier pas de temps uniquement
    L2Final = L2f * params.nr * params.nz /(params.R*params.H))

    # Sur tous les pas de temps
    domaine = params.R * params.H * params.t_fin
    L2SpatioTemporel = np.sqrt(L2 / (domaine))

    return L2SpatioTemporel, L2Final




if __name__ == "__main__":
    # Usage typique du solveur

    # Initialisation des parametres
    resolution_dr=[]
    resolution_dz=[]
    resolution_dt=[]
    
    
    prm = Parametres(nr = 15, nz = 15,t_fin = 120 *60, dt = 1)
    Z, R = Position(prm)    
    t = 0
    T_init = np.full((prm.nr, prm.nz), prm.T_four)
    T_t = T_init
    
    print("Pourcentage de complétion : {}%".format(round(current_pct, 2)))




    # Affichage de l'évolution de L2 selon R
    plt.figure(figsize=(8, 6))
    plt.plot(pas,erreur_dr)
    plt.title('Erreur de la solution numérique en fonction de la MMS')
    plt.xlabel('Pas de discrétisation selon r')
    plt.ylabel('Erreur L2')
    plt.show()
