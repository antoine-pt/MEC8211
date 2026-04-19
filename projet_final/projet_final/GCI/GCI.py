import numpy as np
import math
try:
    from src.fonctions import *
except ImportError:
    print("Error: Could not import the 'fonctions' module. Please ensure it is in the same directory as this script.")
    exit(1)

if __name__ == "__main__":
    """
    Script de vérification de convergence numérique (GCI) pour un solveur thermique.

    Ce script exécute plusieurs simulations en faisant varier le pas de temps (dt)
    et la résolution du maillage (nr, nz). Les solutions obtenues sont comparées
    à l'aide d'une norme L2 afin d'évaluer l'erreur entre maillages grossiers et fins.

    Le Grid Convergence Index (GCI) est calculé pour chaque paramètre (temps,
    direction radiale et axiale), puis combiné pour estimer l'incertitude numérique
    globale (U_num).
    """

    # Initialisation des parametres
    nr=[15, 30]
    nz=[15, 30]
    dt=[1, 2]
    
    
    l_dt=[]
    for i in dt:
        prm = Parametres(nr = 8, nz = 8,t_fin = 120 *60, dt = i)
        T_init = np.full((prm.nr, prm.nz), prm.T_four)
        T_t = T_init
        while prm.time<prm.t_fin:
            T_tp1 = Temperature(prm, T_t)
            T_t = T_tp1
            prm.Time(prm.dt)
        l_dt.append(T_t)
    
    error=l_dt[0]-l_dt[1]
    l2_dt=np.sum(error**2)/(T_t.size)
    
    GCI_dt=1.25*l2_dt/(2^1-1)
    print("GCI_t : {}K".format(GCI_dt))
    
    
    
    
    l_nr=[]
    for i in nr:
        prm = Parametres(nr = i, nz = 15,t_fin = 120 *60, dt = 0.4)
        T_init = np.full((prm.nr, prm.nz), prm.T_four)
        T_t = T_init
        while prm.time<prm.t_fin:
            T_tp1 = Temperature(prm, T_t)
            T_t = T_tp1
            prm.Time(prm.dt)
        l_nr.append(T_t)
    

    lign=np.size(l_nr[1], 0)
    for j in range(0,math.floor(lign/2)):
        l_nr[1]=np.delete(l_nr[1], lign-1-2*j, 0) 
    
    error=l_nr[0]-l_nr[1]
    l2_nr=np.sum(error**2)/(l_nr[0].size)
    
    GCI_nr=1.25*l2_nr/(2^2-1)
    print("GCI_r : {}K".format(GCI_nr))

    
    
    l_nz=[]
    for i in nz:
        prm = Parametres(nr = 15, nz = i,t_fin = 120 *60, dt = 0.4)
        T_init = np.full((prm.nr, prm.nz), prm.T_four)
        T_t = T_init
        while prm.time<prm.t_fin:
            T_tp1 = Temperature(prm, T_t)
            T_t = T_tp1
            prm.Time(prm.dt)
        l_nz.append(T_t)
    

    column=np.size(l_nz[1], 1)
    for j in range(0,math.floor(column/2)):
        l_nz[1]=np.delete(l_nz[1], column-1-2*j, 1) 
    
    error=l_nz[0]-l_nz[1]
    l2_nz=np.sum(error**2)/(l_nz[0].size)
    
    GCI_nz=1.25*l2_nz/(2^2-1)
    print("GCI_z : {}K".format(GCI_nz))
    
    
    U_num=np.sqrt((GCI_nr/2)**2+(GCI_nz/2)**2+(GCI_dt/2)**2)
    print("U_num : {}K".format(U_num))
    