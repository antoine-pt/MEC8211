import numpy as np
import matplotlib.pyplot as plt
import sys
try:
    from fonctions import *
except ImportError:
    print("Error: Could not import the 'fonctions' module. Please ensure it is in the same directory as this script.")
    exit(1)

if __name__ == "__main__":
    # Usage typique du solveur

    # Initialisation des parametres
    prm = Parametres(nr = 30, nz = 30,t_fin = 120 *60, dt = 12)
    Z, R = Position(prm)    
    t = 0
    T_init = np.full((prm.nr, prm.nz), prm.T_four)
    T_t = T_init
    
    pourcentage = 5.0
    while t<prm.t_fin:
        current_pct = (t + prm.dt) / prm.t_fin * 100
        if current_pct >= pourcentage or (t + prm.dt) >= prm.t_fin:
            print("Pourcentage de complétion : {}%".format(round(current_pct, 2)))
            while pourcentage <= current_pct:
                pourcentage += 5.0
        T_tp1 = Temperature(prm, T_t, R)
        T_t = T_tp1
        t += prm.dt

    # Affichage de la température finale
    plt.figure(figsize=(8, 6))
    plt.contourf(Z, R, T_t, levels=50, cmap='inferno')
    plt.colorbar(label='Température (°C)')
    plt.title('Distribution de la température dans le cylindre à t = {} s'.format(prm.t_fin))
    plt.xlabel('Position z (m)')
    plt.ylabel('Position r (m)')
    plt.show()



