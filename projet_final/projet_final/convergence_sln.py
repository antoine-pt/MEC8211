import numpy as np
import matplotlib.pyplot as plt
import sys
try:
    from fonctions import *
except ImportError:
    print("Error: Could not import the 'fonctions' module. Please ensure it is in the same directory as this script.")
    exit(1)

if __name__ == "__main__":

    print("Voir Convergence.png")
    nr_nz = [3, 5, 10, 20]
    for nr_nz_i in nr_nz:
        print("================================")
        print("Nombre de points selon r et z : {}".format(nr_nz_i))
        T_centre_array = []
        dt_array = []
        dt = 0.1
        while dt<=2: 
            print("Pourcentage de complétion : {}%".format(round((dt-0.1)/2*100, 2)))
            dt_array.append(dt)
            prm = Parametres(nr = nr_nz_i, nz = nr_nz_i, dt = dt)
            Z, R = Position(prm)    
            t = 0
            T_init = np.full((prm.nr, prm.nz), prm.T_four)
            T_t = T_init
            while t<prm.t_fin:
                T_rendu = Temperature(prm, T_t, R)
                T_t = T_rendu
                t += prm.dt
            Temperature_centre = np.max(T_t)
            T_centre_array.append(Temperature_centre)
            dt += 0.2
        plt.plot(dt_array, T_centre_array, label = "nz et nr = {}".format(prm.nz))




    plt.grid()
    plt.ylabel("Temperature max (\u00b0 K)", fontsize=15)
    plt.xlabel("dt", fontsize=15)
    plt.legend()
    plt.title("Température max après {} secondes".format(prm.t_fin), fontsize=20)
    plt.savefig('Convergence.png', dpi=300)