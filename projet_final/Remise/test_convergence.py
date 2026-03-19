# Fichier afin de trouver quels paramètres à utiliser pour avoir la bonne température
import numpy as np
import matplotlib.pyplot as plt
import sys
try:
    from fonctions import *
except:
    pass

class Parametres:
    
    def __init__(self,epsilon = 0.5, nr = 5, nz = 5, dt = 1):
        self.h = 10         # [W/m^2*K] Coefficient de convection
        self.epsilon = epsilon  # [-] Emissivité
        self.k = 5          # [W/m*K] Conductivité thermique
        self.R = 0.05      # [m] Rayon
        self.H = 0.1     # [m] Hauteur
        self.rho = 2000     # [kg/m^3] Densité
        self.Cp = 1000      # [J/kg*K] Capacité thermique massique
        self.T_inf = 298.15 # [K] Température à l'infini
        self.nr = nr       # [-] Nombre de points selon r
        self.nz = nz    # [-] Nombre de points selon z
        self.t_fin = 60*30  # [s] Temps d'arrêt de la simulation
        self.dt = dt    # [s] Pas de temps
        self.dr = self.R / (self.nr - 1)  # Pas dans la direction r [m]
        self.dz = self.H / (self.nz - 1)  # Pas dans la direction z [m]
        self.sigma = 5.670374e-8  # Constante de Stefan-Boltzmann [W/m^2*K^4]
        self.T_four = 800 + 273.15  # Température initiale en Kelvin
prm = Parametres()


# Vérification du critère de stabilité pour la méthode euler explicite
print('Vérification du critère de stabilité pour la méthode euler explicite :')
cste = ((prm.k * prm.dt) / (prm.rho * prm.Cp))
explicite = 1 - cste*(2/prm.dr**2) - cste*(2/prm.dz**2)

if explicite > 0:
    print('Le critère de stabilité pour la méthode euler explicite est respecté.')
    print('')
else :
    print('Le critère de stabilité pour la méthode euler explicite n\'est pas respecté. Il faut ajuster les valeurs de dt en conséquence. Arrêt du programme.')
    sys.exit()





print("Voir Convergence.png")
nr_nz = [3, 5, 10, 20]
for nr_nz_i in nr_nz: 
    T_centre_array = []
    dt_array = []
    dt = 0.1
    while dt<=2: 
        dt_array.append(dt)
        prm = Parametres(nr = nr_nz_i, nz = nr_nz_i, dt = dt)
        t = 0
        T_init = np.full((prm.nr, prm.nz), prm.T_four)
        T_t = T_init
        while t<prm.t_fin:
            T_rendu = Temperature(prm, T_t)
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
