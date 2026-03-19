# Importation des modules
import numpy as np
import sys
import matplotlib.pyplot as plt
try:
    from fonctions import *
except:
    pass

#------------------------------------------------------------------------------
# Code principal pour l'analyse des résultats
#------------------------------------------------------------------------------

# Assignation des paramètres utilisés pour l'analyse
class Parametres:
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Attention de bien utiliser une combinaison de paramètre qui permet de converge
    # Voir test_convergence.py pour choisir une bonne combinaison de paramètres. Exemple de bons paramètres 
    # EX1 : nr et nz = 10, dt = 0.25
    # EX2 : nr et nz = 5, dt = 1.75
    # EX3 : nr et nz = 3, dt = 1.25
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    def __init__(self,epsilon = 0.5,h = 10, k = 5,t_fin = 71*60):
        self.h = h         # [W/m^2*K] Coefficient de convection
        self.epsilon = epsilon  # [-] Emissivité
        self.k = k          # [W/m*K] Conductivité thermique
        self.R = 0.05      # [m] Rayon
        self.H = 0.1     # [m] Hauteur
        self.rho = 2000     # [kg/m^3] Densité
        self.Cp = 1000      # [J/kg*K] Capacité thermique massique
        self.T_inf = 298.15 # [K] Température à l'infini
        self.nr = 10       # [-] Nombre de points selon r
        self.nz = 10    # [-] Nombre de points selon z
        self.t_fin = t_fin  # [s] Temps d'arrêt de la simulation
        self.dt =  0.25    # [s] Pas de temps
        self.dr = self.R / (self.nr - 1)  # Pas dans la direction r [m]
        self.dz = self.H / (self.nz - 1)  # Pas dans la direction z [m]
        self.sigma = 5.670374e-8  # Constante de Stefan-Boltzmann [W/m^2*K^4]
        self.T_four = 800 + 273.15  # Température initiale en Kelvin
        
    def Biot(self):
        return (self.h*2*self.R/self.k)
    
prm = Parametres()
Z, R = Position(prm)
#------------------------------------------------------------------------------



# Vérification du critère de stabilité pour la méthode euler explicite
import sys
print('Vérification du critère de stabilité pour la méthode euler explicite :')
cste = ((prm.k * prm.dt) / (prm.rho * prm.Cp))
explicite = 1 - cste*(2/prm.dr**2) - cste*(2/prm.dz**2)


if explicite > 0:
    print('Le critère de stabilité pour la méthode euler explicite est respecté.')
    print('')
else :
    print('Le critère de stabilité pour la méthode euler explicite n\'est pas respecté.')
    print('Il faut ajuster les valeurs de nr (dr), nz (dz) et dt en conséquence.')
    print(' Arrêt du programme.')
    sys.exit()
#------------------------------------------------------------------------------


# Initialisation de la température initale dans tout le domaine (1/4 du vrai cyclindre)
# Température intiale de 800 C partout dans le cylindre en sortant du four

T_init = np.full((prm.nr, prm.nz), prm.T_four)

#------------------------------------------------------------------------------

""" Génération d'un graphique montrant l'état de température dans le cylindre pour le domaine choisi après (prm. t_fin) secondes.
Selon les paramètres choisis en entrée.
"""
print("Génération d'un graphique montrant l'état de température dans le cyclindre pour le domaine choisi après", prm. t_fin, "secondes :")
print("Voir Temperature_domaine_cylindre.png")

print('')
t = 0
T_t = T_init
while t<prm.t_fin:
    T_rendu = Temperature(prm, T_t,R)
    T_t = T_rendu
    t += prm.dt
Z,R = Position(prm)
r = R.transpose()[0]
z = Z[0]



fig = plt.figure(1)
fig.set_size_inches(10, 10)
plt.xlabel("Position en z (m)", fontsize=20)
plt.ylabel("Position en r (m)", fontsize=20)
fig.suptitle("État de température dans le domaine simplifié \ndu cylindre après {} secondes".format(prm.t_fin), fontsize=25)
pcm = plt.pcolormesh(z, r, T_t)
colorbar = plt.colorbar(pcm)
colorbar.set_label("Température (\u00b0 Kelvin)", fontsize=20)
colorbar.ax.tick_params(labelsize=20)
plt.tick_params(axis='both', labelsize=20)  
plt.savefig('Temperature_domaine_cylindre.png', dpi=300)
plt.show()
#------------------------------------------------------------------------------

""" Question 1 : Au bout de combien de temps est-ce que le cylindre atteindra une 
température maximalede 130 ◦C ?  Quelle sera alors la température à sa surface ?
"""
print('Question 1 :')


Temperature_centre = 800

t_fin = 0
T_init = np.full((prm.nr, prm.nz), prm.T_four)
T_t = T_init

while Temperature_centre >= (130+273.15):
    T_rendu = Temperature(prm, T_t, R)
    T_t = T_rendu
    t_fin += prm.dt
    Temperature_centre = np.max(T_t)
    
max_index = np.unravel_index(np.argmax(T_rendu), T_rendu.shape)
min_index = np.unravel_index(np.argmin(T_rendu), T_rendu.shape)

Temperature_surface = np.min(T_rendu)
po_z, po_r = Position(prm)


max_temp_extremite = np.max([np.max(T_rendu[0, :]), np.max(T_rendu[:, -1])])

print('Après', np.round(t_fin, 2), 'secondes (',np.round(t_fin/60, 2), 'minutes)', 'la température est de', np.round((Temperature_centre-273.15),2), '\u00b0 Celcius. Cette température maximum est situé en z =', Z[max_index], 'et r = ', np.round(R[max_index],1), '.')
print('Au même moment, la température minimum dans le cylindre est alors de', np.round(Temperature_surface-273.15,2), '\u00b0 Celcius. Cette température minimum est situé en z =', Z[min_index], 'et r = ', R[min_index], '. Soit à la surface du cylindre.')
print('La température maximum aux extrémités est alors de', np.round(max_temp_extremite-273.15, 2), '\u00b0 Celcius. Situé en z = 0.00 et r =  0.05. ')
print('')
#------------------------------------------------------------------------------

""" Question 2 : L’évolution temporelle de la température minimale, maximale et moyenne du cylindre.
N’oubliez pas l’influence de votre système de coordonnées lors du calcul de la moyenne.
"""
print('Question 2 :')
print('Voir Q2.png')
print('')



# Paramètre initiaux
T_init = np.full((prm.nr, prm.nz), prm.T_four)
T_t = T_init
prm.t_fin = 1*60*60 # Important à changer pour la bonne évolution
t_init = prm.dt
t_final = prm.t_fin

temps = np.array([0])

max_temp = np.max(T_init)
max_T_vector = np.array([max_temp])

min_temp = np.min(T_init)
min_T_vector = np.array([min_temp])

average_temperature = prm.T_four
average_temperature_vector = np.array([average_temperature])

Z, R = Position(prm)




# Récupération des températures maximum, moyenne et minimum à chaque pas de temps.
while t_init <= t_final:
    temps = np.append(temps, [t_init]) 
    T_rendu = Temperature(prm, T_t,R)
    
    max_T = np.max(T_rendu)
    max_T_vector = np.append(max_T_vector,[max_T])
    
    min_T = np.min(T_rendu)
    min_T_vector = np.append(min_T_vector,[min_T])
       
    numerator = np.sum(T_rendu * R * prm.dr * prm.dz)
    denominator = np.sum(R * prm.dr * prm.dz)
    average_temperature = numerator / denominator
    average_temperature_vector = np.append(average_temperature_vector, [average_temperature])
    T_t = T_rendu
    # Critère pour arrêter la boucle
    t_init += prm.dt

# Génération du graphique
plt.plot(temps, max_T_vector, label = 'Température maximum')
plt.plot(temps, average_temperature_vector, label = 'Température moyenne') 
plt.plot(temps, min_T_vector, label = 'Température minimum')     
plt.title("Évolution temporelle de la température minimale, maximale \net moyenne dans le cylindre",  pad=20)
plt.xlabel("temps (en secondes)")
plt.ylabel("Température (en \u00b0 Kelvin)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Q2.png', dpi=300)
plt.show()
#------------------------------------------------------------------------------

""" Question 3 : L’influence du transfert thermique radiatif sur vos résultats. 
Que se passe-t’il si on néglige la radiation ?
"""
print('Question 3 : ')
print("Voir Q3.png")
print('')
parametres = [Parametres(epsilon = 0),Parametres(epsilon = 0.1), Parametres(epsilon=0.5)]


fig,axs = plt.subplots(nrows = 1, ncols = 3)
fig.set_size_inches(10, 10)

fig.suptitle("Température dans le domaine \ndu cylindre après {} secondes \npour une emmisivité de 0, 0.1 et 0.5".format(prm.t_fin), fontsize=20)

T_temperatures = []

for para in parametres:
    T_init = np.full((prm.nr, prm.nz), prm.T_four)
    t = 0
    T_t = T_init
    while t < para.t_fin:
        T_rendu = Temperature(para, T_t,R)
        T_t = T_rendu
        t += para.dt
    T_temperatures.append(T_t)



# Find common colorbar limits
vmin = min(np.min(T) for T in T_temperatures)
vmax = max(np.max(T) for T in T_temperatures)


# Plot each temperature map separately on its corresponding subplot
for i, (T_t, para) in enumerate(zip(T_temperatures, parametres)):
    Z, R = Position(para)
    r = R.transpose()[0]
    z = Z[0]

    pcm = axs[i].pcolormesh(z, r, T_t, vmin=vmin, vmax=vmax)
    axs[i].set_xlabel("Position en z (m)", fontsize=10)
    axs[i].set_ylabel("Position en r (m)", fontsize=10)
    axs[i].set_title(f"Epsilon = {para.epsilon}", )
    
plt.subplots_adjust(top=0.85)
plt.colorbar(pcm, ax=axs[len(parametres)-1],label = "Température (\u00b0 Kelvin)")
plt.savefig("Q3.png", dpi = 300)
#------------------------------------------------------------------------------





""" Question 4 : L’impact du nombre de Biot (Bi = hD/k) sur l’écart entre la température minimale et maximale dans votre cylindre. 
Simulation pour un temps de 15 min.
"""
print('Question 4 : ')
print("Voir Q4.png")
print('')

parametres_h = [10,10,10]
parametres_k = [0.1,1,10]
t_fin = 60*15
parametres = []
for i,h in enumerate(parametres_h):
    parametres.append(Parametres(h = parametres_h[i], k = parametres_k[i], t_fin = t_fin))



fig,axs = plt.subplots(nrows = 1, ncols = 3)
fig.set_size_inches(10, 10)

fig.suptitle("Température dans le domaine \ndu cylindre après 15 minutes \np selon le nombre de Biot".format(prm.t_fin), fontsize=20)

T_temperatures = []

for para in parametres:
    T_init = np.full((prm.nr, prm.nz), prm.T_four)
    t = 0
    T_t = T_init
    while t < para.t_fin:
        T_rendu = Temperature(para, T_t,R)
        T_t = T_rendu
        t += para.dt
    T_temperatures.append(T_t)

   

# Find common colorbar limits
vmin = min(np.min(T) for T in T_temperatures)
vmax = max(np.max(T) for T in T_temperatures)

# Plot each temperature map separately on its corresponding subplot
for i, (T_t, para) in enumerate(zip(T_temperatures, parametres)):
    Z, R = Position(para)
    r = R.transpose()[0]
    z = Z[0]

    pcm = axs[i].pcolormesh(z, r, T_t, vmin=vmin, vmax=vmax)
    axs[i].set_xlabel("Position en z (m)", fontsize=10)
    axs[i].set_ylabel("Position en r (m)", fontsize=10)
    axs[i].set_title(f"Biot = {para.Biot()}", )
    
plt.subplots_adjust(top=0.85)
plt.colorbar(pcm, ax=axs[len(parametres)-1],label = "Température (\u00b0 Kelvin)")
plt.savefig("Q4.png", dpi = 300)
#------------------------------------------------------------------------------






""" Question 5 : D’établir un modèle simplifié basé sur une équation différentielle ordinaire qui décrit l’évolution de la température de votre cylindre dans la limite de petits nombres de Biot.
Établissez l’erreur de votre modèle simplifié en fonction du nombre de Biot.
"""
print('Question 5 : ')
print('Voir Q5.png')
print('')

#------------------------------------------------------------------------------

# Évolution temporelle pour le modèle simplifié
T_t = prm.T_four
t = 0
temp_biot = prm.T_four
temp_biot_vector = np.array([temp_biot])

while t<1*60*60:
    T_rendu = petit_biot(prm, T_t)
    T_t = T_rendu
    t += prm.dt
    temp_biot_vector = np.append(temp_biot_vector,[T_rendu])


# Paramètre initiaux
T_init = np.full((prm.nr, prm.nz), prm.T_four)
T_t = T_init
prm.t_fin = 1*60*60 # Important à changer pour la bonne évolution
t_init = prm.dt
t_final = prm.t_fin

temps = np.array([0])

max_temp = np.max(T_init)
max_T_vector = np.array([max_temp])

min_temp = np.min(T_init)
min_T_vector = np.array([min_temp])

average_temperature = prm.T_four
average_temperature_vector = np.array([average_temperature])


# Récupération des températures maximum, moyenne et minimum à chaque pas de temps.
while t_init <= t_final:
    temps = np.append(temps, [t_init]) 
    T_rendu = Temperature(prm, T_t,R)
    
    max_T = np.max(T_rendu)
    max_T_vector = np.append(max_T_vector,[max_T])
    
    min_T = np.min(T_rendu)
    min_T_vector = np.append(min_T_vector,[min_T])
       
    numerator = np.sum(T_rendu * R * prm.dr * prm.dz)
    denominator = np.sum(R * prm.dr * prm.dz)
    average_temperature = numerator / denominator
    average_temperature_vector = np.append(average_temperature_vector, [average_temperature])
    T_t = T_rendu
    # Critère pour arrêter la boucle
    t_init += prm.dt

# Génération du graphique
plt.figure(11)
plt.plot(temps, temp_biot_vector, label='Approximation pour petit nombre de biot')
plt.plot(temps, max_T_vector, label = 'Température maximum')
plt.plot(temps, average_temperature_vector, label = 'Température moyenne') 
plt.plot(temps, min_T_vector, label = 'Température minimum')     
plt.title("Évolution temporelle de la température en fonction de la technique utilisée",  pad=20)
plt.xlabel("temps (en secondes)")
plt.ylabel("Température (en \u00b0 Kelvin)")
plt.legend()
plt.grid()
plt.tight_layout()
plt.savefig('Q5.png', dpi=300)
plt.show()


#------------------------------------------------------------------------------


""" Bonus : Génération de l'évolution de la température dans la totalité du domaine du cylindre.
"""

print("Bonus : Génération de l'évolution de la température dans la totalité du domaine du cylindre.")
print('Voir Q_BONUS.png')


t = 0
T_init = np.full((prm.nr, prm.nz), prm.T_four)
T_t = T_init
while t<prm.t_fin:
    T_rendu = Temperature(prm, T_t,R)
    T_t = T_rendu
    t += prm.dt
Z,R = Position(prm)
r = R.transpose()[0]
z = Z[0]


Z,R = Position(prm)
r = R.transpose()[0]
z = Z[0]

fig = plt.figure(12)
fig.set_size_inches(10, 10)
plt.xlabel("Position en z (m)", fontsize=20)
plt.ylabel("Position en r (m)", fontsize=20)
fig.suptitle("État de température dans l'ensemble \ndu cylindre après {} secondes".format(prm.t_fin), fontsize=25)
pcm = plt.pcolormesh(z, r, T_t)
plt.pcolormesh(z, -r, T_t)
plt.pcolormesh(-z, r, T_t)
plt.pcolormesh(-z, -r, T_t)
colorbar = plt.colorbar(pcm)
colorbar.set_label("Température (\u00b0 Kelvin)", fontsize=20)
colorbar.ax.tick_params(labelsize=20)
plt.tick_params(axis='both', labelsize=20)  
plt.savefig('Q_BONUS.png', dpi=300)
plt.show()







