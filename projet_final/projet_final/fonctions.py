# Importation des modules
import numpy as np
import sys
import sympy as sp

# Assignation des paramètres utilisés pour l'analyse
class Parametres:
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Attention de bien utiliser une combinaison de paramètre qui permet de converge
    # Voir test_convergence.py pour choisir une bonne combinaison de paramètres. Exemple de bons paramètres 
    # EX1 : nr et nz = 10, dt = 0.25
    # EX2 : nr et nz = 5, dt = 1.75
    # EX3 : nr et nz = 3, dt = 1.25
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    def __init__(
        self,
        *,
        epsilon: float = 0.5,
        nr: int = 10,
        nz: int = 10,
        dt: float = 0.25,
        h: float = 10.0,
        k: float = 5.0,
        t_fin: float = 71 * 60,
        solution_MMS_sympy = sp.sympify(None)
    ):

        self.Rmax = 0.05      # [m] Rayon
        self.H = 0.1     # [m] Hauteur
        self.rho = 2000     # [kg/m^3] Densité
        self.Cp = 1000      # [J/kg*K] Capacité thermique massique
        self.T_inf = 298.15 # [K] Température à l'infini
        self.sigma = 5.670374e-8  # Constante de Stefan-Boltzmann [W/m^2*K^4]
        self.T_four = 800 + 273.15  # Température initiale en Kelvin
        self.time = 0 # temps initialisé à 0
 
        self.h = h         # [W/m^2*K] Coefficient de convection
        self.epsilon = epsilon  # [-] Emissivité
        self.k = k          # [W/m*K] Conductivité thermique
        self.nr = nr       # [-] Nombre de points selon r
        self.nz = nz    # [-] Nombre de points selon z
        self.t_fin = t_fin  # [s] Temps d'arrêt de la simulation
        self.dt =  dt   # [s] Pas de temps

        self.dr = self.Rmax / (self.nr - 1)  # Pas dans la direction r [m]
        self.dz = (self.H/2) / (self.nz - 1)  # Pas dans la direction z [m]


        # Vérification du critère de stabilité pour la méthode euler explicite
        print('Vérification du critère de stabilité pour la méthode euler explicite :')
        cste = ((self.k * self.dt) / (self.rho * self.Cp))
        explicite = 1 - cste*(2/self.dr**2) - cste*(2/self.dz**2)
        
        # Calcul du dt maximal
        dt_max = (self.rho * self.Cp) / (self.k * (2/self.dr**2 + 2/self.dz**2))
        print(f"Dt maximal autorisé pour les paramètres d'entrée: {dt_max:.6f} [s]")
        print(f'Dt utilisé : {self.dt:.6f} [s]')

        if explicite > 0:
            print('Le critère de stabilité pour la méthode euler explicite est respecté.')
            print('')
        else :
            print('Le critère de stabilité pour la méthode euler explicite n\'est pas respecté.' \
            ' Il faut ajuster les valeurs de dt en conséquence.')
            
            if input ("Voulez-vous continuer avec le dt actuel ? (y/n) :").lower() != 'y':  
                if input ("voulez-vous plutôt poursuivre avec le dt maximal autorisé ? (y/n) :").lower() == 'y':
                    self.dt = dt_max
                    print(f"Dt ajusté à {self.dt:.6f} [s]")
                else:
                    print("Arrêt du programme.")
                    sys.exit()
            else:
                print('Continuons avec le dt actuel, mais soyez conscient que cela peut entraîner des résultats instables.')
                print('')

        # Calcul des matrices de position
        self.Z, self.R = Position(self)
        self.Rmin = np.min(self.R)

        # Setting the MMS solution if enabled
        if solution_MMS_sympy is None:
            r_var,z_var,t_var = sp.symbols('r z t')
            symbols = (r_var,z_var,t_var)

            self.solution_MMS_sympy = sp.sympify(0)
            self.solution_MMS = sp.lambdify((r_var,z_var,t_var), sp.sympify(0) , 'numpy') # source nulle par défaut
            self.source = np.zeros_like(self.R)
            self.MMS = False

        else:
            r_var,z_var,t_var = sp.symbols('r z t')
            symbols = (r_var,z_var,t_var)

            self.solution_MMS_sympy = solution_MMS_sympy
            self.solution_MMS = sp.lambdify(symbols, self.solution_MMS_sympy)
            self.solution_MMS_diff_r = sp.lambdify(symbols,sp.diff(self.solution_MMS_sympy, r_var))
            self.solution_MMS_diff_z = sp.lambdify(symbols,sp.diff(self.solution_MMS_sympy, z_var))
            self.solution_MMS_diff_diff_r = sp.lambdify(symbols,sp.diff(self.solution_MMS_sympy, r_var,2))
            self.solution_MMS_diff_diff_z = sp.lambdify(symbols,sp.diff(self.solution_MMS_sympy, z_var,2))
            self.solution_MMS_diff_t = sp.lambdify(symbols,sp.diff(self.solution_MMS_sympy, t_var,1))

            self.source = self.rho * self.Cp * self.solution_MMS_diff_t(self.R,self.Z,self.time) - \
                        self.k * self.solution_MMS_diff_diff_r(self.R,self.Z,self.time) - \
                        self.k * (1/self.R) * self.solution_MMS_diff_r(self.R,self.Z,self.time) - \
                        self.k * self.solution_MMS_diff_diff_z(self.R,self.Z,self.time)
            
            self.MMS = True
            
    def Biot(self):
        return (self.h*2*self.Rmax/self.k)
    
    def Time(self,dt):
        self.time += dt
        self.update_source_MMS()
        return self.time
    
    def update_source_MMS(self):
        if self.MMS:
            self.source = self.rho * self.Cp * self.solution_MMS_diff_t(self.R,self.Z,self.time) - \
                self.k * self.solution_MMS_diff_diff_r(self.R,self.Z,self.time) - \
                self.k * (1/self.R) * self.solution_MMS_diff_r(self.R,self.Z,self.time) - \
                self.k * self.solution_MMS_diff_diff_z(self.R,self.Z,self.time)
    
#------------------------------------------------------------------------------


# FONCTIONS UTILISÉES DANS L'ANALYSE

def Position(prm):
    """ Fonction générant deux matrices de discrétisation de l'espace.

    Entrées:
        - prm : Paramètres
            - prm.R : Borne supérieure du domaine en r, Borne inférieure de R étant à l'origine
            - prm.H : Borne supérieure du domaine en z, Borne inférieure de Z étant à l'origine
            - prm.nz : Discrétisation de l'espace en z (nombre de points)
            - prm.nr : Discrétisation de l'espace en r (nombre de points)

    Sorties (dans l'ordre énuméré ci-bas):
        - z : Matrice (array) de dimension (nr x nz) qui contient la position en z
        - r : Matrice (array) de dimension (nr x nz) qui contient la position en r
        
            * Exemple de matrices position :
            * Si prm.H = 2 et prm.R = 2
            * Avec nr = 3 et nz = 3
                r = [[2. 2. 2.]
                     [1. 1. 1.]
                     [0. 0. 0.]]

                z = [[0. 1. 2.]
                     [0. 1. 2.]
                     [0. 1. 2.]]
    """
    rvector = np.linspace(0, prm.Rmax, prm.nr)
    zvector = np.linspace(0, prm.H/2, prm.nz)
    r = np.zeros([prm.nr, prm.nz])
    
    # ici, on inverse nr car on va transposer plus tard
    z = np.zeros([prm.nz, prm.nr])
    
    for i in range(prm.nr):
        r[i] = np.full(prm.nz, rvector[-1-i])
        
    for i in range(prm.nz):
        z[i] = np.full(prm.nr, zvector[i])
    
    
    # transposition pour obtenir le bon format
    z = np.transpose(z)
        
    return z, r

def Milieu(prm, T_tdt_middle):
    """ Fonction permettant de calculer la température au milieu du cylindre à un instant t+dt.

    Entrées:
        - prm : Paramètres
            - prm.Rmax : Rayon [m]
            - prm.H : Hauteur [m]
            - prm.nr : Nombre de points selon r [-]
            - prm.nz : Nombre de points selon z [-]
            - prm.k : Conductivité thermique [W/m*K]
            - prm.rho : Densité [kg/m^3]
            - prm.Cp : Capacité thermique massique [J/kg*K]
            - prm.dt : Pas de temps [s]
        - T_t : Matrice des températures au temps t

    Sorties (dans l'ordre énuméré ci-bas):
        - T_tdt : Matrice "entre-deux" contenant les Températures milieu au temps t+dt et les Températures frontières au temps t
    """

    cste = ((prm.k * prm.dt) / (prm.rho * prm.Cp))
    
    T_tdt = T_tdt_middle.copy() # on copie T_tdt_middle pour ne pas écraser les températures frontières qui sont utilisées dans le calcul des températures milieu

    for r in range(prm.nr):
        for z in range(prm.nz):

            if r != 0 and z != 0 and r != prm.nr-1 and z != prm.nz-1: 
                dist = prm.R[r,z] - np.min(prm.R)
                T_tdt[r,z] = cste*((T_tdt_middle[r-1,z]-2*T_tdt_middle[r,z]+T_tdt_middle[r+1,z])/(prm.dr**2)     \
                                + (1/(2*prm.dr*dist))*(T_tdt_middle[r-1,z]-T_tdt_middle[r+1,z])    \
                                + (T_tdt_middle[r,z-1]-2*T_tdt[r,z]+T_tdt_middle[r,z+1])/(prm.dz**2)) \
                                + (T_tdt_middle[r,z]) \
                                + prm.source[r,z] * prm.dt / (prm.rho * prm.Cp)
           
            else:
                pass
        

    return T_tdt
            

def Temperature(prm, T_t):
    """ Fonction permettant de calculer la température aux frontières du domaine à un instant t+dt.

    Entrées:
        - prm : Paramètres
            - prm.nr : Nombre de points selon r [-]
            - prm.nz : Nombre de points selon z [-]
            - prm.k : Coefficient de conductivité thermique [W/m*K]
            - prm.h : Coefficient de convection [W/m^2*K] 
            - prm.epsilon : Emissivité [-] 
            - prm.sigma : Constante de Stefan-Boltzmann [W/m^2*K^4]
            - prm.T_inf : Température à l'infini
            
        - T_t : Matrice "entre-deux" contenant les Températures milieu au temps t+dt et les Températures frontières au temps t
            

    Sorties (dans l'ordre énuméré ci-bas):
        - T_tdt : Matrice des températures au temps t+dt pour tout le domaine
    """

    T_tdt = T_t.copy() # on copie T_t pour ne pas écraser les températures frontières qui sont utilisées dans le calcul des températures milieu
    


    if not prm.MMS:
        conv_rad = prm.h * (T_t - prm.T_inf) + \
            prm.epsilon * prm.sigma * (T_t**4 - prm.T_inf**4)
        
        for r in range(prm.nr):
            for z in range(prm.nz):
                    # extrémité supérieure (r=R)
                    if r == 0:           
                        T_tdt[r,z] = (1/3) * ( -T_t[r+2,z] \
                                            +  4 * T_t[r+1,z] \
                                                - (2*prm.dr * conv_rad[r,z] / prm.k) )
                
                    # extrémité droite (z=H/2)
                    elif z == prm.nz-1 or (z == prm.nz-1 and r == prm.nr-1):
                        T_tdt[r,z] = (1/3) * ( -T_t[r,z-2] \
                                            +  4 * T_t[r,z-1] \
                                                - (2*prm.dz * conv_rad[r,z] / prm.k) )
                        
                    # extrémité gauche (z=0)
                    elif z == 0 or (z == 0  and r == prm.nr-1):   
                        T_tdt[r, z] = (4/3) * T_t[r, z+1] - (1/3) * T_t[r, z+2]
                    
                    # milieu (r = 0)
                    # condition de symmétrie
                    elif r == prm.nr-1:            
                        T_tdt[r, z] = (4/3) * T_t[r-1, z] - (1/3) * T_t[r-2, z]
                
    else : 
        for r in range(prm.nr):
            for z in range(prm.nz):

                T_hat = prm.solution_MMS(prm.R[r,z], prm.Z[r,z], prm.time+prm.dt)
                
                # extrémité supérieure (r=Rmax)
                if r == 0:       
                       
                    T_tdt[r,z] = T_hat
            
                # extrémité droite (z=H/2)
                elif z == prm.nz-1 or (z == prm.nz-1 and r == prm.nr-1): 
                    T_tdt[r,z] = T_hat
                    
                # extrémité gauche (z=0)
                elif z == 0 or (z == 0  and r == prm.nr-1):   
                    T_tdt[r, z] = T_hat
                
                # milieu (r = 0)
                # condition de symmétrie
                elif r == prm.nr-1:   
                    T_tdt[r, z] = T_hat
            
            
        
    T_tdt = Milieu(prm, T_tdt)
                
    return T_tdt


def normeL1(ana, sim, params):
    """ Calcule la norme L1 entre une solution MMS analytique et une solution
    numérique en 2D. Norme calculée en temps et espace!

    Args:
        ana (np.array): solution MMS
        sim (np.array): solution numérique
        params: paramètres de la simulation

    Returns:
        tuple: (L1SpatioTemporel, L1Final) - normes L1 spatio-temporelle et au dernier pas de temps
    """
    L1 = 0
    r_weights = params.R - params.Rmin # valeurs de r (venant du rdr)
    rayon = params.Rmax-params.Rmin
    for time in range(ana.shape[0]):
        weighted_error = np.abs(ana[time,:,:] - sim[time,:,:]) * r_weights
        L1 += np.sum(weighted_error) * params.dr *params.dz * params.dt
    
    # Dernier pas de temps uniquement
    L1Final = np.sum(weighted_error) * params.dr * params.dz / (rayon * (params.H/2))

    # Sur tous les pas de temps
    domaine = rayon * (params.H/2) * params.t_fin
    L1SpatioTemporel = L1 / (domaine)

    return L1SpatioTemporel, L1Final

def normeL2(ana, sim, params):
    """ Calcule la norme L2 entre une solution MMS analytique et une solution
    numérique en 2D. Norme calculée en temps et espace!

    Args:
        ana (np.array): solution MMS
        sim (np.array): solution numérique
        params: paramètres de la simulation

    Returns:
        tuple: (L2SpatioTemporel, L2Final) - normes L2 spatio-temporelle et au dernier pas de temps
    """
    L2 = 0
    r_weights = params.R - params.Rmin # valeurs de r (venant du rdr)
    rayon = params.Rmax-params.Rmin
    for time in range(ana.shape[0]):
        error = ana[time,:,:] - sim[time,:,:]
        weighted_error_sq = error**2 * r_weights
        L2 += np.sum(weighted_error_sq) * params.dr *params.dz * params.dt

    # Dernier pas de temps uniquement
    L2Final = np.sqrt(np.sum(weighted_error_sq) * params.dr * params.dz /(rayon * (params.H/2)))
    
    # Sur tous les pas de temps
    domaine =  rayon * (params.H/2) * params.t_fin
    L2SpatioTemporel = np.sqrt(L2 / (domaine))

    return L2SpatioTemporel, L2Final

def normeLinf(ana, sim,params):
    """ Calcule la norme Linf entre une solution MMS analytique et une solution
    numérique en 2D. Norme calculée en temps et espace!

    Args:
        ana (np.array): solution MMS
        sim (np.array): solution numérique

    Returns:
        float: norme Linf entre les deux solutions
    """

    return np.max(np.abs(ana - sim))



