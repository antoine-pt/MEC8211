# Importation des modules
import numpy as np

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
    rvector = np.linspace(0, prm.R, prm.nr)
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

def milieu(prm, T_t,R):
    """ Fonction permettant de calculer la température au milieu du cylindre à un instant t+dt.

    Entrées:
        - prm : Paramètres
            - prm.R : Rayon [m]
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
    T_tdt = T_t

    for r in range(prm.nr):
        for z in range(prm.nz):

            if r != 0 and z != 0 and r != prm.nr-1 and z != prm.nz-1: 
                T_tdt[r,z] = cste*((T_t[r-1,z]-2*T_t[r,z]+T_t[r+1,z])/(prm.dr**2) + (1/(2*prm.dr*R[r,z]))*(T_t[r-1,z]-T_t[r+1,z]) + (T_t[r,z-1]-2*T_t[r,z]+T_t[r,z+1])/(prm.dz**2)) + (T_t[r,z])
           
            else:
                pass
        

    return T_tdt
            

def Temperature(prm, T_t,R):
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
    T_tdt = milieu(prm, T_t,R)


    for r in range(prm.nr):
        for z in range(prm.nz):
        
            if r == 0:           
                T_tdt[r,z] = ((4*T_tdt[r+1, z]-1*T_tdt[r+2, z])/(2*prm.dr)+(prm.h/prm.k)*(prm.T_inf)-(prm.epsilon*prm.sigma/prm.k)*(T_t[r,z]**4-prm.T_inf**4))/(3/(2*prm.dr)+prm.h/prm.k)
        
        
            elif z == prm.nz-1:
                T_tdt[r,z] = ((4*T_tdt[r, z-1]-1*T_tdt[r, z-2])/(2*prm.dz)+(prm.h/prm.k)*(prm.T_inf)-(prm.epsilon*prm.sigma/prm.k)*(T_t[r,z]**4-prm.T_inf**4))/(3/(2*prm.dz)+prm.h/prm.k)
           
            
            
            elif r == prm.nr-1:          
                T_tdt[r, z] = (4/3) * T_tdt[r-1, z] - (1/3) * T_tdt[r-2, z]
            
            
            
            elif z == 0:     
                T_tdt[r, z] = (4/3) * T_tdt[r, z+1] - (1/3) * T_tdt[r, z+2]
                
    return T_tdt


def petit_biot(prm, T_t):
    """ Fonction permettant de calculer la température à un instant t+dt pour un petit nombre de biot.

    Entrées:
        - prm : Paramètres
            - prm.h : Coefficient de convection [W/m^2*K] 
            - prm.epsilon : Emissivité [-] 
            - prm.sigma : Constante de Stefan-Boltzmann [W/m^2*K^4]
            - prm.T_inf : Température à l'infini
            - prm.R : Rayon [m]
            - prm.H : Hauteur [m]
            - prm.rho : Densité [kg/m^3]
            - prm.Cp : Capacité thermique massique [J/kg*K]
            - prm.dt : Pas de temps [s]
            
        - T_t : Température "entre-deux" au temps t
            

    Sorties (dans l'ordre énuméré ci-bas):
        - T_tdt : Température à un instant t+dt pour un petit nombre de biot.
    """
    
    cste1 = ((prm.dt * prm.h)  / (prm.rho * prm.Cp )) * ((2*(prm.H+prm.R))/(prm.H * prm.R))
    cste2 = ((prm.dt * prm.epsilon * prm.sigma) / (prm.rho * prm.Cp)) * ((2*(prm.H+prm.R))/(prm.H * prm.R))
    
    T_tdt = T_t - (cste1 * (T_t-prm.T_inf)) - (cste2 * (T_t**4-prm.T_inf**4))
    
    return T_tdt




