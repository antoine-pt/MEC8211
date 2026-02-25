import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


class Parametres():
    """Classe contenant les parametres du problème."""
    def __init__(self, nPts, nTime, endTime, R, D, k, Ce):
        self.nPts = nPts
        self.R = R
        self.nTime = nTime
        self.endTime = endTime
        self.D = D
        self.k = k
        self.Ce = Ce
        
    @property
    def dr(self):
        return self.R / (self.nPts - 1)
    
    @property
    def dt(self):
        return self.endTime / self.nTime
    
    @property
    def pos(self):
        return np.linspace(0, self.R, self.nPts)
    

def normeL1(ana, sim):
    """ Calcule la norme L1 entre une solution analytique et une solution
    numérique.

    Args:
        ana (np.array): solution analytique
        sim (np.array): solution numérique

    Returns:
        float: norme L1 entre les deux solutions
    """
    return np.sum(np.abs(ana - sim)) / ana.size

def normeL2(ana, sim):
    """ Calcule la norme L2 entre une solution analytique et une solution
    numérique.

    Args:
        ana (np.array): solution analytique
        sim (np.array): solution numérique

    Returns:
        float: norme L2 entre les deux solutions
    """
    return np.sqrt(np.sum((ana - sim)**2) / ana.size)

def normeLinf(ana, sim):
    """ Calcule la norme Linf entre une solution analytique et une solution
    numérique.

    Args:
        ana (np.array): solution analytique
        sim (np.array): solution numérique

    Returns:
        float: norme Linf entre les deux solutions
    """
    return np.max(np.abs(ana - sim))

def solveur_avant(params):
    """ Fonction permettant de résoudre le problème numériquement en fonction des paramsètres."""

    ## Création des matrices A et b (Ax = b)
    A = np.zeros((params.nPts,params.nPts))
    b = np.zeros(params.nPts)

    for i in range(params.nPts):
        ri = params.pos[i]

        # Condition limite de neumann en r=0
        if i == 0:
            A[i,i] = -1
            A[i,i+1] = 1

            b[i] = 0

        # Condition limite de dirichlet en r=R
        elif i == params.nPts-1:
            A[i,i] = 1
            b[i] = params.Ce

        # Milieu du domaine
        else:
            A[i,i-1] = 1 / params.dr**2
            A[i,i] = -1 / (params.dr*ri)- 2 / params.dr**2
            A[i,i+1] = 1 / (params.dr*ri)+ 1 / params.dr**2

            b[i] = params.S/ params.D

    return np.linalg.solve(A,b)

def solveur_centre(params):
    """ Fonction permettant de résoudre le problème numériquement en fonction des paramsètres."""

    ## Création des matrices A et b (Ax = b)
    A = np.zeros((params.nPts,params.nPts))
    b = np.zeros(params.nPts)

    for i in range(params.nPts):
        ri = params.pos[i]

        # Condition limite de neumann en r=0
        if i == 0:
            ## Ici, on a une Gear Avant pour la dérivée première
            ## En effet, l'équation fournit dans le devoir ne permet pas
            ## d'évaluer en r=0 avec une différence centrée
            A[i,i] = -3 / (2*params.dr)
            A[i,i+1] = 4 / (2*params.dr)
            A[i,i+2] = -1 / (2*params.dr)

            b[i] = 0

        # Condition limite de dirichlet en r=R
        elif i == params.nPts-1:
            A[i,i] = 1
            b[i] = params.Ce

        # Milieu du domaine
        else:
            A[i,i-1] = 1 / params.dr**2 - 1 / (2 * ri * params.dr)
            A[i,i] = - 2 / params.dr**2
            A[i,i+1] = 1 / (2 * ri * params.dr) + 1 / params.dr**2

            b[i] = params.S/ params.D

    return np.linalg.solve(A,b)

def analytique(params):
    """Fonction renvoyant le vecteur solution analytique au probleme en fonction des parametres."""
    r = sp.Symbol('r')
    func = params.S / (4 * params.D) * ( r**2 - params.R**2) + params.Ce 
    anal = sp.lambdify(r, func, 'numpy')

    return anal(params.pos)

if __name__ == "__main__":

    # Définition des paramètres
    nPts = 5
    R = 0.5
    nTime = 10
    endTime = 4e09
    D = 1e-10
    k = 4e-09
    Ce = 20

    # Création de l'objet params qui contient tous les paramètres
    # on peut passer l'objet params aux fonctions qui en ont besoin
    params = Parametres(nPts=nPts,nTime=nTime,endTime=endTime,R=R,D=D,k=k,Ce=Ce)
    sim = solveur_avant(params)
    ana = analytique(params)


    plt.figure()
    plt.plot(params.pos, sim, "x-", label="solution numérique")
    plt.plot(params.pos, ana, "o-", label="solution analytique")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel("C(r)")
    plt.title("Comparaison entre la solution numérique et la solution analytique pour un schéma avant")
    plt.grid()
    plt.show()

    print(f"Erreur L1 avec {params.nPts} points :", normeL1(ana, sim))
    print(f"Erreur L2 avec {params.nPts} points :", normeL2(ana, sim))
    print(f"Erreur Linf avec {params.nPts} points :", normeLinf(ana, sim))


    # Schéma centré
    sim = solveur_centre(params)

    plt.figure()
    plt.plot(params.pos, sim, "x-", label="solution numérique")
    plt.plot(params.pos, ana, "o-", label="solution analytique")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel("C(r)")
    plt.title("Comparaison entre la solution numérique et la solution analytique pour un schéma avant")
    plt.grid()
    plt.show()




    

