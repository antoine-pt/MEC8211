import numpy as np
import sympy as sp
import matplotlib.pyplot as plt


class Parametres():
    """Classe contenant les parametres du problème."""
    def __init__(self, nPts, nTime, endTime, R, D, k, Ce, solution=None, mmsON = True):


        self.R = R

        self.nTime = nTime
        self.endTime = endTime
        self.time = 0

        self.D = D
        self.k = k
        self.Ce = Ce

        self._r, self._t = sp.symbols('r t', real=True)

        self._solution = None
        self.nPts = nPts

        if solution is not None:
            self.solution = solution

        self.mmsON = mmsON
        
    @property
    def dr(self):
        return self.R / (self.nPts - 1)
    
    @property
    def pos(self):
        return np.linspace(0, self.R, self.nPts)
    
    @property
    def dt(self):
        return self.endTime / self.nTime
    
    @property
    def nPts(self):
        return self._nPts
    @nPts.setter
    def nPts(self, value):
        self._nPts = value
        self._solution = np.zeros(self._nPts)

    # Pour s'assurer que la nouvelle solution a la bonne taille et qu'on a pas fait d'erreur
    @property
    def solution(self):
        return self._solution
    @solution.setter
    def solution(self, value):
        array = np.asarray(value)
        if array.size != self.nPts:
            raise ValueError("La solution doit être un tableau de la même taille que nPts.")
        self._solution = array

    @property
    def mms(self):
        """Renvoie la solution symbolique de la méthode manufacturée."""
        return self._mms
    
    @property
    def mmsON(self):
        """Renvoie si la méthode manufacturée est activée ou non."""
        return self._mms is not None
    @mmsON.setter
    def mmsON(self, value = None):
        if value is None or value == False:
            self._mms = None
        else:
            self._mms = sp.cos(2 * sp.pi * self._r / self.R) * sp.exp(-self._t / (self.endTime/10))


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

# def solveur_avant(params):
#     """ Fonction permettant de résoudre le problème numériquement en fonction des paramsètres."""

#     ## Création des matrices A et b (Ax = b)
#     A = np.zeros((params.nPts,params.nPts))
#     b = np.zeros(params.nPts)

#     for i in range(params.nPts):
#         ri = params.pos[i]

#         # Condition limite de neumann en r=0
#         if i == 0:
#             A[i,i] = -1
#             A[i,i+1] = 1

#             b[i] = 0

#         # Condition limite de dirichlet en r=R
#         elif i == params.nPts-1:
#             A[i,i] = 1
#             b[i] = params.Ce

#         # Milieu du domaine
#         else:
#             A[i,i-1] = 1 / params.dr**2
#             A[i,i] = -1 / (params.dr*ri)- 2 / params.dr**2
#             A[i,i+1] = 1 / (params.dr*ri)+ 1 / params.dr**2

#             b[i] = params.S/ params.D

#     return np.linalg.solve(A,b)

def solveur_centre(params):
    """ Fonction permettant de résoudre le problème numériquement en fonction des paramsètres."""

    ## Création des matrices A et b (Ax = b)
    A = np.zeros((params.nPts,params.nPts))
    b = np.zeros(params.nPts)

    r, t = params._r, params._t

    ## Paramètres mms

    # Si la MMS est activée, on calcule les termes
    try:
        C_mms_dr = sp.diff(params.mms, r)
        C_mms_drr = sp.diff(params.mms, r, 2)
        C_mms_dt = sp.diff(params.mms, t)

        source_term = sp.lambdify((r,t) , C_mms_dt - params.D * (C_mms_dr / params._r + C_mms_drr) + params.k * params.mms)
        C_mms = sp.lambdify((r,t), params.mms)
        C_mms_dr = sp.lambdify((r,t), C_mms_dr)

    # Sinon, les termes sont nuls et on utilise les conditions du problème
    except:
        source_term = lambda r,t: 0
        C_mms = lambda r,t: 0
        C_mms_dr = lambda r,t: 0


    for i in range(params.nPts):
        ri = params.pos[i]

        # Condition limite de neumann en r=0
        if i == 0:
            ## Ici, on a une Gear Avant pour la dérivée première
            ## En effet, l'équation fournit dans le devoir ne permet pas
            ## d'évaluer en r=0 avec une différence centrée
            A[i,i] = -3 / (2*params.dr)
            A[i,i+1] = 2 / (params.dr)
            A[i,i+2] = -1 / (2*params.dr)

            # Ici, comme on a condition de symétrie imposée, il est bien important de 
            # choisir une MMS qui a une dérivée nulle en (r=0,t)
            b[i] = 0

        # Condition limite de dirichlet en r=R
        elif i == params.nPts-1:
            A[i,i] = 1

            # Si MMS imposée
            if params.mms is not None:
                b[i] = C_mms(ri,params.time+params.dt)
            # Condition de Dirichlet du probleme, Ce = 20
            else:
                b[i] = params.Ce

        # Milieu du domaine
        else:
            A[i,i-1] = params.D * params.dt * ((1/(ri * 2 * params.dr)) - (1/(params.dr**2)))
            A[i,i] = 2 * params.D * params.dt / (params.dr**2) + params.k * params.dt + 1
            A[i,i+1] = (-1) * params.D * params.dt * ((1/(ri * 2 * params.dr)) + (1/(params.dr**2)))

            # Pour rappel, source_term est nul si MMS n'est pas activée
            b[i] = params.solution[i] + source_term(ri,params.time + params.dt) * params.dt
        
    params.solution = np.linalg.solve(A,b)
    params.time += params.dt
    return params.solution


if __name__ == "__main__":

    # Définition des paramètres
    nPts = 9
    R = 0.5
    nTime = 100
    endTime = 4e9
    D = 1e-10
    k = 4e-09
    Ce = 20

    # Création de l'objet params qui contient tous les paramètres
    # on peut passer l'objet params aux fonctions qui en ont besoin
    params = Parametres(nPts=nPts,nTime=nTime,endTime=endTime,R=R,D=D,k=k,Ce=Ce)
    for t in range(params.nTime):
        sim = solveur_centre(params)

    try:
        C_mms = sp.lambdify(["r","t"],params.mms)
        ana = C_mms(params.pos, params.endTime)
    except:
        ana = np.zeros(nPts)

    plt.figure()
    plt.plot(params.pos, sim, "x-", label="solution numérique")
    plt.plot(params.pos, ana, "o-", label="solution MMS")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel("C(r)")
    plt.title("Comparaison entre la solution manufacturée et la solution analytique pour un schéma centré.")
    plt.grid()
    plt.show()

    params.mmsON = False
    params.nPts = 11
    for t in range(params.nTime):
        sim = solveur_centre(params)

    plt.figure()
    plt.plot(params.pos, sim, "x-", label="solution numérique")
    plt.legend()
    plt.xlabel("r")
    plt.ylabel("C(r)")
    plt.title("Solution numérique du problème de Mme D'AVIGNON pour un schéma centré, 11 points.")
    plt.grid()
    plt.show()




    

