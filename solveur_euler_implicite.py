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
            # self._mms = sp.cos(2 * sp.pi * self._r / self.R)
            # self._mms = sp.cos(2 * sp.pi * self._r / self.R) * sp.exp(-self._t / (self.endTime/10))
            self._mms = 1 + sp.cos(2 * sp.pi * self._r / self.R) * sp.cos(self._t * 2 * sp.pi/self.endTime)

    def printself(self):
        """Affiche les attributs principaux de l'objet pour debug."""
        print("Paramètres courants :")
        print(f"  nPts    = {self.nPts}")
        print(f"  nTime   = {self.nTime}")
        print(f"  endTime = {self.endTime}")
        print(f"  R       = {self.R}")
        print(f"  D       = {self.D}")
        print(f"  k       = {self.k}")
        print(f"  Ce      = {self.Ce}")
        print(f"  time    = {self.time}")
        print(f"  dr      = {self.dr}")
        print(f"  dt      = {self.dt}")
        print(f"  mmsON   = {self.mmsON}")


def normeL1(ana, sim, params):
    """ Calcule la norme L1 entre une solution MMS analytique et une solution
    numérique. Norme calculée en temps et espace!

    Args:
        ana (np.array): solution MMS
        sim (np.array): solution numérique

    Returns:
        float: norme L1 entre les deux solutions
    """
    L1 = 0
    for time in range(ana.shape[0]):
        error = np.abs(ana[time,:] - sim[time,:])

        # params.pos**2 pour pondérer l'erreur en fonction de la positino sur 
        # le rayon (domaine cylindrique)
        L1 += np.sum(error)/ana.shape[1]
    return L1/ana.shape[0]

def normeL2(ana, sim, params):
    """ Calcule la norme L2 entre une solution MMS analytique et une solution
    numérique. Norme calculée en temps et espace!

    Args:
        ana (np.array): solution MMS
        sim (np.array): solution numérique

    Returns:
        float: norme L2 entre les deux solutions
    """
    L2 = 0
    for time in range(ana.shape[0]):
        error = ana[time,:] - sim[time,:]

        # Pondération cylindrique: intégrale de u² * r dr
        # Pour r=0, utiliser la règle du trapèze qui évite singularité
        r_weights = params.pos  # valeurs de r
        weighted_error_sq = error**2 * r_weights
        L2 += np.sum(weighted_error_sq) * params.dr
    
    # Normalisation par le "volume" cylindrique total et le temps
    # Intégrale de r dr de 0 à R = R²/2
    total_measure = 0.5 * params.R**2 * params.endTime
    return np.sqrt(L2 / total_measure)

def normeLinf(ana, sim, params):
    """ Calcule la norme Linf entre une solution MMS analytique et une solution
    numérique. Norme calculée en temps et espace!

    Args:
        ana (np.array): solution MMS
        sim (np.array): solution numérique

    Returns:
        float: norme Linf entre les deux solutions
    """

    return np.max(np.abs(ana - sim))


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

        # Condition limite de neumann en r=0 (GEAR forward scheme)
        if i == 0:
            ## Ici, on a une Gear Avant pour la dérivée première
            ## En effet, l'équation fournit dans le devoir ne permet pas
            ## d'évaluer en r=0 avec une différence centrée
            A[i,i] = -3 / (2*params.dr)
            A[i,i+1] = 2 / (params.dr)
            A[i,i+2] = -1 / (2*params.dr)
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
            if ri == 0:
                raise ValueError("Erreur : r ne doit pas être égal à 0 dans le milieu du domaine.")
            b[i] = params.solution[i] + source_term(ri,params.time + params.dt) * params.dt
        
    params.solution = np.linalg.solve(A,b)
    params.time += params.dt
    return params.solution


if __name__ == "__main__":
 
    # Définition des paramètres
    nPts = 32
    R = 0.5
    endTime = 0.1
    nTime = 1000
    # endTime = 4e9
    # D = 1e-10
    # k = 4e-09

    D = 1
    k = 4
    Ce = 20

    # Création de l'objet params qui contient tous les paramètres
    # on peut passer l'objet params aux fonctions qui en ont besoin
    params = Parametres(nPts=nPts,nTime=nTime,endTime=endTime,R=R,D=D,k=k,Ce=Ce)
    

    ana = np.zeros((params.nTime,params.nPts))
    sim = np.zeros((params.nTime,params.nPts))
    C_mms = sp.lambdify(["r","t"],params.mms)

    sim[0,:] = C_mms(params.pos, params.dt * 0)
    ana[0,:] = C_mms(params.pos, params.dt * 0)

    for t in range(params.nTime-1):
        sim[t+1,:] = solveur_centre(params)
        ana[t+1,:] = C_mms(params.pos, params.time)


    print("\n")
    print("L1 =", normeL1(ana, sim, params))
    print("L2 =", normeL2(ana, sim, params))
    print("Linf =", normeLinf(ana, sim, params))
    print("nPts =", params.nPts)
    print("dr =", params.dr)
    print("dt =", params.dt)
    print("\n")



    # for i in range(params.nTime-1):
    #     if i%100 == 0:
    #         print(i)
    #         plt.figure()
    #         plt.plot(params.pos, sim[i,:], "x-", label="solution numérique")
    #         plt.plot(params.pos, ana[i,:], "o-", label="solution MMS")
    #         plt.legend()
    #         plt.xlabel("r")
    #         plt.ylabel("C(r)")
    #         plt.title("Comparaison entre la solution manufacturée et la solution analytique pour un schéma centré.")
    #         plt.grid()
    #         plt.show()
    #         # plt.close()
    # print("L2_final =", L2_final(ana,sim,params))
    # params.mmsON = False
    # params.nPts = 11
    # for t in range(params.nTime):
    #     sim = solveur_centre(params)

    # plt.figure()
    # plt.plot(params.pos, sim, "x-", label="solution numérique")
    # plt.legend()
    # plt.xlabel("r")
    # plt.ylabel("C(r)")
    # plt.title("Solution numérique du problème de Mme D'AVIGNON pour un schéma centré, 11 points.")
    # plt.grid()
    # plt.show()
    # plt.close()




    

