import numpy as np
import matplotlib.pyplot as plt
import solveur_permanent


def solve(solveur, norme):
    """Test de l'ordre de convergence pour le solveur avec schéma centré."""
    nPts = 5
    R = 1.0
    D = 1e-10
    S = 1e-8
    Ce = 10

    params = solveur_permanent.Parametres(nPts=nPts, R=R, D=D, S=S, Ce=Ce)
    sim = solveur(params)
    ana = solveur_permanent.analytique(params)

    # Vérifier que la solution numérique est proche de la solution analytique

    N = 10  # Nombre de raffinements
    L = np.empty(N)
    for i in range(N):
        params.nPts = nPts
        sim = solveur(params)
        ana = solveur_permanent.analytique(params)
        L[i] = norme(ana,sim)
        nPts *= 2

    ordre = np.log(L[:-1]/L[1:])/np.log(2)

    k = np.arange(N)
    n = 5 * ( 2**k)
    pas = R/(n-1)

    return pas, L, ordre


if __name__ == "__main__":

    ## Schéma avant
    pasL1, L1, ordreL1 = solve(solveur_permanent.solveur_avant, solveur_permanent.normeL1)
    pasL2, L2, ordreL2 = solve(solveur_permanent.solveur_avant, solveur_permanent.normeL2)
    pasLinf, Linf, ordreLinf = solve(solveur_permanent.solveur_avant, solveur_permanent.normeLinf)
    text = (f"Ordre de convergence L1 : {ordreL1[-1]:.7f} \n"
            f"Ordre de convergence L2: {ordreL2[-1]:.7f} \n"
            f"Ordre de convergence Linf: {ordreLinf[-1]:.7f} \n")
    
    plt.figure()
    plt.loglog(pasL1,L1,"x-", label="L1")
    plt.loglog(pasL2,L2,"x-", label="L2")
    plt.loglog(pasLinf,Linf,"x-", label="Linf")
    plt.legend()
    plt.ylabel("erreur")
    plt.title("Erreur de la solution numérique en fonction du pas de discrétisation pour un schéma avant")
    plt.grid("both", ls="--")
    plt.text(0.7, 0.2, text, fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()

    ## Schéma centré
    pasL1, L1, ordreL1 = solve(solveur_permanent.solveur_centre, solveur_permanent.normeL1)
    pasL2, L2, ordreL2 = solve(solveur_permanent.solveur_centre, solveur_permanent.normeL2)
    pasLinf, Linf, ordreLinf = solve(solveur_permanent.solveur_centre, solveur_permanent.normeLinf)
    text = (f"Ordre de convergence L1 : {ordreL1[-1]:.7f} \n"
            f"Ordre de convergence L2: {ordreL2[-1]:.7f} \n"
            f"Ordre de convergence Linf: {ordreLinf[-1]:.7f} \n")

    plt.figure()
    plt.loglog(pasL1,L1,"x-", label="L1")
    plt.loglog(pasL2,L2,"x-", label="L2")
    plt.loglog(pasLinf,Linf,"x-", label="Linf")
    plt.legend()
    plt.xlabel("pas de discrétisation")
    plt.ylabel("erreur")
    plt.title("Erreur de la solution numérique en fonction du pas de discrétisation pour un schéma centré")
    plt.grid("both", ls="--")
    plt.text(0.3, 0.2, text, fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()
