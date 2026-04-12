# Importation des modules
import numpy as np
import sys
import sympy as sp

# Assignation des paramètres utilisés pour l'analyse
class Parametres:
    
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    # Attention de bien utiliser une combinaison de paramètre qui permet de converger
    # Voir test_convergence.py pour choisir une bonne combinaison de paramètres. Exemple de bons paramètres 
    # EX1 : nr et nz = 10, dt = 0.25
    # EX2 : nr et nz = 5, dt = 1.75
    # EX3 : nr et nz = 3, dt = 1.25
    # !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
    
    def __init__(
        self,
        *,
        epsilon: float = 0.525,
        nr: int = 20,
        nz: int = 20,
        dt: float = 0.6,
        h: float = 10.0,
        k: float = 44.5,
        t_fin: float = 600,
        solution_MMS_sympy = sp.sympify(None),
        verbose: bool = True      # ← nouveau : désactiver les prints en mode MC
    ):

        self.Rmax = 0.0127      # [m] Rayon
        self.H = 0.05           # [m] Hauteur
        self.rho = 7850         # [kg/m^3] Densité
        self.Cp = 448           # [J/kg*K] Capacité thermique massique
        self.T_inf = 298.15     # [K] Température à l'infini
        self.sigma = 5.670374e-8  # Constante de Stefan-Boltzmann [W/m^2*K^4]
        self.T_four = 1118.15   # Température initiale en Kelvin
        self.time = 0           # temps initialisé à 0
 
        self.h = h              # [W/m^2*K] Coefficient de convection
        self.epsilon = epsilon  # [-] Emissivité
        self.k = k              # [W/m*K] Conductivité thermique
        self.nr = nr            # [-] Nombre de points selon r
        self.nz = nz            # [-] Nombre de points selon z
        self.t_fin = t_fin      # [s] Temps d'arrêt de la simulation
        self.dt = dt            # [s] Pas de temps

        self.dr = self.Rmax / (self.nr - 1)  # Pas dans la direction r [m]
        self.dz = (self.H/2) / (self.nz - 1) # Pas dans la direction z [m]

        # Vérification du critère de stabilité pour la méthode euler explicite
        cste = ((self.k * self.dt) / (self.rho * self.Cp))
        explicite = 1 - cste*(2/self.dr**2) - cste*(2/self.dz**2)
        dt_max = (self.rho * self.Cp) / (self.k * (2/self.dr**2 + 2/self.dz**2))

        if verbose:
            print('Vérification du critère de stabilité pour la méthode euler explicite :')
            print(f"Dt maximal autorisé pour les paramètres d'entrée: {dt_max:.6f} [s]")
            print(f'Dt utilisé : {self.dt:.6f} [s]')

        if explicite > 0:
            if verbose:
                print('Le critère de stabilité pour la méthode euler explicite est respecté.')
                print('')
        else:
            if verbose:
                print("Le critère de stabilité pour la méthode euler explicite n'est pas respecté."
                      " Il faut ajuster les valeurs de dt en conséquence.")
                if input("Voulez-vous continuer avec le dt actuel ? (y/n) :").lower() != 'y':
                    if input("Voulez-vous plutôt poursuivre avec le dt maximal autorisé ? (y/n) :").lower() == 'y':
                        self.dt = dt_max
                        print(f"Dt ajusté à {self.dt:.6f} [s]")
                    else:
                        print("Arrêt du programme.")
                        sys.exit()
                else:
                    print('Continuons avec le dt actuel, mais soyez conscient que cela peut entraîner des résultats instables.')
                    print('')
            else:
                # En mode Monte Carlo : on force dt_max si instable pour éviter divergence
                if explicite <= 0:
                    self.dt = dt_max

        # Calcul des matrices de position
        self.Z, self.R = Position(self)
        self.Rmin = np.min(self.R)

        # Setting the MMS solution if enabled
        if solution_MMS_sympy is None:
            r_var, z_var, t_var = sp.symbols('r z t')
            self.solution_MMS_sympy = sp.sympify(0)
            self.solution_MMS = sp.lambdify((r_var, z_var, t_var), sp.sympify(0), 'numpy')
            self.source = np.zeros_like(self.R)
            self.MMS = False
        else:
            r_var, z_var, t_var = sp.symbols('r z t')
            symbols = (r_var, z_var, t_var)
            self.solution_MMS_sympy = solution_MMS_sympy
            self.solution_MMS = sp.lambdify(symbols, self.solution_MMS_sympy)
            self.solution_MMS_diff_r = sp.lambdify(symbols, sp.diff(self.solution_MMS_sympy, r_var))
            self.solution_MMS_diff_z = sp.lambdify(symbols, sp.diff(self.solution_MMS_sympy, z_var))
            self.solution_MMS_diff_diff_r = sp.lambdify(symbols, sp.diff(self.solution_MMS_sympy, r_var, 2))
            self.solution_MMS_diff_diff_z = sp.lambdify(symbols, sp.diff(self.solution_MMS_sympy, z_var, 2))
            self.solution_MMS_diff_t = sp.lambdify(symbols, sp.diff(self.solution_MMS_sympy, t_var, 1))
            self.source = self.rho * self.Cp * self.solution_MMS_diff_t(self.R, self.Z, self.time) - \
                          self.k * self.solution_MMS_diff_diff_r(self.R, self.Z, self.time) - \
                          self.k * (1/self.R) * self.solution_MMS_diff_r(self.R, self.Z, self.time) - \
                          self.k * self.solution_MMS_diff_diff_z(self.R, self.Z, self.time)
            self.MMS = True

    def Biot(self):
        return (self.h * 2 * self.Rmax / self.k)

    def Time(self, dt):
        self.time += dt
        self.update_source_MMS()
        return self.time

    def update_source_MMS(self):
        if self.MMS:
            self.source = self.rho * self.Cp * self.solution_MMS_diff_t(self.R, self.Z, self.time) - \
                self.k * self.solution_MMS_diff_diff_r(self.R, self.Z, self.time) - \
                self.k * (1/self.R) * self.solution_MMS_diff_r(self.R, self.Z, self.time) - \
                self.k * self.solution_MMS_diff_diff_z(self.R, self.Z, self.time)


# ------------------------------------------------------------------------------
# FONCTIONS UTILISÉES DANS L'ANALYSE

def Position(prm):
    """ Fonction générant deux matrices de discrétisation de l'espace. """
    rvector = np.linspace(0, prm.Rmax, prm.nr)
    zvector = np.linspace(0, prm.H/2, prm.nz)
    r = np.zeros([prm.nr, prm.nz])
    z = np.zeros([prm.nz, prm.nr])
    for i in range(prm.nr):
        r[i] = np.full(prm.nz, rvector[-1-i])
    for i in range(prm.nz):
        z[i] = np.full(prm.nr, zvector[i])
    z = np.transpose(z)
    return z, r


def Milieu(prm, T_tdt_middle):
    """ Calcule la température au milieu du cylindre à un instant t+dt. """
    cste = ((prm.k * prm.dt) / (prm.rho * prm.Cp))
    T_tdt = T_tdt_middle.copy()
    for r in range(prm.nr):
        for z in range(prm.nz):
            if r != 0 and z != 0 and r != prm.nr-1 and z != prm.nz-1:
                dist = prm.R[r, z] - np.min(prm.R)
                T_tdt[r, z] = cste*((T_tdt_middle[r-1, z]-2*T_tdt_middle[r, z]+T_tdt_middle[r+1, z])/(prm.dr**2)
                                    + (1/(2*prm.dr*dist))*(T_tdt_middle[r-1, z]-T_tdt_middle[r+1, z])
                                    + (T_tdt_middle[r, z-1]-2*T_tdt[r, z]+T_tdt_middle[r, z+1])/(prm.dz**2)) \
                              + (T_tdt_middle[r, z]) \
                              + prm.source[r, z] * prm.dt / (prm.rho * prm.Cp)
            else:
                pass
    return T_tdt


def Temperature(prm, T_t):
    """ Calcule la température aux frontières du domaine à un instant t+dt. """
    T_tdt = T_t.copy()
    if not prm.MMS:
        conv_rad = prm.h * (T_t - prm.T_inf) + \
                   prm.epsilon * prm.sigma * (T_t**4 - prm.T_inf**4)
        for r in range(prm.nr):
            for z in range(prm.nz):
                if r == 0:
                    T_tdt[r, z] = (1/3) * (-T_t[r+2, z]
                                            + 4 * T_t[r+1, z]
                                            - (2*prm.dr * conv_rad[r, z] / prm.k))
                elif z == prm.nz-1 or (z == prm.nz-1 and r == prm.nr-1):
                    T_tdt[r, z] = (1/3) * (-T_t[r, z-2]
                                            + 4 * T_t[r, z-1]
                                            - (2*prm.dz * conv_rad[r, z] / prm.k))
                elif z == 0 or (z == 0 and r == prm.nr-1):
                    T_tdt[r, z] = (4/3) * T_t[r, z+1] - (1/3) * T_t[r, z+2]
                elif r == prm.nr-1:
                    T_tdt[r, z] = (4/3) * T_t[r-1, z] - (1/3) * T_t[r-2, z]
    else:
        for r in range(prm.nr):
            for z in range(prm.nz):
                T_hat = prm.solution_MMS(prm.R[r, z], prm.Z[r, z], prm.time+prm.dt)
                if r == 0:
                    T_tdt[r, z] = T_hat
                elif z == prm.nz-1 or (z == prm.nz-1 and r == prm.nr-1):
                    T_tdt[r, z] = T_hat
                elif z == 0 or (z == 0 and r == prm.nr-1):
                    T_tdt[r, z] = T_hat
                elif r == prm.nr-1:
                    T_tdt[r, z] = T_hat
    T_tdt = Milieu(prm, T_tdt)
    return T_tdt


def normeL1(ana, sim, params):
    L1 = 0
    r_weights = params.R - params.Rmin
    rayon = params.Rmax - params.Rmin
    for time in range(ana.shape[0]):
        weighted_error = np.abs(ana[time, :, :] - sim[time, :, :]) * r_weights
        L1 += np.sum(weighted_error) * params.dr * params.dz * params.dt
    L1Final = np.sum(weighted_error) * params.dr * params.dz / (rayon * (params.H/2))
    domaine = rayon * (params.H/2) * params.t_fin
    L1SpatioTemporel = L1 / domaine
    return L1SpatioTemporel, L1Final


def normeL2(ana, sim, params):
    L2 = 0
    r_weights = params.R - params.Rmin
    rayon = params.Rmax - params.Rmin
    for time in range(ana.shape[0]):
        error = ana[time, :, :] - sim[time, :, :]
        weighted_error_sq = error**2 * r_weights
        L2 += np.sum(weighted_error_sq) * params.dr * params.dz * params.dt
    L2Final = np.sqrt(np.sum(weighted_error_sq) * params.dr * params.dz / (rayon * (params.H/2)))
    domaine = rayon * (params.H/2) * params.t_fin
    L2SpatioTemporel = np.sqrt(L2 / domaine)
    return L2SpatioTemporel, L2Final


def normeLinf(ana, sim, params):
    LinfSpatioTemporel = np.max(np.abs(ana - sim))
    LinfFinal = np.max(np.abs(ana[-1, :, :] - sim[-1, :, :]))
    return LinfSpatioTemporel, LinfFinal


# ==============================================================================
# QUANTITÉ D'INTÉRÊT (QOI)
# ==============================================================================

def QOI(T, prm):
    """
    Quantité d'intérêt (QOI) utilisée pour l'analyse d'incertitude ASME V&V 20.

    Température moyenne sur la surface extérieure du cylindre (r = Rmax, i.e. r == 0)
    pondérée en coordonnées cylindriques (intégrale en z sur H/2).

    Args:
        T   : np.array (nr x nz) — champ de température final
        prm : Parametres

    Returns:
        float : T_moy_surface [K]
    """
    # Ligne r=0 correspond à Rmax (la surface extérieure)
    T_surface = T[0, :]                          # (nz,)
    z_vec     = prm.Z[0, :]                      # positions z le long de la surface
    # Intégration trapézoidale sur z
    try:
        T_moy = np.trapezoid(T_surface, z_vec) / (prm.H / 2)  # NumPy >= 2.0
    except AttributeError:
        T_moy = np.trapz(T_surface, z_vec) / (prm.H / 2)      # NumPy < 2.0
    return T_moy


# ==============================================================================
# SIMULATION UNIQUE
# ==============================================================================

def run_simulation(epsilon, k, nr, nz, dt, t_fin, verbose=False):
    """
    Lance une simulation complète et retourne la QOI.

    Args:
        epsilon  : émissivité
        k        : conductivité thermique [W/m·K]
        nr, nz   : discrétisation spatiale
        dt       : pas de temps [s]
        t_fin    : temps final [s]
        verbose  : afficher les prints de Parametres

    Returns:
        float : valeur de la QOI (température moyenne en surface à t_fin)
    """
    prm = Parametres(epsilon=epsilon, k=k, nr=nr, nz=nz, dt=dt, t_fin=t_fin, verbose=verbose)
    T_t = np.full((prm.nr, prm.nz), prm.T_four)

    while prm.time < prm.t_fin:
        T_t = Temperature(prm, T_t)
        prm.Time(prm.dt)

    return QOI(T_t, prm)


# ==============================================================================
# MONTE CARLO — u_input selon ASME V&V 20
# ==============================================================================

def monte_carlo_uinput(
    nr, nz, dt, t_fin,
    epsilon_nom, epsilon_std,
    k_nom,       k_std,
    N=200,
    seed=None,
    verbose_mc=True
):
    """
    Calcule l'incertitude d'entrée u_input selon ASME V&V 20 §4
    par propagation Monte Carlo sur les variables epsilon et k.

    Hypothèses :
        - epsilon ~ Normal(epsilon_nom, epsilon_std)
        - k       ~ Normal(k_nom,       k_std)
        - Les deux variables sont indépendantes.

    Selon ASME V&V 20, u_input est l'écart-type de la distribution
    de la QOI obtenue en propageant les incertitudes des entrées :

        u_input = std( QOI(epsilon_i, k_i) )    i = 1…N

    Args:
        nr, nz          : discrétisation spatiale
        dt              : pas de temps [s]
        t_fin           : temps final [s]
        epsilon_nom     : valeur nominale de l'émissivité
        epsilon_std     : écart-type de l'émissivité  (u_epsilon)
        k_nom           : valeur nominale de k [W/m·K]
        k_std           : écart-type de k              (u_k)
        N               : nombre de tirages Monte Carlo
        seed            : graine aléatoire (reproductibilité)
        verbose_mc      : afficher la progression

    Returns:
        dict avec :
            'u_input'       : incertitude d'entrée combinée [K]
            'QOI_nominal'   : QOI avec les valeurs nominales [K]
            'QOI_samples'   : tableau des N valeurs de QOI [K]
            'epsilon_samples': tirages d'émissivité
            'k_samples'     : tirages de conductivité
            'mean_QOI'      : moyenne des QOI simulées [K]
            'std_QOI'       : écart-type des QOI simulées (= u_input) [K]
    """
    rng = np.random.default_rng(seed)

    # --- Tirages selon distributions normales ---
    epsilon_samples = rng.normal(loc=epsilon_nom, scale=epsilon_std, size=N)
    k_samples       = rng.normal(loc=k_nom,       scale=k_std,       size=N)

    # --- QOI nominale ---
    if verbose_mc:
        print("=" * 60)
        print("  ANALYSE D'INCERTITUDE — Monte Carlo (ASME V&V 20)")
        print("=" * 60)
        print(f"  Paramètres d'entrée :")
        print(f"    epsilon  : {epsilon_nom} ± {epsilon_std}  (1σ)")
        print(f"    k        : {k_nom} ± {k_std} W/m·K  (1σ)")
        print(f"  Nombre de tirages N = {N}")
        print("-" * 60)
        print("  Calcul de la QOI nominale…")

    QOI_nominal = run_simulation(epsilon_nom, k_nom, nr, nz, dt, t_fin, verbose=False)

    if verbose_mc:
        print(f"  QOI nominale = {QOI_nominal:.4f} K")
        print("-" * 60)
        print("  Propagation Monte Carlo en cours…")

    # --- Boucle Monte Carlo ---
    QOI_samples = np.zeros(N)
    for i in range(N):
        QOI_samples[i] = run_simulation(
            epsilon_samples[i], k_samples[i],
            nr, nz, dt, t_fin, verbose=False
        )
        if verbose_mc and (i + 1) % max(1, N // 10) == 0:
            print(f"    Tirage {i+1:>4d}/{N}  —  QOI = {QOI_samples[i]:.4f} K")

    # --- u_input : écart-type de la distribution des QOI (ASME V&V 20 Eq. 4-1) ---
    mean_QOI = np.mean(QOI_samples)
    std_QOI  = np.std(QOI_samples, ddof=1)   # ddof=1 : estimateur sans biais
    u_input  = std_QOI

    if verbose_mc:
        print("=" * 60)
        print("  RÉSULTATS")
        print("=" * 60)
        print(f"  QOI nominale              = {QOI_nominal:.6f} K")
        print(f"  Moyenne QOI (MC)          = {mean_QOI:.6f} K")
        print(f"  Écart-type QOI (MC)       = {std_QOI:.6f} K")
        print(f"  u_input (ASME V&V 20)     = {u_input:.6f} K")
        print(f"  Intervalle 95 % (±2σ)     = [{mean_QOI - 2*u_input:.4f}, {mean_QOI + 2*u_input:.4f}] K")
        print("=" * 60)

    return {
        'u_input':         u_input,
        'QOI_nominal':     QOI_nominal,
        'QOI_samples':     QOI_samples,
        'epsilon_samples': epsilon_samples,
        'k_samples':       k_samples,
        'mean_QOI':        mean_QOI,
        'std_QOI':         std_QOI,
    }
