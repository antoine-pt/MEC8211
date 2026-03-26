"""
Conversion Python 3 des scripts MATLAB : launch_simulationLBM.m, Generate_sample.m, LBM.m
Auteurs originaux : Sébastien Leclaire (2014), modifié par David Vidal

Parallélisation :
  - Generate_sample : remplissage de la grille entièrement vectorisé (NumPy, sans boucle Python)
  - LBM             : noyau de simulation compilé JIT par Numba avec parallel=True
                      → utilise automatiquement TOUS les cœurs disponibles via prange

Dépendances :
    pip install numpy pillow matplotlib numba
"""

import numpy as np
from scipy.stats import lognorm
from PIL import Image
import matplotlib.pyplot as plt
from numba import njit, prange


# ==============================================================================
# FONCTION : Generate_sample
# ==============================================================================

def Generate_sample(seed, filename, mean_d, std_d, poro, nx, dx):
    """
    Crée une structure 2D de fibres et l'exporte en format TIFF.

    Paramètres
    ----------
    seed     : int    - graine du générateur aléatoire (0 = aléatoire)
    filename : str    - nom du fichier TIFF de sortie
    mean_d   : float  - diamètre moyen des fibres [µm]
    std_d    : float  - écart-type des diamètres  [µm]
    poro     : float  - porosité cible
    nx       : int    - taille latérale du domaine [cellules]
    dx       : float  - taille d'une cellule [m]

    Retourne
    --------
    d_equivalent : float - diamètre équivalent [µm]
    """

    rng    = np.random.default_rng(None if seed == 0 else seed)
    dx_um  = dx * 1e6
    domain = nx * dx_um

    # -------------------------------------------------------------------------
    # Distribution des fibres
    # -------------------------------------------------------------------------
    dist_full = rng.normal(mean_d, std_d, 10000)

    nb_fiber     = 1
    poro_eff     = 1.0 - np.sum(dist_full[:nb_fiber] ** 2 / 4 * np.pi) / domain ** 2
    poro_eff_old = poro_eff

    while poro_eff >= poro:
        poro_eff_old = poro_eff
        nb_fiber    += 1
        poro_eff     = 1.0 - np.sum(dist_full[:nb_fiber] ** 2 / 4 * np.pi) / domain ** 2

    if abs(poro_eff - poro) > abs(poro_eff_old - poro):
        nb_fiber -= 1
        poro_eff  = poro_eff_old

    dist_d       = np.sort(dist_full[:nb_fiber])[::-1]
    d_equivalent = np.sum(dist_d ** 2) / np.sum(dist_d)
    print(f"d_equivalent     = {d_equivalent:.4f} µm")

    # -------------------------------------------------------------------------
    # Positionnement des fibres (sans chevauchement, conditions périodiques)
    # Vérification vectorisée sur les 9 images périodiques
    # -------------------------------------------------------------------------
    circles     = np.zeros((nb_fiber, 3))
    circles[0]  = [rng.random() * domain, rng.random() * domain, dist_d[0]]
    fiber_count = 1
    offsets     = np.array([0.0, domain, -domain])

    while fiber_count < nb_fiber:
        di = dist_d[fiber_count]
        xi = rng.random() * domain
        yi = rng.random() * domain

        xc = circles[:fiber_count, 0]
        yc = circles[:fiber_count, 1]
        dc = circles[:fiber_count, 2]
        r2 = (di + dc) ** 2                            # (fiber_count,)

        ox, oy = np.meshgrid(offsets, offsets, indexing='ij')
        ox = ox.ravel()                                  # (9,)
        oy = oy.ravel()

        # distances² : (fiber_count, 9)
        dx2 = (xi - xc[:, np.newaxis] + ox[np.newaxis, :]) ** 2
        dy2 = (yi - yc[:, np.newaxis] + oy[np.newaxis, :]) ** 2

        if np.any(dx2 + dy2 < r2[:, np.newaxis]):
            continue   # chevauchement → réessayer

        circles[fiber_count] = [xi, yi, di]
        fiber_count += 1

    print(f"number_of_fibres = {fiber_count}")

    # -------------------------------------------------------------------------
    # Remplissage de la grille — vectorisé NumPy (pas de boucle Python i,j)
    # -------------------------------------------------------------------------
    coords       = (0.5 + np.arange(nx)) * dx_um
    px, py       = np.meshgrid(coords, coords, indexing='ij')   # (nx, nx)
    poremat      = np.zeros((nx, nx), dtype=bool)

    xc = circles[:, 0]
    yc = circles[:, 1]
    r2 = (circles[:, 2] / 2) ** 2

    for k in range(nb_fiber):
        for ox in offsets:
            for oy in offsets:
                poremat |= (px - (xc[k] + ox)) ** 2 + (py - (yc[k] + oy)) ** 2 < r2[k]

    # -------------------------------------------------------------------------
    # Export TIFF et affichage
    # -------------------------------------------------------------------------
    # poremat est en convention (i=x, j=y) — on transpose pour l'export image
    # (lignes=y, cols=x), ce qui donne de vrais cercles à l'affichage.
    poremat_img = poremat.T   # (NY, NX) : lignes=y, colonnes=x
    Image.fromarray(poremat_img.astype(np.uint8) * 255).save(filename)

    return d_equivalent


# ==============================================================================
# NOYAU LBM COMPILÉ PAR NUMBA  (parallel=True → prange utilise tous les cœurs)
# ==============================================================================

@njit(parallel=True, cache=True)
def _lbm_step(N, SOLID, W, cx, cy, NX, NY, deltaP, dx, rho0, dt, OMEGA, bb_idx):
    """
    Un pas de temps LBM D2Q9 compilé JIT.

    parallel=True + prange → Numba distribue automatiquement les boucles
    sur tous les cœurs logiques disponibles (via OpenMP en arrière-plan).

    Retourne
    --------
    N_out    : ndarray (NX*NY, 9) - nouvelles fonctions de distribution
    ux_out   : ndarray (NX*NY,)   - vitesse x de chaque cellule
    FlowRate : float              - débit moyen sur la première rangée
    """
    NQ    = 9
    NCELL = NX * NY

    # ------------------------------------------------------------------
    # 1) Streaming périodique — parallélisé sur les directions q
    # ------------------------------------------------------------------
    N_stream = np.empty_like(N)
    N_stream[:, 0] = N[:, 0]   # direction au repos : aucun déplacement

    for q in prange(1, NQ):
        shift_x = int(cx[q])
        shift_y = int(cy[q])
        for idx in range(NCELL):
            i     = idx // NY
            j     = idx  % NY
            src_i = (i - shift_x) % NX
            src_j = (j - shift_y) % NY
            N_stream[idx, q] = N[src_i * NY + src_j, q]

    # ------------------------------------------------------------------
    # 2) Sauvegarde des nœuds solides (bounce-back avant collision)
    # ------------------------------------------------------------------
    N_solid_save = np.empty((NCELL, NQ), dtype=N.dtype)
    for idx in prange(NCELL):
        if SOLID[idx]:
            for q in range(NQ):
                N_solid_save[idx, q] = N_stream[idx, bb_idx[q]]

    # ------------------------------------------------------------------
    # 3) Moments macroscopiques + collision BGK — parallélisé sur NCELL
    # ------------------------------------------------------------------
    ux_out = np.empty(NCELL)

    for idx in prange(NCELL):
        rho_i = 0.0
        ux_i  = 0.0
        uy_i  = 0.0
        for q in range(NQ):
            f      = N_stream[idx, q]
            rho_i += f
            ux_i  += f * cx[q]
            uy_i  += f * cy[q]

        ux_i    = ux_i / rho_i + deltaP / (2.0 * NX * dx * rho0) * dt
        uy_i    = uy_i / rho_i
        ux_out[idx] = ux_i

        u2 = ux_i ** 2 + uy_i ** 2
        for q in range(NQ):
            cu  = ux_i * cx[q] + uy_i * cy[q]
            feq = rho_i * W[q] * (1.0 + 3.0 * cu + 4.5 * cu ** 2 - 1.5 * u2)
            N_stream[idx, q] += OMEGA * (feq - N_stream[idx, q])

    # ------------------------------------------------------------------
    # 4) Rétablissement des nœuds solides
    # ------------------------------------------------------------------
    for idx in prange(NCELL):
        if SOLID[idx]:
            for q in range(NQ):
                N_stream[idx, q] = N_solid_save[idx, q]

    # ------------------------------------------------------------------
    # 5) Débit sur la première rangée (indices 0..NY-1)
    # ------------------------------------------------------------------
    flow = 0.0
    for j in range(NY):
        flow += ux_out[j]
    flow /= (NX * dx)

    return N_stream, ux_out, flow


# ==============================================================================
# FONCTION : LBM
# ==============================================================================

def LBM(filename, NX, deltaP, dx, d_equivalent):
    """
    Calcule l'écoulement à travers le mat de fibres par la méthode LBM (D2Q9).

    Paramètres
    ----------
    filename     : str   - fichier TIFF de la structure de fibres
    NX           : int   - taille du domaine (carré NX×NX)
    deltaP       : float - chute de pression [Pa]
    dx           : float - taille d'une cellule [m]
    d_equivalent : float - diamètre équivalent des fibres [µm]
    """
    NY      = NX
    OMEGA   = 1.0
    rho0    = 1.0
    mu      = 1.8e-5
    epsilon = 1e-8

    dt = (1.0 / OMEGA - 0.5) * rho0 * dx ** 2 / 3.0 / mu

    # Lecture de la structure
    A     = np.array(Image.open(filename)).astype(bool)
    SOLID = A.flatten()

    # Paramètres D2Q9
    W  = np.array([4/9, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36, 1/9, 1/36])
    cx = np.array([0,   0,   1,    1,   1,    0,   -1,   -1,  -1  ], dtype=np.float64)
    cy = np.array([0,   1,   1,    0,  -1,   -1,   -1,    0,   1  ], dtype=np.float64)
    bb_idx = np.array([0, 5, 6, 7, 8, 1, 2, 3, 4], dtype=np.int64)

    N            = np.outer(np.ones(NX * NY), rho0 * W)
    FlowRate_old = 1.0
    FlowRate     = 0.0
    t_           = 1

    # Boucle temporelle
    while FlowRate == 0.0 or abs(FlowRate_old - FlowRate) / abs(FlowRate) >= epsilon:
        N, ux, FlowRate_new = _lbm_step(
            N, SOLID, W, cx, cy, NX, NY, deltaP, dx, rho0, dt, OMEGA, bb_idx
        )
        FlowRate_old = FlowRate
        FlowRate     = FlowRate_new
        t_          += 1

    # Résultats
    poro_eff = 1.0 - SOLID.sum() / (NX * NY)
    u_mean   = ux[:NY].mean()
    Re  = rho0 * u_mean * poro_eff * d_equivalent * 1e-6 / (mu * (1 - poro_eff))
    k   = u_mean * mu / deltaP * (NX * dx) * 1e12

    print(f"\nporo_eff         = {poro_eff:.6f}")
    print(f"Re               = {Re:.6e}")
    print(f"k_in_micron2     = {k:.6f} µm²")

    return k


# ==============================================================================
# SCRIPT PRINCIPAL
# ==============================================================================

if __name__ == "__main__":

    result = []
    nombre_echantillon = 10

    seed         = 0
    deltaP       = 0.1
    NX           = 75
    poro         = 0.9
    std_poro     = 7.5e-3
    mean_fiber_d = 12.5
    std_d        = 2.85
    dx           = 1e-6
    filename     = 'fiber_mat.tiff'
    liste_poro = np.random.normal(poro, std_poro, nombre_echantillon)

    for poro_i in liste_poro:
        
        d_equivalent = Generate_sample(seed, filename, mean_fiber_d, std_d, poro_i, NX, dx)
        k = LBM(filename, NX, deltaP, dx, d_equivalent)
        
        result.append(k)


    # Calcul des incertitudes en distribution lognormale

    result = np.array(result)

    # Paramètres log-normaux
    mu = np.mean(np.log(result))
    sigma = np.std(np.log(result), ddof=1)

    k_med = np.exp(mu)
    fvg = np.exp(sigma)

    k_min = np.exp(mu - sigma)
    k_max = np.exp(mu + sigma)

    # Comme l'incertitude est asymétrique, on prend la plus grande des deux
    u_input = max((k_max - k_med), (k_med - k_min))

    print("\n")
    print("===================================")
    print(f"Médiane : {k_med:.6e}")
    print(f"FVG : {fvg:.6e}")
    print(f"Intervalle 68% : [{k_min:.6e}, {k_max:.6e}]")
    print(f"Incertitude (u_input) : {u_input:.6e}")
    print("===================================")

    # =========================
    # PDF (histogramme + fit)
    # =========================

    # plot d'une pdf lisse
    x = np.linspace(min(result)*0.8, max(result)*1.2, 200)
    pdf = lognorm.pdf(x, s=sigma, scale=np.exp(mu))

    plt.figure()
    plt.hist(result, bins=10, density=True, alpha=0.5, label="Histogramme (Monte Carlo)")
    plt.plot(x, pdf, 'r-', label="PDF log-normale (fit)")

    plt.axvline(k_med, linestyle='--', label="Médiane")
    plt.axvline(k_min, linestyle=':', label="+34.1%")
    plt.axvline(k_max, linestyle=':', label="-34.1%")

    plt.xlabel("Perméabilité k (µm²)")
    plt.ylabel("Densité de probabilité")
    plt.title("PDF de la perméabilité")
    plt.legend()
    plt.grid()
    plt.savefig('devoir3uinput/pdf_k.png', dpi=300) # Sauvegarde du graphique
    plt.show()
    plt.close()

    # =========================
    # CDF discrète + théorique
    # =========================

    sorted_k = np.sort(result)
    cdf_emp = np.arange(1, len(sorted_k)+1) / len(sorted_k) # 0 - 1

    # points pour la CDF log-normale
    cdf_theo = lognorm.cdf(x, s=sigma, scale=np.exp(mu))

    plt.figure()
    plt.plot(sorted_k, cdf_emp, 'o', label="CDF discrète (Monte Carlo)")
    plt.plot(x, cdf_theo, '-', label="CDF log-normale (fit)")

    plt.xlabel("Perméabilité k (µm²)")
    plt.ylabel("Probabilité cumulée")
    plt.title("CDF de la perméabilité")
    plt.legend()
    plt.grid()
    plt.savefig('devoir3uinput/cdf_k.png', dpi=300) # Sauvegarde du graphique
    plt.show()
    plt.close()