import numpy as np
import matplotlib.pyplot as plt
import sys

try:
    from fonctions_monte_carlo import *
except ImportError:
    try:
        from fonctions import *
    except ImportError:
        print("Error: Could not import the fonctions module.")
        exit(1)

# ==============================================================================
# PARAMÈTRES DE LA SIMULATION
# ==============================================================================

NR, NZ   = 15, 15
DT       = 0.6
T_FIN    = 600

# Valeurs nominales
EPSILON_NOM = 0.525
K_NOM       = 47.5

# Incertitudes standards (1σ) — à ajuster selon les fiches techniques / calibration
# ASME V&V 20 : u_i est l'incertitude standard (k=1) associée à chaque entrée
EPSILON_STD = 0.19     
K_STD       = 3.75     

# Nombre de tirages Monte Carlo
N_MC = 200    # Augmenter (ex: 1000) pour plus de précision


# ==============================================================================
# 1.  SIMULATION NOMINALE + AFFICHAGE
# ==============================================================================

if __name__ == "__main__":

    print("=" * 60)
    print("  SIMULATION NOMINALE")
    print("=" * 60)

    prm = Parametres(nr=NR, nz=NZ, t_fin=T_FIN, dt=DT,
                     epsilon=EPSILON_NOM, k=K_NOM)
    T_t = np.full((prm.nr, prm.nz), prm.T_four)

    pourcentage = 5.0
    while prm.time < prm.t_fin:
        current_pct = (prm.time + prm.dt) / prm.t_fin * 100
        if current_pct >= pourcentage:
            print(f"  Avancement : {round(current_pct, 1)} %")
            while pourcentage <= current_pct:
                pourcentage += 5.0
        T_t = Temperature(prm, T_t)
        prm.Time(prm.dt)

    QOI_nom = QOI(T_t, prm)
    print(f"\n  QOI nominale (T_moy surface) = {QOI_nom:.4f} K\n")

    # Tracé du champ nominal
    fig, ax = plt.subplots(figsize=(8, 6))
    cf = ax.contourf(prm.Z, prm.R, T_t, levels=50, cmap='inferno')
    plt.colorbar(cf, ax=ax, label='Température (K)')
    ax.set_title(f'Distribution de température — t = {T_FIN} s (nominal)')
    ax.set_xlabel('Position z (m)')
    ax.set_ylabel('Position r (m)')
    plt.tight_layout()
    plt.savefig('temperature_nominale.png', dpi=150)
    plt.show()

    # ==============================================================================
    # 2.  ANALYSE D'INCERTITUDE — Monte Carlo (ASME V&V 20)
    # ==============================================================================

    resultats = monte_carlo_uinput(
        nr=NR, nz=NZ, dt=DT, t_fin=T_FIN,
        epsilon_nom=EPSILON_NOM, epsilon_std=EPSILON_STD,
        k_nom=K_NOM,             k_std=K_STD,
        N=N_MC,
        seed=42,
        verbose_mc=True
    )

    u_input      = resultats['u_input']
    QOI_samples  = resultats['QOI_samples']
    mean_QOI     = resultats['mean_QOI']

    # ==============================================================================
    # 3.  VISUALISATIONS MONTE CARLO
    # ==============================================================================

    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    fig.suptitle("Analyse d'incertitude — Monte Carlo (ASME V&V 20)", fontsize=13, fontweight='bold')

    # --- Histogramme de la QOI ---
    ax = axes[0]
    ax.hist(QOI_samples, bins=30, color='steelblue', edgecolor='white', alpha=0.85)
    ax.axvline(mean_QOI,              color='orange', linewidth=2, label=f'Moyenne = {mean_QOI:.3f} K')
    ax.axvline(mean_QOI - u_input,    color='red',    linewidth=1.5, linestyle='--', label=f'±u_input = {u_input:.4f} K')
    ax.axvline(mean_QOI + u_input,    color='red',    linewidth=1.5, linestyle='--')
    ax.axvline(mean_QOI - 2*u_input,  color='crimson',linewidth=1.2, linestyle=':',  label='±2·u_input (95 %)')
    ax.axvline(mean_QOI + 2*u_input,  color='crimson',linewidth=1.2, linestyle=':')
    ax.set_xlabel('QOI — T_moy surface (K)')
    ax.set_ylabel('Fréquence')
    ax.set_title('Distribution de la QOI')
    ax.legend(fontsize=8)

    # --- Scatter epsilon vs QOI ---
    ax = axes[1]
    sc = ax.scatter(resultats['epsilon_samples'], QOI_samples,
                    c=resultats['k_samples'], cmap='coolwarm', s=20, alpha=0.7)
    plt.colorbar(sc, ax=ax, label='k [W/m·K]')
    ax.set_xlabel('Émissivité ε [-]')
    ax.set_ylabel('QOI (K)')
    ax.set_title('Sensibilité à ε  (couleur = k)')

    # --- Scatter k vs QOI ---
    ax = axes[2]
    sc2 = ax.scatter(resultats['k_samples'], QOI_samples,
                     c=resultats['epsilon_samples'], cmap='RdYlGn', s=20, alpha=0.7)
    plt.colorbar(sc2, ax=ax, label='ε [-]')
    ax.set_xlabel('Conductivité k [W/m·K]')
    ax.set_ylabel('QOI (K)')
    ax.set_title('Sensibilité à k  (couleur = ε)')

    plt.tight_layout()
    plt.savefig('monte_carlo_uinput.png', dpi=150)
    plt.show()

    # ==============================================================================
    # 4.  RÉSUMÉ FINAL
    # ==============================================================================

    print("\n" + "=" * 60)
    print("  RÉSUMÉ — u_input (ASME V&V 20)")
    print("=" * 60)
    print(f"  Source d'incertitude    Valeur nominale   u_i (1σ)")
    print(f"  ─────────────────────   ───────────────   ──────────────")
    print(f"  Émissivité  ε           {EPSILON_NOM:.4f}           {EPSILON_STD:.6f}")
    print(f"  Conductivité k          {K_NOM:.2f} W/m·K     {K_STD:.4f} W/m·K")
    print(f"  ─────────────────────   ───────────────   ──────────────")
    print(f"  u_input combiné (QOI)                     {u_input:.6f} K")
    print(f"  Intervalle 95 %  (±2σ)  [{mean_QOI - 2*u_input:.4f}, {mean_QOI + 2*u_input:.4f}] K")
    print("=" * 60)
