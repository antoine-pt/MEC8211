import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
import os
try:
    from fonctions import *
except ImportError:
    print("Error: Could not import the 'fonctions' module. Please ensure it is in the same directory as this script.")
    exit(1)


if __name__ == "__main__":

    study_type = "t"  # ou "t" selon le type d'étude

    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "resultats")
    files_npz = sorted([f for f in os.listdir(results_dir) if f.endswith(".npz")])

    cell_size_list = []
    L1_st_list, L1_f_list = [], []
    L2_st_list, L2_f_list = [], []
    Li_st_list, Li_f_list = [], []

    for file in files_npz:
        data = np.load(os.path.join(results_dir, file))

        if str(data["study_type"]) != study_type:
            continue

        nr = int(data["nr"])
        nz = int(data["nz"])
        dt = float(data["dt"])
        T_simu = data["T_simu"]
        T_ana  = data["T_ana"]

        prm = Parametres(nr=nr, nz=nz, t_fin=60, dt=dt)

        L1_st,  L1_f  = normeL1(T_ana,  T_simu, prm)
        L2_st,  L2_f  = normeL2(T_ana,  T_simu, prm)
        Li_st,  Li_f  = normeLinf(T_ana, T_simu, prm)

        L1_st_list.append(L1_st);  L1_f_list.append(L1_f)
        L2_st_list.append(L2_st);  L2_f_list.append(L2_f)
        Li_st_list.append(Li_st);  Li_f_list.append(Li_f)
        cell_size_list.append(np.sqrt(prm.dr**2 + prm.dz**2) if study_type == "rz" else prm.dt)

    # Trier par paramètre croissant
    idx        = np.argsort(cell_size_list)
    params     = np.array(cell_size_list)[idx]
    L1_st_arr  = np.array(L1_st_list)[idx];  L1_f_arr  = np.array(L1_f_list)[idx]
    L2_st_arr  = np.array(L2_st_list)[idx];  L2_f_arr  = np.array(L2_f_list)[idx]
    Li_st_arr  = np.array(Li_st_list)[idx];  Li_f_arr  = np.array(Li_f_list)[idx]

    xlabel = r"Évolution de $n_r = n_z, \Delta t = 0.02$" if study_type == "rz" else r"Évolution de $\Delta t, n_r = n_z $"

    def convergence_order(params, errors, n_fit=3):
        """Calcule l'ordre de convergence sur les n_fit plus grandes discrétisations."""
        p = params[-n_fit:]
        e = errors[-n_fit:]
        coeffs = np.polyfit(np.log(p), np.log(e), 1)
        fit_full = np.exp(np.polyval(coeffs, np.log(params)))  # fit sur tous les points pour affichage
        return coeffs[0], fit_full

    # ---------- Graphiques -------------------------------------------
    norms     = ["L1",       "L2",       "L∞"      ]
    st_arrays = [L1_st_arr,  L2_st_arr,  Li_st_arr ]
    f_arrays  = [L1_f_arr,   L2_f_arr,   Li_f_arr  ]
    colors    = ["tab:blue", "tab:orange","tab:green"]

    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    fig.suptitle(f"Convergence — étude : {xlabel}", fontsize=13)

    for col, (norm, st, f, color) in enumerate(zip(norms, st_arrays, f_arrays, colors)):
        for row, (errors, title) in enumerate([(st, "Spatio-temporel"), (f, "Final")]):
            ax = axes[row, col]
            order, fit = convergence_order(params, errors)
            ax.loglog(params, errors, "o-",  color=color, label=norm)
            ax.loglog(params, fit,    "--",  color=color, label=f"Ordre ≈ {order:.2f}")
            ax.set_xlabel(xlabel)
            ax.set_ylabel("Erreur")
            ax.set_title(f"{norm} — {title}")
            ax.legend()
            ax.grid(True, which="both", linestyle=":", alpha=0.6)

    fig.tight_layout()
    plt.show()