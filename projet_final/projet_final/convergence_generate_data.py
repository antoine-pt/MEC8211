## Assisté de Claude
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

    # --------- Initialisation de la solution en MMS ------------------
    t_fin = 60
    r, z, t = sp.symbols('r z t')
    T_ref = 800 + 273.15
    amplitude = 25.0
    solution_MMS_sympy = T_ref + amplitude * sp.exp(-t/t_fin) * sp.cos(sp.pi*r/0.05) * sp.cos(sp.pi*z/0.05)

    # ---------- Configurations à tester (étude de convergence) -------
    configurations = [
        {"study_type": "rz", "nr": 5, "nz": 5, "dt": 0.01},
        {"study_type": "rz", "nr": 10, "nz": 10, "dt": 0.01},
        {"study_type": "rz", "nr": 15, "nz": 15, "dt": 0.01},
        {"study_type": "rz", "nr": 20, "nz": 20, "dt": 0.01},
        {"study_type": "rz", "nr": 25, "nz": 25, "dt": 0.01},
        {"study_type": "rz", "nr": 30, "nz": 30, "dt": 0.01},
        {"study_type": "rz", "nr": 50, "nz": 50, "dt": 0.01},
        {"study_type": "t", "nr": 25, "nz": 25, "dt": 0.25},
        {"study_type": "t", "nr": 25, "nz": 25, "dt": 0.30},
        {"study_type": "t", "nr": 25, "nz": 25, "dt": 0.35},
        {"study_type": "t", "nr": 25, "nz": 25, "dt": 0.40},
        {"study_type": "t", "nr": 25, "nz": 25, "dt": 0.41},
        {"study_type": "t", "nr": 25, "nz": 25, "dt": 0.42},
        {"study_type": "t", "nr": 25, "nz": 25, "dt": 0.43},

    ]

    # ---------- Dossier de résultats ----------------------------------
    # Dossier du script courant
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Dossier "resultats" au même niveau que le script
    results_dir = os.path.join(script_dir, "resultats")
    os.makedirs(results_dir, exist_ok=True)

    # ---------- Boucle sur les configurations -------------------------
    for config in configurations:
        nr   = config["nr"]
        nz   = config["nz"]
        dt   = config["dt"]

        print(f"\nSimulation : nr={nr}, nz={nz}, dt={dt}")

        # ---------- Initialisation ------------------------------------
        prm = Parametres(nr=nr, nz=nz, t_fin=t_fin, dt=dt, epsilon = 0.0, solution_MMS_sympy=solution_MMS_sympy)
        T_init = np.asarray(prm.solution_MMS(prm.R, prm.Z, 0.0), dtype=float)
        T_t = T_init.copy()

        n_steps = int(round(prm.t_fin / prm.dt))
        T_simu = np.zeros((n_steps, nr, nz))
        T_ana  = np.zeros_like(T_simu)

        # ---------- Simulation ----------------------------------------
        timestep = 0
        pourcentage = 5.0
        while prm.time < prm.t_fin - prm.dt * 0.5: # Pour éviter les problèmes de précision flottante
            current_pct = (prm.time + prm.dt) / prm.t_fin * 100
            if current_pct >= pourcentage or (prm.time) >= prm.t_fin:
                print("Pourcentage de complétion : {}%".format(round(current_pct, 2)))
                while pourcentage <= current_pct:
                    pourcentage += 5.0
            T_tp1 = Temperature(prm, T_t)
            T_t = T_tp1
            prm.Time(prm.dt)

            T_simu[timestep, :, :] = T_tp1
            T_ana[timestep, :, :]  = np.asarray(
                prm.solution_MMS(prm.R, prm.Z, prm.time), dtype=float
            )
            timestep += 1

        # ---------- Sauvegarde des résultats --------------------------
        # Vecteurs r et z (axes spatiaux)
        r_vec = prm.R[:, 0]   # shape (nr,)  — 1ère colonne de la grille R
        z_vec = prm.Z[0, :]   # shape (nz,)  — 1ère ligne   de la grille Z
        t_vec = np.arange(1, n_steps + 1) * dt   # shape (n_steps,)

        if config["study_type"] == "rz":
            filename = f"rz_nr{nr}_nz{nz}_dt{str(dt).replace('.', 'p')}.npz"
        else:
            filename = f"t_nr{nr}_nz{nz}_dt{str(dt).replace('.', 'p')}.npz"

        filepath = os.path.join(results_dir, filename)

        np.savez(
            filepath,
            T_simu = T_simu,   # (n_steps, nr, nz)
            T_ana  = T_ana,    # (n_steps, nr, nz)
            t      = t_vec,    # (n_steps,)
            r      = r_vec,    # (nr,)
            z      = z_vec,    # (nz,)
            nr     = nr,
            nz     = nz,
            dt     = dt,
            study_type = config["study_type"]
        )
        print(f"  → Résultats sauvegardés : {filepath}")

    print("\nToutes les simulations sont terminées.")





