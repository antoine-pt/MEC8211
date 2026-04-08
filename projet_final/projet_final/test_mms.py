import numpy as np
import matplotlib.pyplot as plt
import sympy as sp
try:
    from fonctions import *
except ImportError:
    print("Error: Could not import the 'fonctions' module. Please ensure it is in the same directory as this script.")
    exit(1)

if __name__ == "__main__":
    # Usage typique du solveur

    # --------- Initialisation de la solution en MMS ------------------
    t_fin = 60
    r,z,t = sp.symbols('r z t')
    T_ref = 800 + 273.15
    amplitude = 25.0
    solution_MMS_sympy  = T_ref + amplitude * sp.exp(-t/t_fin) * sp.cos(sp.pi*r/0.05) * sp.cos(sp.pi*z/0.05)

    # ---------- Initialisation des parametres --------------------------
    prm = Parametres(nr = 10, nz = 10,t_fin = t_fin, dt = 1, solution_MMS_sympy = solution_MMS_sympy)
    T_init = np.asarray(prm.solution_MMS(prm.R, prm.Z, 0.0), dtype=float)
    T_t = T_init.copy()
    
    # ---------- Run de la simulation jusqu'au temps final --------------
    pourcentage = 5.0
    while prm.time<prm.t_fin:
        current_pct = (prm.time + prm.dt) / prm.t_fin * 100
        if current_pct >= pourcentage or (prm.time) >= prm.t_fin:
            print("Pourcentage de complétion : {}%".format(round(current_pct, 2)))
            while pourcentage <= current_pct:
                pourcentage += 5.0
        T_tp1 = Temperature(prm, T_t)
        T_t = T_tp1
        prm.Time(prm.dt)

    # ----- Comparaison à la solution exacte de MMS au dernier pas de temps -----
    T_exact_final = np.asarray(prm.solution_MMS(prm.R, prm.Z, prm.time), dtype=float)
    error_final = T_t - T_exact_final

    l1_error = np.mean(np.abs(error_final))
    l2_error = np.sqrt(np.mean(error_final ** 2))

    print("Erreur L1 au dernier pas de temps : {:.6e} K".format(l1_error))
    print("Erreur L2 au dernier pas de temps : {:.6e} K".format(l2_error))




    # -------- Plot des graphiques --------------------------------------------

    # Affichage de la solution MMS exacte finale
    plt.figure(figsize=(8, 6))
    plt.contourf(prm.Z, prm.R, T_exact_final, levels=50, cmap='inferno')
    plt.colorbar(label='Température (K)')
    plt.title('Solution MMS exacte à t = {} s'.format(prm.time))
    plt.xlabel('Position z (m)')
    plt.ylabel('Position r (m)')
    plt.tight_layout()


    # Affichage de l'erreur finale entre la solution simulée et la MMS
    plt.figure(figsize=(8, 6))
    error_abs_max = max(np.max(np.abs(error_final)), 1e-16)
    plt.contourf(
        prm.Z,
        prm.R,
        error_final,
        levels=np.linspace(-error_abs_max, error_abs_max, 51),
        cmap='coolwarm'
    )
    plt.colorbar(label='Erreur de température (K)')
    plt.title('Erreur finale : solution simulée - solution MMS')
    plt.xlabel('Position z (m)')
    plt.ylabel('Position r (m)')
    plt.tight_layout()


    # Affichage de la température finale simulée
    plt.figure(figsize=(8, 6))
    plt.contourf(prm.Z, prm.R, T_t, levels=50, cmap='inferno')
    plt.colorbar(label='Température (K)')
    plt.title('Distribution de la température simulée dans le cylindre à t = {} s'.format(prm.time))
    plt.xlabel('Position z (m)')
    plt.ylabel('Position r (m)')
    plt.tight_layout()
    plt.show()



