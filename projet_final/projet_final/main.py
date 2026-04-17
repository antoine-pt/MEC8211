import numpy as np
import matplotlib.pyplot as plt
try:
    from fonctions import *
except ImportError:
    print("Error: Could not import the 'fonctions' module. Please ensure it is in the same directory as this script.")
    exit(1)

if __name__ == "__main__":

    ## ============ Usage typique du solveur ========================

    # Initialisation des parametres
    prm = Parametres(nr = 15, nz = 15,t_fin = 30 *60, dt = 1)
    T_init = np.full((prm.nr, prm.nz), prm.T_four)
    T_t = T_init
    
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

    # Affichage de la température finale
    plt.figure(figsize=(8, 6))
    plt.contourf(prm.Z, prm.R, T_t, levels=50, cmap='inferno')
    plt.colorbar(label='Température (°C)')
    plt.title('Distribution de la température dans le cylindre à t = {} s'.format(prm.t_fin))
    plt.xlabel('Position z (m)')
    plt.ylabel('Position r (m)')
    plt.show()

    ## ============= Fin Usage Typique du Solveur ======================

    ## ================== MMS Solution =================================

    # --------- Initialisation de la solution en MMS ------------------
    t_fin = 60
    r, z, t = sp.symbols('r z t')
    T_ref = 800 + 273.15
    amplitude = 25.0
    solution_MMS_sympy = T_ref + amplitude * sp.exp(-t/t_fin) * sp.cos(sp.pi*r/0.05) * sp.cos(sp.pi*z/0.05)

    # Initialisation des parametres
    nr = 20
    nz = 20
    dt = 0.01
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
        prm.Time(prm.dt)
        T_t = T_tp1
        

        T_simu[timestep, :, :] = T_tp1
        T_ana[timestep, :, :]  = np.asarray(
            prm.solution_MMS(prm.R, prm.Z, prm.time), dtype=float
        )
        timestep += 1

    # ---------- Calcul des normes d'erreur -------------------------

    L1, L1_final = normeL1(T_ana, T_simu, prm)
    L2, L2_final = normeL2(T_ana, T_simu, prm)
    Linf, Linf_final = normeLinf(T_ana, T_simu, prm)

    print(f"Norme L1 (spatio-temporel) : {L1:.4e}, (final) : {L1_final:.4e}")
    print(f"Norme L2 (spatio-temporel) : {L2:.4e}, (final) : {L2_final:.4e}")
    print(f"Norme L∞ (spatio-temporel) : {Linf:.4e}, (final) : {Linf_final:.4e}")

    # ---------- Plots ---------------------------------------------

    # in case timestep < n_steps
    T_ana_plot = T_ana[:timestep]
    T_simu_plot = T_simu[:timestep]
    T_diff_plot = T_simu_plot - T_ana_plot

    # displays 10 snapshots evenly spaced in time
    for i in range(1,100,10):
        it = int(i/100 * timestep)
        fig, axes = plt.subplots(1, 3, figsize=(18, 5))

        vmin = min(T_ana_plot[it].min(), T_simu_plot[it].min())
        vmax = max(T_ana_plot[it].max(), T_simu_plot[it].max())
        diff_abs = np.max(np.abs(T_diff_plot[it]))


        cf1 = axes[0].pcolormesh(prm.Z, prm.R, T_ana_plot[it], cmap='inferno', vmin=vmin, vmax=vmax, shading='auto')
        cf2 = axes[1].pcolormesh(prm.Z, prm.R, T_simu_plot[it], cmap='inferno', vmin=vmin, vmax=vmax, shading='auto')
        cf3 = axes[2].pcolormesh(prm.Z, prm.R, T_diff_plot[it], cmap='RdBu_r', vmin=-diff_abs, vmax=diff_abs, shading='auto')
        
        fig.colorbar(cf1, ax=axes[0], label='Température (K)')
        fig.colorbar(cf2, ax=axes[1], label='Température (K)')
        fig.colorbar(cf3, ax=axes[2], label='Écart (K)')

        axes[0].set_title(f'Solution analytique, t = {prm.dt*(it+1):.2f} s')
        axes[1].set_title(f'Solution numérique, t = {prm.dt*(it+1):.2f} s')
        axes[2].set_title(f'Différence (num - ana), t = {prm.dt*(it+1):.2f} s')

        for ax in axes:
            ax.set_xlabel('Position z (m)')
            ax.set_ylabel('Position r (m)')

        plt.tight_layout()
        plt.show()

    ## ================== Fin MMS Solution =============================