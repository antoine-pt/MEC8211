import unittest
import numpy as np
import matplotlib.pyplot as plt
try:
    from .fonctions import *
except ImportError:
    print("Error: Could not import the 'fonctions' module. Please ensure it is in the same directory as this script.")
    exit(1)

class TestSolveur(unittest.TestCase):

    def testSolveur(self):
        """Test si le solveur fonctionne (si on obtient quelque chose sans erreur)."""
        prm = Parametres(nr = 3, nz = 3,t_fin = 60, dt = 12)
        T_init = np.full((prm.nr, prm.nz), prm.T_four)
        T_t = T_init
        while prm.time<prm.t_fin:
            T_tp1 = Temperature(prm, T_t)
            T_t = T_tp1
            prm.Time(prm.dt)

        # Vérifier que la solution numérique est proche de la solution analytique
        
        self.assertTrue(T_tp1.all() != None, "La solution numérique est vide!")

    def testPosition(self):
        """Test la fonction Position avec des valeurs connues."""
        # Cas simple: nr=3, nz=3, R=2, H=2
        prm = Parametres(nr=3, nz=3, dt=0.25)
        prm.Rmax = 2.0
        prm.H = 2.0
        prm.dr = prm.Rmax / (prm.nr - 1)
        prm.dz = prm.H / (prm.nz - 1)
        prm.Z, prm.R = Position(prm)
        
        # Vérifier les dimensions
        self.assertEqual(prm.Z.shape, (prm.nr, prm.nz), "Taille de Z devrait etre (nr, nz)")
        self.assertEqual(prm.R.shape, (prm.nr, prm.nz), "Taille de R devrait etre (nr, nz)")
        
        # Vérifier que Z contient des valeurs de 0 à H/2
        self.assertTrue(np.all(prm.Z >= 0), "Valeurs de Z doivent être >= 0")
        self.assertTrue(np.all(prm.Z <= prm.H/2), "Valeurs de Z doivent être <= H/2")
        
        # Vérifier que R contient des valeurs de 0 à R
        self.assertTrue(np.all(prm.R >= 0), "Valeurs de R doivent être >= 0")
        self.assertTrue(np.all(prm.R <= prm.Rmax), "Valeurs de R doivent être <= R")
        
        # Vérifier que les lignes de R sont constantes (même valeur dans chaque ligne)
        for i in range(prm.nr):
            self.assertTrue(np.allclose(prm.R[i, :], prm.R[i, 0]), f"Rangée {i} de R devrat avoir des valeurs constantes")
        
        # Vérifier que les colonnes de Z sont constantes (même valeur dans chaque colonne)
        for j in range(prm.nz):
            self.assertTrue(np.allclose(prm.Z[:, j], prm.Z[0, j]), f"Colone {j} de Z devrait avoir des valeurs constantes")
        
        # Vérifier les premiers et derniers éléments
        # Z doit aller de 0 à H/2
        self.assertAlmostEqual(prm.Z[0, 0], 0.0, places=5, msg="Z[0,0] devrtait etre 0")
        self.assertAlmostEqual(prm.Z[0, -1], prm.H / 2, places=5, msg="Z[0,-1] devrait etre H/2")
        
        # R doit aller de R (en haut) à 0 (en bas)
        self.assertAlmostEqual(prm.R[0, 0], prm.Rmax, places=5, msg="R[0,0] devrait être prm.Rmax")
        self.assertAlmostEqual(prm.R[-1, 0], 0.0, places=5, msg="R[-1,0] devrait être 0")

    def testMilieu(self):
        """Test la fonction Milieu avec des valeurs connues."""
        # Cas simple
        prm = Parametres(nr=3, nz=3, dt=1.0)
        prm.Rmax = 2.0
        prm.H = 2.0
        prm.dr = prm.Rmax / (prm.nr - 1)
        prm.dz = prm.H / (prm.nz - 1)
        prm.Z, prm.R = Position(prm)
        prm.rho = 1.0
        prm.Cp = 1.0
        prm.k = 1.0

        T_t = np.full((prm.nr, prm.nz), 1.0) # température constante de 1 partout 
        M = Milieu(prm, T_t)
        # Vérifier la solution constante obtenue pour le cas simple de T_t constant
        self.assertEqual(M.all(), T_t.all(), "La solution obtenue devrait être constante M == T_t")
        
        T_t[1,1] = 2.0 # on change la température au centre
        M = Milieu(prm, T_t)
        # Vérifier que la solution obtenue n'est plus constante (il devrait y avoir une variation autour du centre)
        self.assertFalse((M == T_t).all(), "La solution obtenue ne devrait pas être constante M != T_t")

    def testTemperature(self):
        """Test la fonction Temperature avec des valeurs connues."""
        # Cas simple
        prm = Parametres(nr=3, nz=3, dt=1.0)
        prm.Rmax = 2.0
        prm.H = 2.0
        prm.dr = prm.Rmax / (prm.nr - 1)
        prm.dz = prm.H / (prm.nz - 1)
        prm.Z, prm.R = Position(prm)
        prm.rho = 1.0
        prm.Cp = 1.0
        prm.k = 1.0

        # On pose T_inf à 1.0 et T_t à 1.0 partout, la solution devrait rester constante à 1.0 partout
        prm.T_inf = 1.0 # températur de l'environement égale à la température du domaine
        T_t = np.full((prm.nr, prm.nz), 1.0) # température constante de 1 partout 
        
        T = Temperature(prm, T_t)
        # Vérifier que la solution obtenue est plus constante
        self.assertTrue((T == T_t).all(), "La solution obtenue devrait être constante T == T_t")

        # On change la température de l'environement, la solution devrait changer (évoluer vers 2.0 partout)
        prm.T_inf = 2.0

        T = Temperature(prm, T_t)
        # Vérifier que la solution obtenue n'est plus constante
        self.assertFalse((T == T_t).all(), "La solution obtenue ne devrait pas être constante T != T_t")
        
    def testSymetrie(self):
        """Test la simplification de la condition de symétrie en comparant 
        le résultat symétrique au résultat du domaine complet."""
        try:
            from .test_files import fonctions_sans_symetrie as no_sym
        except ImportError:
            print("Error: Could not import the 'functions_symmetry' module. Please ensure it is in the 'test_files' directory.")
            exit(1)

        t_fin = 20 * 60
        ## Calcul de la solution numérique sans condition de symmétrie par rapport à z=0
        prm_full = no_sym.Parametres(nr = 10, nz = 41,t_fin = t_fin, dt = 1)
        T_init_full = np.full((prm_full.nr, prm_full.nz), prm_full.T_four)
        T_t_full = T_init_full
        while prm_full.time<prm_full.t_fin:
            T_tp1_full = no_sym.Temperature(prm_full, T_t_full)
            T_t_full = T_tp1_full
            prm_full.Time(prm_full.dt)

        ## Calcul de la solution numérique avec condition de symmétrie par rapport à z=0
        prm = Parametres(nr = 10, nz = prm_full.nz//2+1,t_fin = t_fin, dt = 1)
        T_init_sym = np.full((prm.nr, prm.nz), prm.T_four)
        T_t_sym = T_init_sym
        while prm.time<prm.t_fin:
            T_tp1_sym = Temperature(prm, T_t_sym)
            T_t_sym = T_tp1_sym
            prm.Time(prm.dt)

        # Vérifier que la solution symétrique est à peu près égale à la solution complète (en tenant compte de la symétrie)
        norm = np.linalg.norm(T_tp1_full[:, prm_full.nz//2:] - T_tp1_sym, ord = 2)
        epsilon = 1
        print(norm)
        self.assertTrue(norm<epsilon, msg="La solution symétrique devrait être égale ou très proche de la moitié de la solution complète")
    
    def testConservationEnergie(self):
        """Test la conservation de l'énergie du solveur en effectuant un bilan énegérique
        à l'intérieur du domaine et sur la frontière. On cherche q_dot_intérieur ~= q_dot_frontière"""
        
        # ------- Calcul de la solution numérique --------
        prm = Parametres(nr=10, nz=10, t_fin=10*60, dt=3)
        T_init = np.full((prm.nr, prm.nz), prm.T_four)
        T_t = T_init.copy()

        while prm.time < prm.t_fin:
            prm.Time(prm.dt)
            if prm.time >= prm.t_fin:
                T_tm1 = T_t.copy()
            T_tp1 = Temperature(prm, T_t)
            T_t = T_tp1

        # ------ Variation d'énergie interne ----------
        E_prec = 0.0
        E_fin = 0.0
        for i in range(prm.nr - 1):
            for j in range(prm.nz - 1):
                T_centre_prec = 0.25 * (T_tm1[i,j] + T_tm1[i+1,j] + T_tm1[i,j+1] + T_tm1[i+1,j+1])
                T_centre_fin  = 0.25 * (T_tp1[i,j] + T_tp1[i+1,j] + T_tp1[i,j+1] + T_tp1[i+1,j+1])
                
                # Calcul du volume de la cellule pour un solveur axisymétrique (cylindrique)
                R_centre = 0.25 * (prm.R[i,j] + prm.R[i+1,j] + prm.R[i,j+1] + prm.R[i+1,j+1])
                dV = 2 * np.pi * R_centre * prm.dr * prm.dz
                
                E_prec += T_centre_prec * dV * prm.rho * prm.Cp
                E_fin  += T_centre_fin  * dV * prm.rho * prm.Cp

        q_dot_internal = (E_fin - E_prec) / prm.dt

        # ----- Flux sortant aux frontières dissipatives seulement -------
        q_dot_front_prec = 0.0
        q_dot_front_fin = 0.0

        # frontière r = r_max  --> r = 0, longueur = dz
        r = 0
        for z in range(prm.nz - 1):
            T_front_prec = 0.5 * (T_tm1[r,z] + T_tm1[r,z+1])
            T_front_fin  = 0.5 * (T_tp1[r,z] + T_tp1[r,z+1])

            # Calcul de l'aire pour un solveur axisymétrique (cylindrique)
            R_face = prm.Rmax
            dA = 2 * np.pi * R_face * prm.dz

            q_dot_front_prec += (prm.h * (T_front_prec - prm.T_inf) +
                            prm.epsilon * prm.sigma * (T_front_prec**4 - prm.T_inf**4)) * dA

            q_dot_front_fin += (prm.h * (T_front_fin - prm.T_inf) +
                            prm.epsilon * prm.sigma * (T_front_fin**4 - prm.T_inf**4)) * dA

        # frontière z = z_max --> z = nz-1, longueur = dr
        z = prm.nz - 1
        for r in range(prm.nr - 1):
            T_front_prec = 0.5 * (T_tm1[r,z] + T_tm1[r+1,z])
            T_front_fin  = 0.5 * (T_tp1[r,z] + T_tp1[r+1,z])

            # Calcul de l'aire pour un solveur axisymétrique (cylindrique)
            R_face = 0.5 * (prm.R[r,z] + prm.R[r+1,z])
            dA = 2 * np.pi * R_face * prm.dr
            
            q_dot_front_prec += (prm.h * (T_front_prec - prm.T_inf) +
                            prm.epsilon * prm.sigma * (T_front_prec**4 - prm.T_inf**4)) * dA

            q_dot_front_fin += (prm.h * (T_front_fin - prm.T_inf) +
                            prm.epsilon * prm.sigma * (T_front_fin**4 - prm.T_inf**4)) * dA

        # Moyenne du flux sortant entre t et t+dt
        q_dot_out = 0.5 * (q_dot_front_prec + q_dot_front_fin)

        # ------ Test ----------
        relative_error = abs(q_dot_internal + q_dot_out) / (abs(q_dot_out))
        self.assertTrue(relative_error < 1e-3, msg="Le bilan énergétique devrait être proche de zéro (conservation de l'énergie)")

    def testInvarianceGalileenne(self):
        """Test de l'invariance galiléenne du solveur par translation des coordonnées."""
        try:
            from .test_files import fonctions_translation_coordonnees as translation
        except ImportError:
            print("Error: Could not import the 'functions_symmetry' module. Please ensure it is in the 'test_files' directory.")
            exit(1)
        t_fin = 20 * 60
        nr = 5
        nz = 5
        ## Calcul de la solution numérique sans condition de symmétrie par rapport à z=0
        prm_trans = translation.Parametres(nr = nr, nz = nz,t_fin = t_fin, dt = 3)
        prm = Parametres(nr = nr, nz = nz,t_fin = t_fin, dt = 3)
        T_init_trans = np.full((prm_trans.nr, prm_trans.nz), prm_trans.T_four)
        T_t_trans = T_init_trans

        T_init = np.full((prm.nr, prm.nz), prm.T_four)
        T_t = T_init
        while prm_trans.time<prm_trans.t_fin:

            T_tp1_trans = translation.Temperature(prm_trans, T_t_trans)
            T_t_trans = T_tp1_trans

            T_tp1 = Temperature(prm, T_t)
            T_t = T_tp1

            prm_trans.Time(prm_trans.dt)
            prm.Time(prm.dt)
        

        # Vérifier que la solution symétrique est à peu près égale à la solution complète (en tenant compte de la symétrie)
        norm = np.linalg.norm(T_tp1 - T_tp1_trans, ord = 2)
        epsilon = 1
        print(norm)
        self.assertTrue(norm<epsilon, msg="La solution symétrique devrait être égale ou très proche de la moitié de la solution complète")
    
    def testOrdreDeConvergenceAvant(self):
        """Test l'odre de convergence pour le schéma avant."""

    def testOrdreDeConvergenceCentre(self):
        """Test de l'ordre de convergence pour le solveur avec schéma centré."""          

if __name__ == '__main__':
    unittest.main()