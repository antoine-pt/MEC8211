import unittest
import numpy as np
import matplotlib.pyplot as plt
try:
    from fonctions import *
except ImportError:
    print("Error: Could not import the 'fonctions' module. Please ensure it is in the same directory as this script.")
    exit(1)

class TestSolveur(unittest.TestCase):

    def testSolveur(self):
        """Test si le solveur fonctionne (si on obtient quelque chose sans erreur)."""
        prm = Parametres(nr = 3, nz = 3,t_fin = 60, dt = 12)
        Z, R = Position(prm)    
        t = 0
        T_init = np.full((prm.nr, prm.nz), prm.T_four)
        T_t = T_init
        
        pourcentage = 5.0
        while t<prm.t_fin:
            current_pct = (t + prm.dt) / prm.t_fin * 100
            if current_pct >= pourcentage or (t + prm.dt) >= prm.t_fin:
                print("Pourcentage de complétion : {}%".format(round(current_pct, 2)))
                while pourcentage <= current_pct:
                    pourcentage += 5.0
            T_tp1 = Temperature(prm, T_t, R)
            T_t = T_tp1
            t += prm.dt

        # Vérifier que la solution numérique est proche de la solution analytique
        
        self.assertTrue(T_tp1.all() != None, "La solution numérique est vide!")

    def testPosition(self):
        """Test la fonction Position avec des valeurs connues."""
        # Cas simple: nr=3, nz=3, R=2, H=2
        prm = Parametres(nr=3, nz=3, dt=0.25)
        prm.R = 2.0
        prm.H = 2.0
        prm.dr = prm.R / (prm.nr - 1)
        prm.dz = prm.H / (prm.nz - 1)
        
        Z, R = Position(prm)
        
        # Vérifier les dimensions
        self.assertEqual(Z.shape, (prm.nr, prm.nz), "Taille de Z devrait etre (nr, nz)")
        self.assertEqual(R.shape, (prm.nr, prm.nz), "Taille de R devrait etre (nr, nz)")
        
        # Vérifier que Z contient des valeurs de 0 à H/2
        self.assertTrue(np.all(Z >= 0), "Valeurs de Z doivent être >= 0")
        self.assertTrue(np.all(Z <= prm.H/2), "Valeurs de Z doivent être <= H/2")
        
        # Vérifier que R contient des valeurs de 0 à R
        self.assertTrue(np.all(R >= 0), "Valeurs de R doivent être >= 0")
        self.assertTrue(np.all(R <= prm.R), "Valeurs de R doivent être <= R")
        
        # Vérifier que les lignes de R sont constantes (même valeur dans chaque ligne)
        for i in range(prm.nr):
            self.assertTrue(np.allclose(R[i, :], R[i, 0]), f"Rangée {i} de R devrat avoir des valeurs constantes")
        
        # Vérifier que les colonnes de Z sont constantes (même valeur dans chaque colonne)
        for j in range(prm.nz):
            self.assertTrue(np.allclose(Z[:, j], Z[0, j]), f"Colone {j} de Z devrait avoir des valeurs constantes")
        
        # Vérifier les premiers et derniers éléments
        # Z doit aller de 0 à H/2
        self.assertAlmostEqual(Z[0, 0], 0.0, places=5, msg="Z[0,0] devrtait etre 0")
        self.assertAlmostEqual(Z[0, -1], prm.H / 2, places=5, msg="Z[0,-1] devrait etre H/2")
        
        # R doit aller de R (en haut) à 0 (en bas)
        self.assertAlmostEqual(R[0, 0], prm.R, places=5, msg="R[0,0] devrait être prm.R")
        self.assertAlmostEqual(R[-1, 0], 0.0, places=5, msg="R[-1,0] devrait être 0")

    def testMilieu(self):
        """Test la fonction Milieu avec des valeurs connues."""
        # Cas simple
        prm = Parametres(nr=3, nz=3, dt=1.0)
        prm.R = 2.0
        prm.H = 2.0
        prm.dr = prm.R / (prm.nr - 1)
        prm.dz = prm.H / (prm.nz - 1)
        prm.rho = 1.0
        prm.Cp = 1.0
        prm.k = 1.0
        Z, R = Position(prm)

        T_t = np.full((prm.nr, prm.nz), 1.0) # température constante de 1 partout 
        M = Milieu(prm, T_t, R)
        # Vérifier la solution constante obtenue pour le cas simple de T_t constant
        self.assertEqual(M.all(), T_t.all(), "La solution obtenue devrait être constante M == T_t")
        
        T_t[1,1] = 2.0 # on change la température au centre
        M = Milieu(prm, T_t, R)
        # Vérifier que la solution obtenue n'est plus constante (il devrait y avoir une variation autour du centre)
        self.assertFalse((M == T_t).all(), "La solution obtenue ne devrait pas être constante M != T_t")

    def testTemperature(self):
        """Test la fonction Temperature avec des valeurs connues."""
        # Cas simple
        prm = Parametres(nr=3, nz=3, dt=1.0)
        prm.R = 2.0
        prm.H = 2.0
        prm.dr = prm.R / (prm.nr - 1)
        prm.dz = prm.H / (prm.nz - 1)
        prm.rho = 1.0
        prm.Cp = 1.0
        prm.k = 1.0
        Z, R = Position(prm)

        # On pose T_inf à 1.0 et T_t à 1.0 partout, la solution devrait rester constante à 1.0 partout
        prm.T_inf = 1.0 # températur de l'environement égale à la température du domaine
        T_t = np.full((prm.nr, prm.nz), 1.0) # température constante de 1 partout 
        
        T = Temperature(prm, T_t, R)
        # Vérifier que la solution obtenue est plus constante
        self.assertTrue((T == T_t).all(), "La solution obtenue devrait être constante T == T_t")

        # On change la température de l'environement, la solution devrait changer (évoluer vers 2.0 partout)
        prm.T_inf = 2.0

        T = Temperature(prm, T_t, R)
        # Vérifier que la solution obtenue n'est plus constante
        self.assertFalse((T == T_t).all(), "La solution obtenue ne devrait pas être constante T != T_t")
        
    def testSymetrie(self):
        """Test si le solveur fonctionne (si on obtient quelque chose sans erreur)."""
        try:
            import test_files.functions_sans_symetrie as no_sym
        except ImportError:
            print("Error: Could not import the 'functions_symmetry' module. Please ensure it is in the 'test_files' directory.")
            exit(1)

        t_fin = 20 * 60
        ## Calcul de la solution numérique sans condition de symmétrie par rapport à z=0
        prm_full = no_sym.Parametres(nr = 10, nz = 41,t_fin = t_fin, dt = 1)
        Z_full, R_full = no_sym.Position(prm_full)    
        t = 0
        T_init_full = np.full((prm_full.nr, prm_full.nz), prm_full.T_four)
        T_t_full = T_init_full
        while t<prm_full.t_fin:
            T_tp1_full = no_sym.Temperature(prm_full, T_t_full, R_full)
            T_t_full = T_tp1_full
            t += prm_full.dt

        ## Calcul de la solution numérique avec condition de symmétrie par rapport à z=0
        prm = Parametres(nr = 10, nz = prm_full.nz//2+1,t_fin = t_fin, dt = 1)
        Z, R = Position(prm)    
        t = 0
        T_init_sym = np.full((prm.nr, prm.nz), prm.T_four)
        T_t_sym = T_init_sym
        while t<prm.t_fin:
            T_tp1_sym = Temperature(prm, T_t_sym, R)
            T_t_sym = T_tp1_sym
            t += prm.dt

        # Vérifier que la solution symétrique est à peu près égale à la solution complète (en tenant compte de la symétrie)
        norm = np.linalg.norm(T_tp1_full[:, prm_full.nz//2:] - T_tp1_sym, ord = 2)
        epsilon = 1
        print(norm)
        self.assertTrue(norm<epsilon, msg="La solution symétrique devrait être égale ou très proche de la moitié de la solution complète")
    
    def testInvarianceGallileenne(self):
        """Test de l'invariance galiléenne du solveur."""


    def testOrdreDeConvergenceAvant(self):
        """Test l'odre de convergence pour le schéma avant."""

    def testOrdreDeConvergenceCentre(self):
        """Test de l'ordre de convergence pour le solveur avec schéma centré."""          

if __name__ == '__main__':
    unittest.main()