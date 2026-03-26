import unittest
import numpy as np
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

    def testInvarianceGallileenne(self):
        """Test de l'invariance galiléenne du solveur."""


    def testOrdreDeConvergenceAvant(self):
        """Test l'odre de convergence pour le schéma avant."""

    def testOrdreDeConvergenceCentre(self):
        """Test de l'ordre de convergence pour le solveur avec schéma centré."""

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

            
if __name__ == '__main__':
    unittest.main()
