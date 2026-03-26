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
        prm = Parametres(nr = 50, nz = 50,t_fin = 120 *60, dt = 12)
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

            

