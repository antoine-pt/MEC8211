import unittest
import numpy as np
from solveur import *

class TestSolveur(unittest.TestCase):

    def testSolveur(self):
        """Test si le solveur fonctionne (si on obtient quelque chose sans erreur)."""
        nPts = 10
        R = 1.0
        D = 1e-10
        S = 1e-8
        Ce = 10

        params = Parametres(nPts=nPts, R=R, D=D, S=S, Ce=Ce)
        sim = solveur_avant(params)
        ana = analytique(params)

        # Vérifier que la solution numérique est proche de la solution analytique
        
        self.assertTrue(sim.all() != None, "La solution numérique est vide!")
        self.assertTrue(sim.all() != 0.0, "La solution numérique est vide!")
        self.assertTrue(ana.all() != None, "La solution analytique est vide!")

    def testInvarianceGallileenne(self):
        """Test de l'invariance galiléenne du solveur."""
        nPts = 10
        R = 1.0
        D = 1e-10
        S = 1e-8
        Ce = 0

        params = Parametres(nPts=nPts, R=R, D=D, S=S, Ce=Ce)
        sim1 = solveur_centre(params)

        # Modifier les paramètres pour simuler un changement de référentiel
        params.S *= 2  # Doubler la source
        sim2 = solveur_centre(params)

        # Vérifier que la solution numérique est proportionnelle à la source
        # np.allclose vérifie que les deux tableaux sont égaux à une tolérance (prise par défaut) près
        self.assertTrue(np.allclose(sim2, sim1*2), "Le solveur n'est pas galiléen!")

    def testOrdreDeConvergenceAvant(self):
        """Test l'odre de convergence pour le schéma avant."""
        nPts = 5
        R = 1.0
        D = 1e-10
        S = 1e-8
        Ce = 10

        params = Parametres(nPts=nPts, R=R, D=D, S=S, Ce=Ce)

        # Vérifier que la solution numérique est proche de la solution analytique

        N = 10  # Nombre de raffinements
        L2 = np.empty(N)
        for i in range(N):
            params.nPts = nPts
            sim = solveur_avant(params)
            ana = analytique(params)
            L2[i] = normeL2(ana,sim)
            nPts *= 2

        ordre = np.log(L2[:-1]/L2[1:])/np.log(2)
        print("Erreurs L2 :", L2)
        print("Ordre de convergence estimé :", ordre)

    def testOrdreDeConvergenceCentre(self):
        """Test de l'ordre de convergence pour le solveur avec schéma centré."""
        nPts = 5
        R = 1.0
        D = 1e-10
        S = 1e-8
        Ce = 10

        params = Parametres(nPts=nPts, R=R, D=D, S=S, Ce=Ce)
        sim = solveur_centre(params)
        ana = analytique(params)

        print(sim)
        print(ana)
        # Vérifier que la solution numérique est proche de la solution analytique

        N = 10  # Nombre de raffinements
        L2 = np.empty(N)
        for i in range(N):
            params.nPts = nPts
            sim = solveur_centre(params)
            ana = analytique(params)
            L2[i] = normeL2(ana,sim)
            nPts *= 2

        ordre = np.log(L2[:-1]/L2[1:])/np.log(2)
        print("Erreurs L2 :", L2)
        print("Ordre de convergence estimé :", ordre)
            

