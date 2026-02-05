import unittest
import numpy as np
from solveur import *

class TestSolveur(unittest.TestCase):
    def test_solveur(self):
        nPts = 10
        R = 1.0
        D = 1e-10
        S = 1e-8
        Ce = 10

        params = Parametres(nPts=nPts, R=R, D=D, S=S, Ce=Ce)
        sim = solveur(params)
        ana = analytique(params)

        # Vérifier que la solution numérique est proche de la solution analytique
        max_diff = np.max(np.abs(sim - ana))
        self.assertLess(max_diff, 1e-5, "La solution numérique diverge de la solution analytique.")
