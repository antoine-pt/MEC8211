Utilisation du code :

Dépendances : sympy, matplotlib, numpy

======================

Comment exécuter en ligne de commande :

1. Se situer à la racine du projet.
2. python -m dossier.fichier, exemples :
    * python -m src.main
    * python -m GCI.GCI
3. Pour les unittests : python -m unittest src.tests
4. pour des tests en particulier : python -m unittest src.tests.TestSolveur.<nomDuTest> 
    * python -m unittest src.tests.TestSolveur.testSolveur
    * python -m unittest src.tests.TestSolveur.testMilieu

Note : le fichier src/convergence_generate_data.py peut prendre quelques temps à exécuter puisqu'il génère beaucoup de données. Les données générées sont indipensables au calculs des ordres de convergence.

Certains programmes requièrent un input de la part de l'utilisateur (dans le terminal).

D'autres détails sur les différents fichiers sont présents dans le rapport PowerPoint associé. 