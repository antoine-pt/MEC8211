Explication de la codebase (./projet_final) :

Il y a trois fichiers et un dossier imporants :
1. main.py : un exemple typique de l'utilisation du code
2. fonctions.py : le solveur. C'est le coeur du code
3. tests.py : le fichier comportant tous les tests réalisés pour la vérification de code / solution.
    pour run les tests : `python -m unittest.TestSolveur`
    pour test spécifique : `python -m unittest.TestSolveur.testSymetrie` ou tout autre test spécifique.
4. le dossier test_files contient des fichier de fonctions qui ne sont utiles qu'aux tests. Ils servent notamment à perturber le cadre de fonctionnement normal (ex.: translation des coordonnées) afin de pouvoir effectuer des tests de comparaison avec le comportement normal. Voir testInvarianceGalileenne pour un exemple.


Ce qu'il reste à faire :
1. MMS
2. Avec MMS, mesurer les ordres de convergence
3. Calculer le GCI
4. Potentiellement validation de solution avec simulation externe qu'on prend pour "données réelles"
5. Autre?