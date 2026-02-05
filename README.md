Pour run le code :

1. Dépendances : il faut avoir numpy et sympy
Je recommande de créer un environnement virtuel (un genre de poste de travail) pour tout installer ces dépendances. Dans le terminal de vscode :

  `python -m venv .venv`\\
  `.\.venv\Scripts\activate`\\
  `pip install numpy sympy`\\

Quand on veut sortir de l'environnement, toujours dans le terminal :

  deactivate

2. Exécuter les tests :
* Soit on clique directement sur run dans l'IDE
* Soit on roule les tests spécifiquement à partir du terminal
  ex : python -m unittest test.TestSolveur.testOrdreDeConvergenceAvant
  ou simplement : python -m unittest (ça run tout)
   
  
