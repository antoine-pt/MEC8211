#!/usr/bin/env bash
set -euo pipefail

##################################
# DÉSACTIVER SI PAS SUR ORDI PERSO
# Activate virtual environment
PYTHON=.venv/Scripts/python
##################################

# messages
# --------
  USAGE="Usage: convergence {nom du solveur} {fichier contenant les nombres de points}"

  if test $# -ne 2
  then echo "Nombre d'arguments incorrect : $USAGE"
      exit
  fi

# Initialisations
  editor=nano
  nb_resolutions=`wc -l $2 | awk '{print $1}'`
  compteur=1

  echo ""
  echo "*******************************************"
  echo "        ANALYSE DE CONVERGENCE"
  echo "*******************************************"
  echo ""
  echo "--> Nombre de résolutions testées : "$nb_resolutions
  echo ""

  echo "L1,L1Final,L2,L2Final,Linf,nPts,dr,dt" > resultats

  if test -s $1.py ; then
	  echo "--> Le fichier "$1" existe !"
	  echo ""
          echo "Editer le fichier "$1".cpp (o-n) ?"
          read rep
               if test "$rep" = "o"; then
		       $editor $1.py
               fi
  else echo "ALERTE: Le fichier "$1".py n'existe pas ! Sortie du script..."
       exit
  fi

  #
  # Boucle principale
  #

  echo ""
  echo "Convergence en temps ou en espace (t-e) ?"
  read rep
    

  while read -r nPts; do

    # change nPts dans le ficher
    # stream editor "substitution / pattern / replacement/" > overwrite in temp_$1.py 

    sed "s/^[[:space:]]*D = .*/    D = 1/" $1.py > temp_$1.py
    sed -i "s/^[[:space:]]*k = .*/    k = 4/" temp_$1.py

    if test "$rep" = "e"; then
      sed -i "s/^[[:space:]]*nPts = .*/    nPts = $nPts/" temp_$1.py
      sed -i "s/^[[:space:]]*nTime = .*/    nTime = 1000/" temp_$1.py
    else
      sed -i "s/^[[:space:]]*nPts = .*/    nPts = 1000/" temp_$1.py
      sed -i "s/^[[:space:]]*nTime = .*/    nTime = $nPts/" temp_$1.py
    fi
    
    sed -i '/^[[:space:]]*plt.figure()/,$d' temp_$1.py
    # exécuter le script et écrire les résultats dans le fichier résultats
    $PYTHON -m temp_$1 | awk '
    /^L1 = / { L1 = $3 }
    /^L1Final = / { L1Final = $3 }
    /^L2 = / { L2 = $3 }
    /^L2Final = / { L2Final = $3 }
    /^Linf = / { Linf = $3 }
    /^nPts = / { nPts = $3 }
    /^dr = / { dr = $3 }
    /^dt = / { dt = $3 }
    END { print L1 "," L1Final "," L2 "," L2Final "," Linf "," nPts "," dr "," dt }
    ' >> resultats



    #delete le temp file
    rm temp_$1.py

  done < $2 # from liste des resolutions

  # exécution du script pour la génération des graphiques
  # et le calcul de l'ordre
  $PYTHON -m analyse_convergence_implicite "$rep"