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

  echo "L2,dr,dt" > resultats

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
    if test "$rep" = "e"; then
      sed "s/^[[:space:]]*nPts = .*/    nPts = $nPts/" $1.py > temp_$1.py
    else
      sed "s/^[[:space:]]*nPts = .*/    nPts = 100/" $1.py >> temp_$1.py
      sed -i "s/^[[:space:]]*nTime = .*/    nTime = $nPts/" temp_$1.py
    fi
    sed -i "s/^[[:space:]]*plt.show()//" temp_$1.py

    # exécuter le script et écrire les résultats dans le fichier résultats
    $PYTHON -m temp_$1 | awk '/L2 = / { L2= $3 } /dr = / {dr = $3} /dt = / {dt=$3} END {print L2 "," dr "," dt}'>> resultats



    #delete le temp file
    rm -f temp_$1/temp_$1.py

  done < $2 # from liste des resolutions

  # exécution du script pour la génération des graphiques
  # et le calcul de l'ordre
  $PYTHON -m analyse_convergence_implicite "$rep"

  rm temp_$1.py