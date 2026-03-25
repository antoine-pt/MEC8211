#!/usr/bin/env bash
set -euo pipefail

PYTHON=.venv/Scripts/python

USAGE="Usage: convergence_devoir_3 {nom du solveur}"

if test $# -ne 2
then echo "Nombre d'arguments incorrect : $USAGE"
    exit
fi


echo ""
echo "*******************************************"
echo "        ANALYSE DE CONVERGENCE - DEVOIR 3"
echo "*******************************************"
echo ""
echo "--> Nombre de résolutions testées : 6"
echo "--> Nombre de seeds testés : 3"
echo ""

solver="$1"

if ! test -s "${solver}.py"; then
    echo "ERROR: File ${solver}.py does not exist."
    exit 1
fi

seeds=(101 202 303)
NX_list=(50 75 100 150 200 400)

h_value="2e-4"

results_csv="results_k.csv"
logs_dir="logs_mesh_runs"

mkdir -p "$logs_dir"

echo "seed,NX,dx,k_in_micron2" > "$results_csv"

for seed in "${seeds[@]}"; do
    echo "==========================================="
    echo "Seed = $seed"
    echo "==========================================="

    for NX in "${NX_list[@]}"; do

        # calcul de dx pour l'affichage et pour le fichier résultats
        dx=$(awk -v h="$h_value" -v NX="$NX" 'BEGIN { printf "%.16e", h/NX }')

        temp_file="temp_${solver}.py"
        log_file="${logs_dir}/seed_${seed}_NX_${NX}.log"

        cp "${solver}.py" "$temp_file"

        # Disable plotting
        sed -i '1i import matplotlib; matplotlib.use("Agg")' "$temp_file"
        sed -E -i 's/plt\.show[[:space:]]*\([^)]*\)/pass/g' "$temp_file"

        # Mise à jour des paramètres dans le fichier temporaire
        sed -i "s/^[[:space:]]*seed[[:space:]]*=.*/    seed         = ${seed}/" "$temp_file"
        sed -i "s/^[[:space:]]*h[[:space:]]*=.*/    h = ${h_value}/" "$temp_file"
        sed -i "s/^[[:space:]]*NX[[:space:]]*=.*/    NX           = ${NX}/" "$temp_file"

        echo "--> Running seed=$seed, NX=$NX, dx=$dx"

        "$PYTHON" "$temp_file" > "$log_file" 2>&1  # 2>&1 pour rediriger stderr vers le même fichier que stdout

        # Pour aller récupérer la valeur de k sans le mu
        k_val=$(grep -a -oE 'k_in_micron2[[:space:]]*=[[:space:]]*[0-9.eE+-]+' "$log_file" | tail -n 1 | grep -oE '[0-9.eE+-]+$')

        echo "${seed},${NX},${dx},${k_val}" >> "$results_csv"

        rm -f "$temp_file"
    done
done

echo ""
echo "Fin."
echo "Resultats dans : $results_csv"
echo "Logs dans :    $logs_dir"

"$PYTHON" GCI.py