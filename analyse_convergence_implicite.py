import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

with open("resultats", mode="r", newline="") as file:   
    argument = sys.argv[1]
    csvr = csv.DictReader(file)
    L2 = []
    dh = []
    index = 0
    for row in csvr:
        L2.append(float(row["L2"]))
        # temps ou espace
        if argument == "e":
            dh.append(float(row["dr"]))
        elif argument == "t":
            dh.append(float(row["dt"]))
        else:
            print(argument)
            print("Missing input argument!")
            sys.exit(1)
        index+=1

    L2 = np.asarray(L2)
    dh = np.asarray(dh)
    
    ordres = np.log(L2[:-1]/L2[1:])/np.log(dh[:-1]/dh[1:])
    print(ordres)
    plt.plot(dh,L2,label="Some shit")
    plt.show()

    text = (f"Ordre de convergence L2 : {ordres[-1]:.7f} \n")

    plt.figure()
    plt.loglog(dh,L2,"x-", label="L2")
    plt.legend()
    plt.xlabel("pas de discrétisation")
    plt.ylabel("erreur")
    plt.title("Erreur de la solution numérique en fonction de la MMS pour un schéma centré")
    plt.grid("both", ls="--")
    plt.text(0.3, 0.2, text, fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()

