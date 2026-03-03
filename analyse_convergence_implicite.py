import matplotlib.pyplot as plt
import numpy as np
import csv
import sys

with open("resultats", mode="r", newline="") as file:   
    argument = sys.argv[1]
    csvr = csv.DictReader(file)
    L1 = []
    L2 = []
    Linf = []
    dh = []
    index = 0
    for row in csvr:

        # temps ou espace
        if argument == "e":
            dh.append(float(row["dr"]))
        elif argument == "t":
            dh.append(float(row["dt"]))
        else:
            print(argument)
            print("Missing input argument!")
            sys.exit(1)

        L1.append(float(row["L1"]))
        L2.append(float(row["L2"]))
        Linf.append(float(row["Linf"]))

        index+=1

    L1 = np.asarray(L1)
    L2 = np.asarray(L2)
    Linf = np.asarray(Linf)
    dh = np.asarray(dh)
    
    ordresL1 = np.log(L1[:-1]/L1[1:])/np.log(dh[:-1]/dh[1:])
    ordresL2 = np.log(L2[:-1]/L2[1:])/np.log(dh[:-1]/dh[1:])
    ordresLinf = np.log(Linf[:-1]/Linf[1:])/np.log(dh[:-1]/dh[1:])

    print(ordresL1)

    print(ordresL2)

    print(ordresLinf)

    string = ""
    for i in range(len(ordresL2)):
        string += f"Ordre de convergence L2 entre les résolutions {i} et {i+1} : {ordresL2[i]:.7f} \n"
    text = (string)

    plt.figure()
    plt.loglog(dh,L1,"x-", label="L1")
    plt.loglog(dh,L2,"x-", label="L2")
    plt.loglog(dh,Linf,"x-", label="Linf")
    plt.legend()
    plt.xlabel("pas de discrétisation")
    plt.ylabel("erreur")
    plt.title("Erreur de la solution numérique en fonction de la MMS pour un schéma centré")
    plt.grid("both", ls="--")
    plt.text(0.4, 0.75, text, fontsize=8, ha='center', va='center', transform=plt.gca().transAxes)
    plt.show()

