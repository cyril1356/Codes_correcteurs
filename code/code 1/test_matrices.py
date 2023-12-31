import numpy as np
from encodage import *

G = np.array(
    [
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0,1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    ]
)
h_permu=construction_H_via_G(G)
print(h_permu)
print(calucul_distance(h_permu))

H_reduit = np.array(
    [

        [1, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [1, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [1, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
        [1, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    ]
)
s=np.ones((1,12))
s[0][0]=0
H=np.concatenate((s,H_reduit),axis=0)
print(np.dot(H,H.T))


nouveauxH=np.array([[1, 1, 0, 1, 1, 1,0, 0, 0, 1, 0]])
for i in range (10):
    V=np.zeros(11)
    ligne=nouveauxH[i]
    for j in range (len(nouveauxH[i])):
        if j==len(ligne)-1:
            V[0]=ligne[j]
        else:
            V[j+1]=ligne[j]
    print(V,"V")

    nouveauxH=np.concatenate((nouveauxH,[V]),axis=0)
print(nouveauxH,"h")
print(np.ones((11,1)))
nouveauxH=np.concatenate((nouveauxH,np.ones((11,1))),axis=1)

H_reduit = np.hstack((nouveauxH, np.identity(11)))
print(H_reduit)

print(calucul_distance(H_reduit))
G_reduit = np.array(
    [   
        
        [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
        [0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        [0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    ]
)

H_reduit = np.array(
    [
        
        [1, 1, 0, 1, 1, 1, 0, 0, 0, 1, 0],
        [1, 0, 1, 1, 1, 0, 0, 0, 1, 0, 1],
        [0, 1, 1, 1, 0, 0, 0, 1, 0, 1, 1],
        [1, 1, 1, 0, 0, 0, 1, 0, 1, 1, 0],
        [1, 1, 0, 0, 0, 1, 0, 1, 1, 0, 1],
        [1, 0, 0, 0, 1, 0, 1, 1, 0, 1, 1],
        [0, 0, 0, 1, 0, 1, 1, 0, 1, 1, 1],
        [0, 0, 1, 0, 1, 1, 0, 1, 1, 1, 0],
        [0, 1, 0, 1, 1, 0, 1, 1, 1, 0, 0],
        [1, 0, 1, 1, 0, 1, 1, 1, 0, 0, 0],
        [0, 1, 1, 0, 1, 1, 1, 0, 0, 0, 1],
    ]
)

print(calucul_distance_avec_poids_minimal(H_reduit, 5))


H_reduit = np.hstack((H_reduit, np.identity(11)))

# print(calucul_distance(construction_H_via_G(construction_martice_generalise_forme_systematique(120,127))))
print(calucul_distance(H_reduit))
