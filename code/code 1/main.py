from encodage import *



 
mot=encode('bpnjourjevaisbien')
print(mot)


mot_decoupe=grand_decoupage_numpy(mot,4)
H_reduit=np.array([[1,0,1,1,1,0,0],[1,1,0,1,0,1,0],[1,1,1,0,0,0,1]])

G=np.array([[1, 0, 0, 0] ,[ 0, 1, 0, 0],[ 0 ,0 ,1, 0 ],[ 0, 0, 0, 1 ], [1 ,0 ,1 ,1 ],[ 1, 1, 0, 1 ], [1, 1, 1, 0 ]]).T
print(G)
syndorme = crea_tab_sindorme_new(H_reduit,7)
print(syndorme)


mot_encode_decoupe=encode_decoupe(mot_decoupe,G)
mot_brouiller= brouillage(mot_encode_decoupe,40)

V=mot_brouiller
X=reparation(mot_brouiller,syndorme,H_reduit)

print(retour_mot(decriptage(V)),"-----mot reçus")
print(retour_mot(decriptage(X)),'-----mot reparer')