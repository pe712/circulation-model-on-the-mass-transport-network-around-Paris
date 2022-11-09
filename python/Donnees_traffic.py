import sqlite3 as sql
import numpy as np
import matplotlib.pyplot as pl
from traitement_BDD import creation_liste_dest, correction_directs_ligne, facteur_population


#le route_id existe-il
def test_existence(c, route_id):
    c.execute('select count(*) from routes where route_id="{}"'.format(route_id))
    if c.fetchone()[0]==1:
        return True
    else:
        return False


##données de database - creation de points, distances, gares,
def creer_points_id(c, c2, route_id):
    gares=[]
    c.execute('''
    select distinct from_id_groupe
    from graphe
    join stops_groupe
        on stops_groupe.id_groupe=graphe.from_id_groupe
    where route_id="{}"
    order by from_id_groupe'''.format(route_id))
    points=[]; liste_id=[]; gares=[]
    compt=0
    for from_id_groupe in c:
        id_g=from_id_groupe[0]
        liste_id.append(id_g)
        c2.execute('select x,y from stops_groupe where id_groupe={}'.format(id_g))
        x, y = c2.fetchone()
        points.append([x,y])
        gares.append([[], []]) #gare=[quais, frequentations]
        compt+=1
        print(id_g)
    return liste_id, points, gares

def creer_distances_graphe_gares(liste_id, c, route_id, gares):
    c.execute('select from_id_groupe, to_id_groupe, distance from graphe where route_id="{}"'.format(route_id))
    liste=c.fetchall()
    n=len(liste)
    print(n, 'liaisons')
    N=len(liste_id)
    print(N, 'gares')
    distances=np.zeros((N,N))
    compt=0
    for from_id_groupe, to_id_groupe, distance in liste:
        [numgare1, numgare2] = correspondance_DB_traffic([from_id_groupe, to_id_groupe], liste_id)
        distances[numgare1, numgare2] = distance
        nb_voies=1;
        gares[numgare1][0].append([numgare2, 0, nb_voies])
        compt+=1
        if compt%10==0:
            print(compt)
    gares=correction_nb_voies(gares)
    return distances, gares

# rajoute des gares fictives qui permette de faire sortir du modèle les trains en fin de service
def voies_garages(gares, dest, liste_id, c):
    compt=0
    for destination in dest:
        num_terminus = destination[-2] #destination[-1]=-1 fin de service
        if gares[num_terminus][0][-1]!=[-1, 0, 1]:
            gares[num_terminus][0].append([-1, 0, 1]) #voie de garage
    return gares

# 2 voies par direction dans les gares ayant plus de deux directions (fourches)
def correction_nb_voies(gares):
    for numgare in range(len(gares)):
        nb_dir=len(gares[numgare])
        if nb_dir>2: #plus de 2 dir
            for numdir in range(nb_dir):
                gares[numgare][numdir][2]=2
    return gares




## calcul de la distribution des flux à l'échelle d'une ligne - transit
'''flux_emis=flux_recu=frequentation_m*facteur_population
ceci est logique en considérant qu'il s'agit de flux domicile-travail qui sont donc symétrique
il s'agit de déterminer alpha, beta et K pour que la répartition soit correcte
En particulier il faut que la somme des flux_ij sur i soit Ei et la somme des flux_ij sur j soit Aj

'''

alpha=-0.0001
beta=-0.0003 #résultats meilleurs que la distance moyenne

def calcul_de_beta():
    conn=sql.connect(r"F:/informatique/TIPE/database/produit exploitable/GTFS.db")
    c=conn.cursor()
    c.execute('select avg(distance) from graphe')
    dist_moy=c.fetchone()[0]
    conn.close()
    return 1/dist_moy
#renvoie 0.003


def init_transit(route_id):
    conn=sql.connect(r"F:/informatique/TIPE/database/produit exploitable/GTFS.db")
    c=conn.cursor()
    c2=conn.cursor()
    liste_id, points, gares = creer_points_id(c, c2, route_id)
    return liste_id, points, c, c2, conn

def maj_transits(route_id='810:B'):
    liste_id, points, c, c2, conn = init_transit(route_id)
    distribution_transits(liste_id, points, route_id, c, c2, conn)
    conn.close


#le graphe ne renseigne que les gares directement reliées, il faut prendre en compte toutes les stations de la ligne
#la distance utilisé pour calculer l'impédance est à vol d'oiseau car il est inutile et compliqué de calculer en cumulé selon les stations reliées
def distribution_transits(liste_id, points, route_id, c, c2, conn):
    print('calcul transit... \n')
    c.execute('delete from flux_transit')
    n=len(liste_id)
    for i in range(n):
        id_grp1 = liste_id[i]
        xi, yi = points[i]
        c2.execute('select frequentation_m from flux_emis where id_groupe="{idg}" and route_id="{idr}"'.format(idg=id_grp1, idr=route_id))
        Vi=c2.fetchone()[0]
        Ei=Vi*facteur_population #flux émis par la gare i
        flux_emis_j=0
        for j in range(n):
            if i!=j:
                id_grp2 = liste_id[j]
                xj, yj = points[j]
                dist = np.sqrt((yj-yi)**2+(xj-xi)**2)
                c2.execute('select frequentation_m from flux_emis where id_groupe="{idg}" and route_id="{idr}"'.format(idg=id_grp2, idr=route_id))
                Vj=c2.fetchone()[0]
                Aj=facteur_population*Vj
                impedance=dist**alpha*np.exp(beta*dist)
                flux_ij=Ei*Aj*impedance
                flux_emis_j+=flux_ij
                c.execute('insert into flux_transit(from_id_groupe, to_id_groupe, deplacements, route_id) values(?,?,?,?)', (id_grp1, id_grp2, int(flux_ij), route_id))
        K=Ei/flux_emis_j
        c2.execute('update flux_transit set deplacements=deplacements*{K} where from_id_groupe={idg1}'.format(K=K, idg1=id_grp1))
    conn.commit()
    print('done \n')

def comparaison_params():
    global alpha, beta
    for alpha in [-0.5, -0.7, -1]:
        for beta in [-0.001, -0.02, -0.01]:
            maj_transits(False)



##représentation des transits
def repr_transit(maj, route_id='810:B', id_groupe=8824):
    liste_id, points, c, c2, conn = init_transit(route_id)
    if maj:
        distribution_transits(liste_id, points, route_id, c, c2, conn)
    representation_transits(points, liste_id, c, c2, id_groupe, route_id)
    conn.close

def representation_transits(points, liste_id, c, c2, id_groupe, route_id):
    c2.execute('select stop_name from stops_groupe where id_groupe={}'.format(id_groupe))
    nom_gare = c2.fetchone()[0]
    fig_flux, [ax_pie, ax_graph] = pl.subplots(1, 2, figsize=[23, 16])
    c.execute('select sum(deplacements) from flux_transit where from_id_groupe="{idg}" and route_id="{idr}"'.format(idg=id_groupe, idr=route_id))
    total=c.fetchone()[0]  #on peut aussi récupérer frequentation_m dans flux_emisen multipliant par facteur_population
    taux = total/50
    c.execute('''
    select deplacements, stop_name
    from flux_transit
    join stops_groupe
        on stops_groupe.id_groupe=flux_transit.to_id_groupe
    where flux_transit.from_id_groupe={}
    '''.format(id_groupe))
    labels=[]; sizes=[]
    for dep, stop_name in c:
        if dep>taux:
            labels.append(stop_name)
            sizes.append(dep)
            print(dep)
    patches, text, other = ax_pie.pie(sizes, autopct = lambda x: str(round(x, 2)) + '%')
    title='representation transit au départ de ' + str(nom_gare) + ' (' + str(int(total))+' voyageurs par jour)'
    ax_pie.set_title(title)
    ax_pie.legend(patches, labels, loc='lower right')
    ax_graph.set_title("évolution de l'impédance en fonction de la distance")
    X=np.arange(0.1, 10, 0.001)
    Y=[]
    for x in X:
        dist=x*1000
        impedance=dist**alpha*np.exp(beta*dist)
        Y.append(impedance)
    ax_graph.plot(X, Y)
    ax_graph.set_xlabel('distance en km')
    pl.savefig(title, dpi=300,  bbox_inches='tight')
    pl.show()


##remplissage et fréquentations
def remp_freq(dest, gares, liste_id, c):
    c.execute('select from_id_groupe, to_id_groupe, deplacements from flux_transit')
    for id_grp1, id_grp2, deplacements in c:
        print(id_grp1, id_grp2)
        [numgare1, numgare2] = correspondance_DB_traffic([id_grp1, id_grp2], liste_id)
        condition=False
        for destinat in dest:
            if numgare1 in destinat and numgare2 in destinat: #il existe un train qui emmenera les voyageurs de a à b
                condition=True
        if condition and deplacements>10: #on évite de surcharger avec des flux minimes
            gares[numgare1][1].append([numgare2, 0, deplacements, 0])



##outils
# transforme un id_groupe en son équivalent pour le traffic et réciproquement
def correspondance_traffic_DB(liste_id_traffic, liste_id):
    conn=sql.connect(r"F:/informatique/TIPE/database/produit exploitable/GTFS.db")
    c=conn.cursor()
    rep=[]
    for indice in liste_id_traffic:
        id_groupe=liste_id[indice]
        c.execute('select stop_name from stops_groupe where id_groupe={}'.format(id_groupe))
        rep.append([id_groupe, c.fetchone()[0]])
    conn.close()
    return rep

def correspondance_DB_traffic(liste_id_groupe, liste_id):
    rep=[]
    for id_groupe in liste_id_groupe:
        for indice in range(len(liste_id)):
            if id_groupe==liste_id[indice]:
                rep.append(indice)
    return rep

def verif_dessin(dest, points):
    ax=pl.axes(xlim=(630000,650000),ylim=(670000,680000))
    ax.axis('equal')
    for i in range(len(dest)):
        ligne=dest[i]
        for gare in ligne:
            if i==0:
                col='blue'
            else:
                col='red'
            ax.plot(points[gare][0],points[gare][1],marker='+',color=col)
    pl.show()

def somme(liste):
    somme=0
    for elmt in liste:
        somme+=elmt
    return somme

def relatif(liste):
    total=somme(liste)
    new_liste=[]
    for elmt in liste:
        new_liste.append(elmt/total)
    return new_liste

##feuille de route et horaires
'''' on créera d'abord un type/horaire par train existant puis on les rassemblera'''

'''on ne prend que
-les trains non exceptionnels (plus de 5 qui ont le même itinéraire)
-circulant à la date donnée
-pour le route_id'''


def selection_trips_ids(route_id, c):
    c.execute('''
    select trips.trip_id
    from stop_times
    join trips
        on trips.trip_id=stop_times.trip_id
    join calendar
        on calendar.service_id=trips.service_id
            where trips.route_id="{}"
            and stop_times.stop_sequence=1
            and calendar.start_date<20200522 and calendar.stop_date>20200522
    group by stop_times.stop_id, stop_times.departure_time
    order by stop_times.stop_id, stop_times.departure_time
        '''.format(route_id))
    tup=()
    for trip_id in c:
        tup+=trip_id
    print("trip_id selectionnées")
    return tup

#☻pour remplir les colonnes inutiles (route_type et route_id) on selectionnera des chiffres
def creation_dest(c, selection_trips):
    c.execute('''
    select 1, 2, groupe.id_groupe, stop_times.stop_sequence
    from stop_times
    join groupe
        on stop_times.stop_id=groupe.stop_id
    where stop_times.trip_id in {s}
    order by stop_times.trip_id desc, stop_times.stop_sequence asc

    '''.format(s=selection_trips))
    liste_dest=creation_liste_dest(c)
    dest=liste_dest[0][2]
    print('dest full ok\n')
    return dest

#on utilise également creation_liste_dest pour regrouper les horaires de passage d'un même trip
#l'horaire de départ sera utilisé dans horaires tandis que tous seront utilisés dans stop_times0 pour pouvoir calculer ponctualite()
def creation_horaires(c, selection_trips):
    c.execute('''
    select 1, stop_times.trip_id, stop_times.departure_time, stop_times.stop_sequence
    from stop_times
    where stop_times.trip_id in {s}
    order by stop_times.trip_id desc, stop_times.stop_sequence asc
    '''.format(s=selection_trips))
    liste_horaires=creation_liste_dest(c)
    horaires_full=[]
    stop_times0=[]
    for trip in liste_horaires:
        trip_id=trip[1]
        dep_times=trip[2][0]
        dep_times_corr=[]
        for h_m in dep_times:
            h_m=h_m.split(':')
            dep_times_corr.append(float(h_m[0])+float(h_m[1])/60)
        horaires_full.append([dep_times_corr[0], trip_id])
        stop_times0.append([trip_id, dep_times_corr])
    print('horaires full ok\n')
    return horaires_full, stop_times0

#on regroupe les dest par trip et on les joint à leurs horaires par ind qui deviendra le no_feuille_route
def reunion(dest_full, horaires_full):
    dest=[]; horaires=[]
    for indice_full in range(len(dest_full)):  # len(horaires_full)=len(dest_full)
        n=len(dest)
        indice=0
        while indice<n and dest_full[indice_full]!=dest[indice]:
            indice+=1
        if indice==n: #nouvelle feuille de route
            dest.append(dest_full[indice_full])
            horaires.append([len(dest)-1] + horaires_full[indice_full])
        else:
            horaires.append([indice] + horaires_full[indice_full])
    return dest, horaires


def creation_direct(dest):
    chgmt=1
    direct=[[] for k in range(len(dest))]
    while chgmt!=0:
        dest, chgmt, direct = correction_directs_ligne(dest, True, direct)
    return dest, direct

#on remplace tous les id_groupe par des numgare allant de 1 à n
def convertir(dest, direct, liste_id):
    n=len(dest)
    dest_traffic=[]; direct_traffic=[]
    for indice in range(n):
        new_destin=correspondance_DB_traffic(dest[indice], liste_id)
        new_destin.append(-1)
        dest_traffic.append(new_destin)
        direct_traffic.append(correspondance_DB_traffic(direct[indice], liste_id))
    return dest_traffic, direct_traffic

##tri horaires
#mise en place d'un quicksort qui agit in place en utilisant la première valeur comme pivot
def partition(horaires, p, r):
    piv = horaires[p][1]
    j=p # horaires[j] est le dernier élément lu qui est plus petit que piv
    for i in range(p+1, r):
        if horaires[i][1]<=piv:
            j+=1
            echange(horaires, i, j)
    echange(horaires, p, j)
    return j

def tri_rapide_rec(horaires, p, r):
    if r>p+1:
        q = partition(horaires, p, r)
        tri_rapide_rec(horaires, p, q)
        tri_rapide_rec(horaires, q+1, r)

def quicksort(horaires):
    n = len(horaires)
    tri_rapide_rec(horaires, 0, n)

def echange(liste, i, j):
    if i!=j:
        liste[i], liste[j] = liste[j], liste[i]


##types
# [position,vitesse_actuelle, tps_attendu_arret, gar_départ, gar_arrivée, remplissage en personnes, no_feuille de route]
def creation_types(dest):
    modele_train=[0, 0, 0, 0, -1, 'terminus à remplir', [], 'no feuille']
    n=len(dest)
    types=[]; direct=[]
    for no_feuille in range(n):
        term_depart=dest[no_feuille][0]
        dest[no_feuille].remove(term_depart)
        new_type=modele_train.copy()
        new_type[5]=term_depart; new_type[7]=no_feuille
        types.append(new_type)
    return types, dest

##affichage
def cadre_num_max_trains(horaires, c, route_id):
    num_max_trains=int(len(horaires)/10) #approximation du  nombre simultanné de trains
    c.execute('''
    select min(x), min(y), max(x), max(y)
    from stops_groupe
    join graphe
        on graphe.from_id_groupe=stops_groupe.id_groupe
    where graphe.route_id="{}"'''.format(route_id))
    (minx, miny, maxx, maxy) = c.fetchone()
    cadre=[[minx-2000, maxx+2000], [miny-20000, maxy+20000]]
    return num_max_trains, cadre

##étalement horaire
def etalement(horaires):
    plages=[0]*24
    N=len(horaires)
    for hor in horaires:
        t_dep=hor[1]
        num_p=int(t_dep)
        plages[num_p]+=1
    for k in range(24):
        plages[k]/=N
    return plages

##variables
def variables(route_id):
    global liste_id
    conn=sql.connect(r"F:/informatique/TIPE/database/produit exploitable/GTFS.db")
    c=conn.cursor()
    c2=conn.cursor()
    liste_id, points, gares = creer_points_id(c, c2, route_id)
    distances, gares = creer_distances_graphe_gares(liste_id, c, route_id, gares)
    #on récupère toutes les informations de la ligne dans stop_times, on a : len(dest_full)=len(horaires_full)
    selection_trips=selection_trips_ids(route_id, c)
    dest_full=creation_dest(c, selection_trips)
    horaires_full, stop_times0=creation_horaires(c, selection_trips)
    #on les réduit
    dest, horaires = reunion(dest_full, horaires_full)
    #on trie les horaires
    quicksort(horaires)
    #on calcule la répartition des flux
    plages = etalement(horaires)
    #on modifie dest pour corriger les directs
    dest, direct = creation_direct(dest)
    #on transforme avec les id du traffic
    dest, direct = convertir(dest, direct, liste_id)
    #on crée les types et voies de garage
    voies_garages(gares, dest, liste_id, c)
    types, dest = creation_types(dest)
    #affichage
    num_max_trains, cadre = cadre_num_max_trains(horaires, c, route_id)
    #on rajoute dans gares les freq
    remp_freq(dest, gares, liste_id, c)
    conn.close()
    return points, liste_id, distances, dest, types, horaires, gares, direct, num_max_trains, cadre, stop_times0, dest_full, plages


