import numpy as np
import matplotlib.pyplot as pl
import random as rd
import matplotlib.animation as animation
import sqlite3 as sql
from Donnees_traffic import variables, correspondance_traffic_DB
from matplotlib.widgets import Button
from algorithmes_de_minimisation import minimum, monteCarlo
import copy


route_id="810:B"
## Creation et initialisation du maillage
"""
dt en minutes

points=[[x1, y1], [x2, y2], ... ]
distances= matrice n*n diagonale ij = distances entre i et j
si 0 alors non reliées

gares=[[gare1], [gare2], ... ]
gare=[occupations_des_voies, fréquentations]
occupations_des_voies=[[direction(vers où), nombre de voies occupées, nombre de voies total], [dir2, n2, nprim2]...]
fréquentations=[direction, passagers_en_gare,  passagers_quotidiens, dernière heure de desserte]

dest=[[feuille de route1], [feuille2], ...]
feuille de route=[gares par lesquelles passer]
direct= meme format que dest -> gares ou le train est sans arret

types=[[typ1], [typ2], ...]

trains contient des typ
typ=[position,vitesse_actuelle, acceleration, tps_attendu_arret, gar_départ, gar_arrivée, remplissage, no_feuille_route = no_feuille_direct]
    si gar_depart=-1 alors le train est arrivé en gar_arrivé
    la pos est en proportion de la distance, la vit est absolue
    il y en a un par feuille de route

remplissage=[[gare de destination, nombre de personnes], [gare2, n2], ...]


destinations contient des dest
dest=[gare1, gare2...]

sans_arret contient des direct
direct=[gare_évitée1, gare_évitée2 ...]

Les deux fonctionnent en paire

arret contient des numtrains dont l'arret est contraint par un autre train
arret_force contient des numtrains dont l'arret est contraint par une perturbation (dans le but d'en évaluer l'impact)

fin_de_service contient des  [numtrain,numgare] allant être supprimés en numgare

horaires=[[type de train, horaire de départ, (trip_id1)], [typ2, horaire2, (trip_id2)], ...]

Attention, stop_times et stop_times0 ne son pas au même format

stop_times=[[numtrain, 'trip_id', passages(liste)], [numtrain2, 'trip_id2', passages(liste)], ... ]
passages=[numgare, heure, voy_montants, minutes_attendues]
stop_times_stock est similaire à stop_times mais avec des id_groupe à la place des numgare

stop_times0=[trip1, trip2 ...]
trip=[trip_id, dep_times]
dep_times=[h1, h2 ...]

profil_suivi=[intergare1, intergare2, ...]
intergare=[gare_dep, gare_ar, valeurs1, valeurs2, ...]
valeur=[pos, heure, vit, accel, remplissage] avec remplissage correspondant au contenu du train numtrain_course

"""

def variables_neutres():
    global destinations, arret, ralentissement, trains, fin_de_service, sans_arret, timer, pause, service, erreurs, stop_times, stop_times_stock, arret_force, affluence, annote, profil, profil_suivi, condition_arret_prgm, numtrain_course, pert_on, dessin, affichage, consigne_pert, exemple
    destinations=[]; arret=[]; ralentissement=[]; trains=[]; fin_de_service=[]; sans_arret=[]; timer=False; pause=False; service=True; erreurs=[]; stop_times=[]; stop_times_stock=[];  arret_force=[]; annote=False; affluence=False; profil=False; profil_suivi=[]; condition_arret_prgm=False; numtrain_course=-1; pert_on=False; dessin=False; affichage=True; consigne_pert=False; exemple=False

#il faut copier le contenu de chaque sous_liste de gares
#pas besoin pour horaires qui n'est pas modifié (seulement suppr)
def etat_initial():
    variables_neutres()
    global horaires, gares, zeros
    gares = copy.deepcopy(gares0)
    horaires = horaires0.copy()
    zeros=[0]*len(gares)

## donnees GTFS
#on peut choisir de lancer le modèle avec des paramètres différents
def donnees_GTFS(route_id = route_id):
    global dimension, annote
    dimension = 600
    annote = False
    global points, liste_id, distances, dest, types, horaires0, gares0, direct, num_max_trains, cadre, stop_times0, dest_full, plages
    # try:
    points, liste_id, distances, dest, types, horaires0, gares0, direct, num_max_trains, cadre, stop_times0, dest_full, plages = variables(route_id)
    # except Exception:
        # print('erreur de donnée')


##global
# les données sont sauvegardees au débur pour ne pas les recalculer
def demarrage():
    etat_initial()
    global affichage, dessin, service
    affichage=True; dessin=True; service=True
    evolution_traffic(6, 0.1, 8, 20)
'''
num_h=4  départ à 6.0   trip_id_course='115072256-1_14393'; duree=0.2; tm=6.2; g1=34; g2=37
num_h=51  départ à 7.38  trip_id_course='115054849-1_16156'; duree=0.2; tm=7.5; g1=30; g2=29
'''
def profil_course():
    etat_initial()
    global profil, trip_id_course, dessin, service, exemple
    profil = True; dessin=True; service=False; exemple=False
    trip_id_course='115054849-1_16156'
    evolution_traffic(5, 0.1, 23, 200)
    comparaison(trip_id_course)


def evaluation_pert():
    etat_initial()
    global pert_on, dessin, service, profil, trip_id_course, consigne_pert
    pert_on=True; dessin=False; service=False; profil=True; consigne_pert=False
    trip_id_course='115054849-1_16156'; duree=0.2; tm=7.5; g1=30; g2=29
    variables_de_pert(duree, tm, g1, g2)
    evolution_traffic(5.0, 0.1, 23, 20)
    comparaison(trip_id_course)
    # reecriture_DB()
    # ecart_db(True)


#fait une itération du modèle pour les paramètres données, il s'agit de la fonction à minimiser
#on ne calcule qu'une seule course
def retour_ecart(variabs):
    etat_initial()
    global vitesse_nominale, vitesse_arrivee, contenance, tps_arret, tps_population
    [vitesse_nominale, vitesse_arrivee, contenance, tps_arret, tps_population]=variabs
    global affichage, horaires, service, dessin
    affichage=False; service=False; dessin=False
    horaires=[horaires0[51]] #train quelconque '115054849-1_16156'
    evolution_traffic(7.2, 0.1, 9, 100)
    ec = ponctualite()
    return ec

def evolution_traffic(t0, dt1, t_fin, time_par_frame):  #dt en minutes, t_fin et t_depart en heures
    global t_depart, dt
    delta_t = t_fin-t0
    t_depart = t0
    dt = dt1
    n = int(delta_t//(dt/60))
    if affichage:
        print(n, 'frames')
    initialisation_horaires(t0)
    if dessin:
        creation_artists(num_max_trains, cadre[0], cadre[1])
        global ani
        ani = animation.FuncAnimation(fig1, anim, fargs=(5, ), init_func = init, frames = n, blit = False, interval = time_par_frame, repeat = False)
        pl.show()
    else:
        for i in range(n):
            if condition_arret_prgm:
                break
            heure = t_depart+i*dt/60
            temps_suivant(heure)
            if i%50==0 and affichage:
                print('temps:', conversion_h_m(heure))
    stop_times_stock.extend(stop_times)
    statistiques_attente()

##Avancement d'une durée dt
def temps_suivant(heure):
    nb_trains = len(trains)
    mise_en_service(heure)
    for numtrain in range(nb_trains):
        if est_en_gare(numtrain):
            train_gare(numtrain, heure)
        else:
            train_voie(numtrain, heure)
    execute_fin_service()
    gestion_arret_force(heure)
    if profil:
        gestion_profil(heure)
    if pert_on:
        mise_en_place_pert(heure)


##gares
def est_en_gare(numtrain):
    if trains[numtrain][4]==-1:
        return True
    else:
        return False

def train_gare(numtrain, heure):
    gare1 = trains[numtrain][5]
    if len(destinations[numtrain])<=1:  #fin de service à la fin (modif des numtrains)
        fin_de_service.append([numtrain, gare1])
    else:
        gare2 = destinations[numtrain][0]
        ind = recherche_indice_quai(gare1,gare2)
        if ind is None:
            print('suppression du train', numtrain)
            erreurs.append([numtrain, 'destinations incohérentes de', gare1, 'à', gare2])
            fin_de_service.append([numtrain, gare1])
        else:
            personnes = voyageurs_quai(gare1, numtrain)
            if condition_redemarrage_gare(numtrain, personnes, gare1, gare2):
                execute_train_redemarrage(numtrain, gare1, gare2, personnes, ind, heure)
            else:
                execute_train_attente(numtrain)

#condition de tps d'attente: 30 sec +1 min toute les 1000 pers en gare
#condition voie libre: pas de trains dans la premiere moitié
def condition_redemarrage_gare(numtrain, personnes, gare1, gare2):
    pos_liaison = distance_train_proche(gare1, gare2, numtrain)
    if len(pos_liaison)>0:
        pos_proche = min(pos_liaison)
    else:
        pos_proche = 1 #pas de train
    if pos_proche>0.6 and trains[numtrain][3]>tps_arret+personnes*tps_population and (numtrain not in arret_force):
        return True
    else:
        return False


#on récupère les positions de tous les trains sur la liaison excepté le numtrain considéré (utile pour un train en voie)
def distance_train_proche(gare1, gare2, numtrain1):
    pos_proche=[]
    for numtrain in range(len(trains)):
        if trains[numtrain][4]==gare1 and trains[numtrain][5]==gare2 and numtrain!=numtrain1:
            pos = trains[numtrain][0]
            pos_proche.append(pos)
    return pos_proche


def recherche_indice_quai(gare1,gare2):
    occup=gares[gare1][0]
    for ind in range(len(occup)):
        if occup[ind][0]==gare2:
            return ind
    print('erreur, ce train souhaite aller de', gare1, 'à', gare2)

#on ne fait monter et descendre les voyageurs qu'au redémarrage car ces derniers interviennent dans les calculs de temps d'attente en gare
def execute_train_redemarrage(numtrain, gare1, gare2, personnes, ind, heure):
    trains[numtrain][4:6]=[gare1,gare2]
    trains[numtrain][1]=0     #sans vitesse initiale
    trains[numtrain][3]=heure                           #reset temps d'attente
    gares[gare1][0][ind][1]-=1  #libère un quai
    monter_voyageurs(numtrain,gare1, heure)

def execute_train_attente(numtrain):
    trains[numtrain][3]+=dt

##voies
def train_voie(numtrain, heure):
    [pos,vit] = trains[numtrain][0:2]
    gare1, gare2 = int(trains[numtrain][4]), int(trains[numtrain][5])
    dist = distances[gare1, gare2]
    gare3 = destinations[numtrain][1]
    ind = recherche_indice_quai(gare2,gare3)
    if ind is None or gare1==gare2:
        print('suppr')
        erreurs.append([numtrain, 'destinations incohérentes de', gare2, 'à', gare3])
    else:
        test_securite(numtrain, pos, gare1, gare2, gare3, ind, dist)
        #traite tous les cas d'arret, de ralentissement et d'arret forcé par la pert
        vit = trains[numtrain][1]
        accel = trains[numtrain][2]
        if (numtrain not in arret) and (numtrain not in ralentissement) and (numtrain not in arret_force):
            if arrive_en_gare(pos, vit, dist):
                if gare2 in sans_arret[numtrain]:
                    execute_train_direct_gare(numtrain,gare2,gare3)
                else:
                    execute_train_arrive_gare(numtrain, gare2, ind, heure, gare3)
            else:
                execute_train_avancer(numtrain, pos, vit, accel, dist)


def execute_train_arrive_gare(numtrain, gare2, ind, heure, gare3):
    augmenter_voyageurs_gares(gare2, numtrain, heure)
    descendre_voyageurs(numtrain,gare2)
    trains[numtrain][0:3]=[0,0,0]    #arret
    trains[numtrain][3]=0  #tps attente
    trains[numtrain][4]=-1
    trains[numtrain][5]=gare2
    remplissage = trains[numtrain][6]
    gare3 = int(destinations[numtrain][1])
    gares[gare2][0][ind][1]+=1 #occupe un quai
    destinations[numtrain].remove(gare2)
    suivi_ajout_passage(numtrain, gare2, heure)

def execute_train_direct_gare(numtrain,gare2,gare3):
    trains[numtrain][4:6]=[gare2,gare3]
    trains[numtrain][0]=0
    destinations[numtrain].remove(gare2)
    sans_arret[numtrain].remove(gare2) #pas utile mais plus clair

#la régulation de la dynamique du train se fait sur le couple qui est proportionnel à l'accélértion
#couple de démarrage constant-> montée linéaire de couple jusqu'à vitesse_nominale ou couple_nominal -> couple nul-> couple négatif de freinage
def execute_train_avancer(numtrain, pos, vit, accel, dist):
    distance=pos*dist
    if distance<distance_demarrage:
        accel1=accel_nominale
        vit1=vit+accel1*dt
        if vit1>vitesse_nominale:
            vit1=vitesse_nominale
    elif (dist-distance)<distance_freinage: #proche de l'arrivée
        vit1=vit+freinage_nominal*dt
        if vit1>vitesse_arrivee: #on ne veut pas qu'il s'arrete mais seulement qu'il ralentisse
            accel1=freinage_nominal
        else:
            vit1=vitesse_arrivee
            accel1=0
    else: #milieu de course
        accel1=0
        vit1=vitesse_nominale
    avance=vit1*dt
    pos1=pos+avance/dist
    trains[numtrain][0:3]=[pos1, vit1, accel1]

##sécurité
#deux trains proches
def test_securite_train(numtrain, pos, gare1, gare2, dist):
    liste_pos = distance_train_proche(gare1, gare2, numtrain)
    arret_demande = False; ralentissement_demande = False
    for pos1 in liste_pos:
        if abs((pos-pos1)*dist)<distance_securite and pos<pos1:
            ralentissement_demande=True
        if abs((pos-pos1)*dist)<distance_securite_mini and pos<pos1:
            arret_demande = True
    return ralentissement_demande, arret_demande

#voie non dispo
def test_securite_gare(numtrain, pos, gare2, gare3, ind, dist):
    arret_demande = False; ralentissement_demande = False
    if not voie_disponible_gare(gare2,gare3, ind):
        if (1-pos)*dist<distance_securite:
            ralentissement_demande = True
        if (1-pos)*dist<distance_securite_mini:
            arret_demande = True
    return ralentissement_demande, arret_demande

#on compile les raisons de s'arreter
def test_securite(numtrain, pos, gare1, gare2, gare3, ind, dist):
    r1, a1 = test_securite_train(numtrain, pos, gare1, gare2, dist)
    r2, a2 = test_securite_gare(numtrain, pos, gare2, gare3, ind, dist)
    arret_du_train((a1 or a2), numtrain)
    ralentissement_du_train((r1 or r2), numtrain, dist)

def arret_du_train(condition, numtrain):
    #arret force correspond à une perturbation qui impose un arret qui n'est pas lié aux conditions de sécurité énoncées ci-dessus
    #cet arret prime sur les autres
    if arret_force_de(numtrain):
        execute_train_attente(numtrain) #le train est déjà arreté
    else:
        if condition:
            if numtrain in arret:
                execute_train_attente(numtrain)
            else:
                execute_arret(numtrain)
        else:
            if numtrain in arret:
                arret.remove(numtrain)

def ralentissement_du_train(condition, numtrain, dist):
    if condition:
        if numtrain not in ralentissement:
            execute_ralentissement(numtrain, dist)
    else:
        if numtrain in ralentissement:
            ralentissement.remove(numtrain)

def arrive_en_gare(pos, vit, dist):
    if pos+vit*dt/dist>1:
        return True
    else:
        return False

def voie_disponible_gare(gare2,gare3, ind):
    if gares[gare2][0][ind][1]<gares[gare2][0][ind][2]:
        return True
    else:
        return False

def execute_arret(numtrain):
    arret.append(numtrain)
    trains[numtrain][1:3]=[0,0]   #arret
    print('train ', numtrain, 'arreté')

def execute_ralentissement(numtrain, dist):
    ralentissement.append(numtrain)
    pos, vit = trains[numtrain][0:2]
    trains[numtrain][2]=freinage_nominal
    if vit>vitesse_arrivee:
        vit1=vit+freinage_nominal*dt
        print('train', numtrain, 'ralenti')
    else:
        vit1=vitesse_arrivee
    pos1=pos+vit1*dt/dist
    trains[numtrain][0:2]=[pos1, vit1]


##mise en service

#permet de commencer à n'importe quelle heure en élimninant les trains passés
def initialisation_horaires(heure):
    global horaires
    num_horaire=0
    temps_deb=horaires[0][1]
    while temps_deb<=heure and len(horaires)>1:
        num_horaire+=1
        temps_deb=horaires[num_horaire][1]
    horaires=horaires[num_horaire:]
    if affichage:
        print(num_horaire, "horaires en dehors de la zone")
    for gare in gares:
        for direction in gare[1]:
            direction[3]=heure

def mise_en_service(heure):
    if len(horaires)>0:
        num_horaire=0
        temps_deb=horaires[num_horaire][1]
        while temps_deb<=heure and len(horaires)>num_horaire+1:
            temps_deb=horaires[num_horaire+1][1]
            num_horaire+=1
        #num_horaire a atteint un train en dehors de la zone des horaires alors num_horaire ne doit pas être mis en place
        if temps_deb>heure:
            j=1
        else: #num_horaire a atteint la fin de la liste
            j=0
        for k in range(num_horaire-j, -1, -1):
            execute_mise_service(k, heure)


def execute_mise_service(num_horaire, heure):
    typ = horaires[num_horaire][0]
    trip_id = horaires[num_horaire][2]
    train = copy.deepcopy(types[typ])
    no_feuille_route = train[7]
    desti = dest[no_feuille_route].copy()
    destinations.append(desti)
    s_a=direct[no_feuille_route].copy()
    sans_arret.append(s_a)
    numtrain = len(trains)
    trains.append(train)
    gare1 = train[5]
    gare2 = desti[0]
    ind = recherche_indice_quai(gare1,gare2)
    gares[gare1][0][ind][1]+=1 #un train de plus en gare
    tps_litteral = conversion_h_m(heure)
    augmenter_voyageurs_gares(gare1, numtrain, heure)
    if service: #affichage
        nom_gare = correspondance_traffic_DB([gare1], liste_id)[0][1]
        print('train', trip_id, 'mis en service à', nom_gare)
    stop_times.append([numtrain, trip_id, [[gare1, heure, 0, 0]]])
    if profil and trip_id==trip_id_course:
        global numtrain_course
        numtrain_course = numtrain
        print('numtrain_course:', numtrain_course, heure)
        gestion_profil(heure)
    horaires.remove(horaires[num_horaire])  #intérêt du compteur descendant de mise_en_service

##fin de service
#il faut un compteur descendant
def execute_fin_service():
    global fin_de_service
    for k in range(len(fin_de_service)-1, -1, -1):
        [numtrain, numgare] = fin_de_service[k]
        execute_suppr_suivi(numtrain, numgare)
        execute_changement_indice_trains(numtrain)
    fin_de_service=[]


def execute_suppr_suivi(numtrain, numgare):
    global trains, destinations, stop_times, numtrain_course, sans_arret
    index = recherche_indice_suivi(numtrain)
    stop_times_stock.append(stop_times[index])
    trip_id=stop_times[index][1]
    stop_times = stop_times[:index]+stop_times[index+1:] #on enlève ce train des trains suivis
    if service:
        print('il restait',int(total_voyageurs_dans_train(numtrain)),'voyageurs', 'fin de service de', trip_id, numtrain)
    if profil and numtrain_course==numtrain:
        global condition_arret_prgm
        condition_arret_prgm = True  #il ne sert à rien de continuer
        numtrain_course= -1
    trains = trains[:numtrain]+trains[numtrain+1:]
    sans_arret = sans_arret[:numtrain]+sans_arret[numtrain+1:]
    destinations = destinations[:numtrain]+destinations[numtrain+1:]
    ind = recherche_indice_quai(numgare, -1)
    gares[numgare][0][ind][1]-=1 #le train libère le quai

#il faut faire correspondre les numtrains suivis/arretés/course avec les nouveaux numtrains
def execute_changement_indice_trains(numtrain):
    n = len(arret)
    for k in range(n):
        if arret[k]>numtrain:
            arret[k]-=1

    n=len(ralentissement)
    for k in range(n):
        if ralentissement[k]>numtrain:
            ralentissement[k]-=1

    n = len(stop_times)
    for i in range(n):
        if stop_times[i][0]>numtrain:
            stop_times[i][0]-=1

    global numtrain_course
    if numtrain_course>numtrain:
        numtrain_course-=1


##animation

def creation_artists(num_max_trains, xlim1, ylim1):
    global plot_trains,annotations_trains,plot_gares,annotations_gares,plot_trains_voie,fig1,train_ax
    fig1 = pl.figure(figsize=(8, 8))
    train_ax = pl.axes([0.2,0.1,0.7,0.8], xlim = xlim1, ylim = ylim1)
    activation_boutons()
    train_ax.set_title('traffic RER')
    plot_trains=[]; plot_gares=[]; annotations_trains=[];  annotations_gares=[]; plot_trains_voie=[]
    for i in range(num_max_trains):
        plot_trains.extend(train_ax.plot([], [], marker='+', color='green',markeredgewidth = 8, markersize = 10))
        plot_trains_voie.extend(train_ax.plot([], [],color='red',linewidth = 5))
        annotations_trains.append(train_ax.annotate(i,xy=(-1,-1), color='green', annotation_clip = False, fontsize = 6))
    for k in range(len(points)):
        plot_gares.extend(train_ax.plot([], [],marker='o',color='black',markersize = 1))
        # if exemple and k%6==0:
        #     texte = str(k) + correspondance_traffic_DB([k], liste_id)[0][1]
        # elif exemple:
        #     texte = ''
        texte = str(k)
        annotations_gares.append(train_ax.annotate(texte,xy=(-1,-1), xycoords='data',color='black', annotation_clip = False, fontsize = 9))


def init():
    n = len(points)
    dessiner_gares(n)
    dessiner_graphe(n)
    annoter_gare(n)
    return plot_gares, annotations_gares   #ajouter l'augmentation de la taille des gares en fct des voyageurs

def anim(i, instance):
    global pause
    heure = t_depart+i*dt/60
    if timer:
        print('temps:', conversion_h_m(heure))
    if pause:
        print("l'éxecution est en pause, appuyer sur une touche")
        pl.waitforbuttonpress(-1)
        pause = not pause
    temps_suivant(heure)
    n = len(points)
    nb_trains = len(trains)
    coef = 0.5
    numeroVoie = 0
    for numtrain in range(nb_trains):
        if est_en_gare(numtrain):
            numgare1 = int(trains[numtrain][5])
            x,y = dessiner_trains_gare(numgare1,numtrain)
        else:
            numgare1, numgare2 = int(trains[numtrain][4]), int(trains[numtrain][5])
            x,y = dessiner_trains_voie(numgare1, numgare2, numtrain, coef, numeroVoie, plot_trains_voie)
            numeroVoie+=1
        dessiner_trains(x,y,numtrain)
    for k in range(numeroVoie, num_max_trains):
        plot_trains_voie[k].set_data([],[])
    for k in range(nb_trains, num_max_trains):
        plot_trains[k].set_data([],[])
        annotations_trains[k].set_position((-1,-1))
    taille_des_gares()
    annoter_gare(n)
    return plot_trains, annotations_trains, plot_trains_voie, plot_gares


##dessin
def dessiner_trains_gare(numgare, numtrain):
    pt = points[numgare]
    return pt[0],pt[1]

def dessiner_trains_voie(numgare1, numgare2, numtrain, coef, numeroVoie, plot_trains_voie):
    pt1, pt2 = points[numgare1], points[numgare2]
    X,Y=[pt1[0],pt2[0]],[pt1[1],pt2[1]]
    pos,vit = trains[numtrain][0:2]
    vect = X[1]-X[0],Y[1]-Y[0]
    x1,y1 = X[0]+pos*vect[0],Y[0]+pos*vect[1]
    pos0 = pos-vit*dt/distances[numgare1,numgare2]*coef
    x0,y0 = X[0]+pos0*vect[0],Y[0]+pos0*vect[1]
    plot_trains_voie[numeroVoie].set_data([x0,x1],[y0,y1])
    return x1,y1

def dessiner_trains(x,y,numtrain):
    if annote:
        annotations_trains[numtrain].set_position((x+dimension,y+dimension))  #eviter la  supersposition
    else:
        annotations_trains[numtrain].set_position((-1,-1))
    if exemple:
        plot_trains[numtrain].set_color('red')
    elif total_voyageurs_dans_train(numtrain)>1700:
        plot_trains[numtrain].set_color('red')
    else:
        plot_trains[numtrain].set_color('blue')
    plot_trains[numtrain].set_data(x,y)

def dessiner_gares(n):
    for numgare in range(n):
        pt = points[numgare]
        plot_gares[numgare].set_data(pt[0],pt[1])
        scalaire = total_voyageurs_en_gare(numgare)/100
        if scalaire>10:
            scalaire = 10
            plot_gares[numgare].set_color('orange')
        if scalaire<5:
            scalaire = 5
        plot_gares[numgare].set_markersize(scalaire)

def annoter_gare(n):
    for numgare in range(n):
        pt = points[numgare]
        if annote:
            annotations_gares[numgare].set_position((pt[0]+dimension, pt[1]-dimension))
        else:
            annotations_gares[numgare].set_position((-1,-1))


def dessiner_graphe(n):
    for lig in range(n):
        for col in range(n):
            if distances[lig,col]!=0:
                pt1 = points[lig]
                pt2 = points[col]
                X,Y=[pt1[0],pt2[0]],[pt1[1],pt2[1]]
                train_ax.plot(X,Y,color='gray',linewidth = 2.5)

def taille_des_gares():
    n = len(points)
    for numgare in range(n):
        if exemple:
            scalaire = 7
        elif affluence:
            scalaire = total_voyageurs_en_gare(numgare)/500
            if scalaire<5:
                scalaire = 5
            elif scalaire>10:
                scalaire = 10
                plot_gares[numgare].set_color('orange')
        else:
            scalaire = 5
        plot_gares[numgare].set_markersize(scalaire)



## interaction (animation)
def changer_etat(nom_var):
    g = globals()
    g[nom_var]= not g[nom_var]

nom_func=["annote", "pause", "timer", "service", "affluence"]

#on ajoute au dictinnaire des variables global les axes et les boutons
#on les associe au changement d'état de la variable du dictionnaire func_boutons
def activation_boutons():
    g = globals()
    for i in range(len(nom_func)):
        ax = pl.axes([0.01,0.1*i,0.1,0.1], xticks=[], yticks=[])
        btn = Button(ax, label = nom_func[i])
        nom_button="btn_{}".format(i)
        g[nom_button]=btn
        globals()[nom_button].on_clicked(lambda event,i = i:changer_etat(nom_func[i]))




##outil voyageurs
#calcul du montant de voyageurs à numgare sur le quai du train numtrain
#permet d'affiner le temps d'arret
def voyageurs_quai(numgare, numtrain):
    total=0
    freq_gare=gares[numgare][1]
    directions_desservies=destinations[numtrain]
    for destination in freq_gare:
        if destination[0] in directions_desservies:
            total+=destination[1]
    return total

#calcul du montant total de voyageurs dans la gare toutes directions confondues, a un instant donné
def total_voyageurs_en_gare(numgare):
    total = 0
    gare = gares[numgare]
    frequentations = gare[1]
    for destination in frequentations:
        total+=destination[1]
    return total

#calcul du montant de voyageurs qui transitent par numgare en une journée
def total_voyageurs_jour(numgare):
    total = 0
    gare = gares[numgare]
    frequentations = gare[1]
    for destination in frequentations:
        total+=destination[2]
    return total

#calcul du montant de voyageurs à bord
def total_voyageurs_dans_train(numtrain):
    total = 0
    remplissage = trains[numtrain][6]
    for destination in remplissage:
        total+=destination[1]
    return total


##voyageurs et remplissage du train

#même si tous les passagers en gare ne peuvent pas monter on reset le tps d'attente sur toutes les directions desservies (il ne sert qu'à comparer à la situation théorique dans laquelle le traffic n'est pas)
#les temps sont ici en minutes
def monter_voyageurs(numtrain, numgare, heure):
    freq=gares[numgare][1]
    remplissage=trains[numtrain][6]
    fraction, passagers_montants = calcul_passagers_montant(numtrain, numgare)
    directions_desservies=destinations[numtrain]
    tps_total=0
    nombre=0
    if fraction>0:
        for direction in freq:
            gare_voulue=direction[0]
            if gare_voulue in directions_desservies: #si le train passe par la gare voulue
                passagers_montants_dest=round(direction[1]*fraction, 3)
                minutes_attendues=(heure-direction[3])*60
                tps_total+=minutes_attendues
                nombre+=1
                execute_monter_passagers(passagers_montants_dest, remplissage, gare_voulue)
            direction[3]=heure #reset temps d'attente
    if nombre==0:
        tps_attente_moyen=0
    else:
        tps_attente_moyen=round(tps_total/nombre, 4)
    index=recherche_indice_suivi(numtrain)
    stop_times[index][2][-1][2:4]=[passagers_montants, tps_attente_moyen]


#calcule la fraction des passagers qui va monter dans numtrain en numgare
def calcul_passagers_montant(numtrain, numgare):
    passagers_voulant_monter=voyageurs_quai(numgare, numtrain)
    voyageurs_a_bord=total_voyageurs_dans_train(numtrain)
    if passagers_voulant_monter==0:
        return 0, 0
    if voyageurs_a_bord+passagers_voulant_monter<contenance:
        passagers_montants=passagers_voulant_monter
    else:
        passagers_montants=contenance-voyageurs_a_bord
    if passagers_montants>10: #permet d'alléger les calculs sans vraiment fausser les résultats
        fraction=passagers_montants/passagers_voulant_monter
    else:
        fraction = 0
    return fraction, round(passagers_montants)

def execute_monter_passagers(passagers_montants_dest, remplissage, gare_voulue):
    rajout_destin=True
    for destination in remplissage:  #il y-a-t-il des passagers pour cette direction
        gare_dir=destination[0]
        if gare_voulue==gare_dir:
            destination[1]+=passagers_montants_dest
            rajout_destin=False
    if rajout_destin:
        remplissage.append([gare_voulue, passagers_montants_dest])

def descendre_voyageurs(numtrain, numgare):
    remplissage=trains[numtrain][6]
    for destination in remplissage:
        if destination[0]==numgare:  #les voyageurs sont arrivés à destination
            remplissage.remove(destination)
            break #ne sert à rien de continuer

## voyageurs et remplissage des gares
'''plages liste les freq relatives des flux de voyageurs découpée par heures'''

#on fait arriver les voyageurs juste avant l'arrivée du train numtrain pour optimiser les calculs et donc que selon les directions desservies
def augmenter_voyageurs_gares(numgare, numtrain, heure):  #deltat est en heures, longueur de plage aussi
    num_p = int(heure)
    directions_desservies=destinations[numtrain]
    freq_gare=gares[numgare][1]
    for destination in freq_gare:
        if destination[0] in directions_desservies:
            voyageurs_par_jour = destination[2]
            deltat = heure- destination[3]
            destination[1]+=plages[num_p]*voyageurs_par_jour*deltat



##outils temps
#int pour enlever les 1.0 -> 1
def conversion_h_m(heure):
    h = int(np.floor(heure))
    mins = int((heure-h)*60)
    return str(h)+':'+str(mins)

def conversion_minutes(liste_h_m):
    liste_h_m = liste_h_m.split(':')
    heure = float(liste_h_m[0])*60+float(liste_h_m[1])
    return heure

def conversion(vit, accel):
    return vit/1000*60, accel/60

##statistiques
'''
temps_cumulé = somme(temps d'attente sur une gare avant de monter dans un train * nb_voyageurs montant dans le train)
temps_cumule_gare
voyageurs_desservis_total
voyageurs_desservis_gare = voyageurs_desservis par gare (liste)
tps_moyen_total = temps_cumulé/voyageurs_desservis_total
tps_moyen_gare = le temps_d'attente moyen par gare (liste)
'''

def statistiques_attente():
    global temps_cumule_gare, voyageurs_desservis_gare, tps_moyen_gare
    temps_cumule, voyageurs_desservis_total = ponctualite_charge()

    #on calcule aussi des stats plus détaillées par gare:
    tps_moyen_gare = zeros.copy()
    for numgare in range(len(gares)):
        if voyageurs_desservis_gare[numgare]!=0:
            tps_moyen_gare[numgare]=temps_cumule_gare[numgare]/voyageurs_desservis_gare[numgare]
        else:
            tps_moyen_gare[numgare]=-1  #non desservie

    if voyageurs_desservis_total!=0:
        tps_moyen_total = temps_cumule/voyageurs_desservis_total
    else:
        tps_moyen_total = 0
    if affichage:
        print('temps cumulé', round(temps_cumule/60, 2), "heures")
        print(round(voyageurs_desservis_total), 'voyageurs desservis')
        print("soit un tps d'attente moyen de", round(tps_moyen_total,2), "minutes")
        print("pour le detail par gare voir temps_cumule_gare, tps_moyen_gare et voyageurs_desservis_gare")

# calcule le temps attendu par les voyageurs*nombre de voyageurs qui ont attendu
#tps_attendu est en minutes
def ponctualite_charge():
    global temps_cumule_gare, voyageurs_desservis_gare
    voyageurs_desservis_total = 0; temps_cumule = 0
    temps_cumule_gare = zeros.copy(); voyageurs_desservis_gare = zeros.copy()
    for train in stop_times_stock:
        for passage in train[2]:
            [numgare, heure, voy,tps_attendu] = passage
            tps = voy*tps_attendu
            temps_cumule+=tps
            voyageurs_desservis_total+=voy
            temps_cumule_gare[numgare]+=tps
            voyageurs_desservis_gare[numgare]+=voy
    return temps_cumule, voyageurs_desservis_total

## écarts aux horaires en traffic normal - critère de ponctualité
'''il faut faire correspondre stop_times0 qui est [trip_id, [horaire1, horaire2, ...] et dont les trip_id sont ordonnées de manière décroissante
avec stop_times qui est [numtrain, trip_id, [groupe_id1, horaire1], [groupe_id2, horaire2], ...]
il arrive que stop_times contiennent moins d'horaires que stop_times0 lorsque les trains ne finissent pas leur parcours'''
'''dans ces modules il faut avoir au préalable converti les id_traffic id_groupe'''

def ponctualite():
    recuperation_groupe_id()
    carre_ecart = 0
    ecart_flat = 0
    for trip in stop_times_stock:
        trip_id = trip[1]
        index = recherche_index_dichotomie(trip_id)
        trip0 = stop_times0[index]
        dep_times0 = trip0[1]
        passages = trip[2]
        n=len(passages)
        N=len(dep_times0)
        for i in range(N):
            h1 = float(dep_times0[i])
            if i<n:
                h2 = float(passages[i][1])
                carre_ecart+=(h2-h1)**2 #écart en heure^2
                ecart_flat+=h2-h1
    # if ecart_flat<0:
    #     print('modèle en avance sur la théorie', ecart_flat)
    # else:
    #     print('modèle en retard sur la théorie', ecart_flat)
    return np.sqrt(carre_ecart)


#renvoie l'index dans stop_times0 de trip_id
def recherche_index_dichotomie(trip_id):
    N = len(stop_times0)
    mini = 0; maxi = N-1
    index = int((mini+maxi)/2)
    while mini!=index and maxi!=index:
        if trip_id>stop_times0[index][0]:
            maxi = index
        else:
            mini = index
        index = int((mini+maxi)/2)
    if trip_id!=stop_times0[index][0]:
        print('erreur')
    else:
        return index

##suivi des trains
#numtrain1 arrive en gare2, s'il est suivi on ajoute l'horaire de passage à trains suivis
def suivi_ajout_passage(numtrain1, gare2, heure):
    index = recherche_indice_suivi(numtrain1)
    stop_times[index][2].append([gare2, heure, 0, 0])

#donne les trains dans la gare numgare
def recherche_train_dans_gare(numgare):
    rep=[]
    for numtrain in range(len(trains)):
        train = trains[numtrain]
        if train[4]==-1 and train[5]==numgare:
            rep.append(numtrain)
    return rep

#retourne l'index dans la liste stop_times du train numtrain
def recherche_indice_suivi(numtrain):
    for index in range(len(stop_times)):
        if stop_times[index][0]==numtrain:
            return index
    print('le train', numtrain, 'circulait sans être suivi en suivi_global')

#on remplace dans la liste prééxistente le numgare par sa correspondance dans la DB en terme de id_groupe
def recuperation_groupe_id():
    for train in stop_times_stock:
        n=len(train[2])
        for index_passage in range(n):
            numgare = train[2][index_passage][0]
            stop_groupe = liste_id[numgare]
            train[2][index_passage][0]=stop_groupe

##suivi d'un profil de course
#on crée profil_suivi qui est plus précis que juste stoptimes des trains: on ne regarde pas les horaires de passage en gare mais la position à toute heure
#on n'utilisera pas ce module pour calculer ponctualite car on n'est pas sur du profil theorique
def gestion_profil(heure):
    if numtrain_course!=-1:
        train_suiv=trains[numtrain_course]
        pos, vit, accel, tps, gare_dep, gare_ar, remplissage = train_suiv[0:7]
        if gare_ar!=-1:
            vit, accel = conversion(vit, accel)
            if len(profil_suivi)>0 and profil_suivi[-1][0]==gare_dep:
                #toujours sur le même intergare
                profil_suivi[-1].append([pos, heure, vit, accel, []])
            else:
                profil_suivi.append([gare_dep, gare_ar, [pos, heure, vit, accel, copy.deepcopy(remplissage)]])


def profil_theorique(trip_id, pos_ax):
    global num_horaire_course
    num_horaire_course = recherche_indice_horaire(trip_id)
    index = recherche_index_dichotomie(trip_id)
    stop_time = stop_times0[index][1]
    #il faut récupérer les destinations pour avoir les distances
    num_typ = horaires0[num_horaire_course][0]
    typ = types[num_typ]
    gare_dep = typ[5]
    feuille_route = [gare_dep] + dest[num_typ][:-1]   #on enlève le -1 (fictif) et on rajoute le terminus de départ
    direc = direct[num_typ]
    profil_th = []
    dist_tot = 0
    indice_temps = 0
    n = len(feuille_route)
    for indice_feuille in range(n-1):
        gare_dep = feuille_route[indice_feuille]
        gare_ar = feuille_route[indice_feuille+1]
        t_dep = stop_time[indice_temps]
        if gare_dep not in direc: #le train s'arrete bien et donc il est pris en compte dans stop_time
            pos_ax.scatter(x=t_dep, y=dist_tot, s=8, color='blue')
            indice_temps+=1
        #dans le cas contraire, on ne modifie pas indice_temps (les directs ne sont pas pris en compte dans stop_times) et on ne plot pas
        dist_tot+= distances[gare_dep][gare_ar]
    t_ar = stop_time[indice_temps]
    pos_ax.scatter(x=t_ar, y=dist_tot, s=8, color='blue')



def dessin_course(vit_ax, pos_ax, remp_ax, accel_ax, color):
    n=len(gares)
    global remp_liste
    t_liste=[]; t_cut_liste=[]; pos_liste=[]; vit_liste=[]; accel_liste=[]; bins=[]
    remp_liste=[[] for k in range(n)]; weight_liste=[[] for k in range(n)]; labels=['']*n
    distance_parcourue = 0
    compt = 0
    # lim_remp = 17; lim_vit = 40; inf = 15 #pour l'étude de trip_id_course='115054849-1_16156'
    lim_remp = np.inf; lim_vit = np.inf; inf = 0 #pour l'étude de trip_id_course='115072256-1_14393'
    t0=profil_suivi[0][2][1]
    t0_cut=t0
    for intergare in profil_suivi:
        gare_dep = intergare[0]
        gare_ar = intergare[1]
        # print(correspondance_traffic_DB([gare_dep], liste_id)[0][1])
        # print(gare_dep)
        if gare_dep!=-1:
            dist_gares = distances[gare_dep][gare_ar]
            if dist_gares==0:
                print('ce train est direct')
        else:
            dist_gares = 0

        for pos, heure, vit, accel, remplissage in intergare[2:]:
            dist = pos*dist_gares
            pos_liste.append(dist+distance_parcourue)
            t_liste.append(heure)
            if compt<inf:
                t0_cut=heure
            else:
                if compt<lim_vit:
                    t_cut_liste.append(heure)
                    vit_liste.append(vit)
                    accel_liste.append(accel)
                    tf_cut=heure
        if gare_dep!=-1:
            pos, heure, vit, accel, remplissage = intergare[2]
            if compt<lim_remp:
                bins.append(heure)
                tot=0
                for numgare, passagers in remplissage:
                    if passagers>40:
                        remp_liste[numgare].append(heure)
                        weight_liste[numgare].append(passagers)
                        remp_ax.text(heure, tot+passagers/2, str(int(passagers)), fontsize=7)
                        name = correspondance_traffic_DB([numgare], liste_id)[0][1]
                        if passagers>200:
                            labels[numgare]=name
                        tot+=passagers
        tf = intergare[-1][1]
        bins.append(tf)
        distance_parcourue+=dist_gares
        pos_ax.plot([tf], [distance_parcourue], marker='o', markersize = 4, color=color)
        compt+=1
    pos_ax.plot(t_liste, pos_liste, color=color)
    vit_ax.plot(t_cut_liste, vit_liste, marker='o', markersize = 4)
    accel_ax.plot(t_cut_liste, accel_liste)
    remp_ax.hist(remp_liste, weights=weight_liste, bins=bins, histtype='barstacked', label=labels)
    remp_ax.legend(loc='upper right', fontsize=8)
    return t0, tf, t0_cut, tf_cut


def reperes(remp_ax, vit_ax, accel_ax, t0, tf, t0_cut, tf_cut):
    v_nom, ac_nom = conversion(vitesse_nominale, accel_nominale)
    v_ar, fr_nom = conversion(vitesse_arrivee, freinage_nominal)
    lims=[t0, tf]
    lims_cut=[t0_cut, tf_cut]
    vit_ax.plot(lims, [v_nom, v_nom])
    vit_ax.plot(lims, [v_ar, v_ar])
    accel_ax.plot(lims, [ac_nom, ac_nom])
    accel_ax.plot(lims, [fr_nom, fr_nom])
    vit_ax.set_xlim(lims)
    accel_ax.set_xlim(lims)
    remp_ax.plot(lims, [contenance, contenance])
    remp_ax.set_xlim(lims)









def comparaison(trip_id):
    fig2, axes = pl.subplots(2, 2, figsize=[23, 16])
    for i in range(2):
        for j in range(2):
            axes[i][j].set_xlabel('temps (heures)')
    [[pos_ax, remp_ax], [vit_ax, accel_ax]] = axes
    pos_ax.set_title('position modèle et théorique')
    pos_ax.set_ylabel('distance depuis terminus (m)')
    remp_ax.set_title('remplissage modèle')
    remp_ax.set_ylabel("nombre de passagers en fonction de la gare d'arrivée")
    accel_ax.set_title('acceléréation modèle m/s^2')
    vit_ax.set_title('vitesse modèle')
    vit_ax.set_ylabel('vitesse (km/h)')
    profil_theorique(trip_id, pos_ax)
    color = 'black'
    t0, tf, t0_cut, tf_cut = dessin_course(vit_ax, pos_ax, remp_ax, accel_ax, color)
    reperes(remp_ax, vit_ax, accel_ax, t0, tf, t0_cut, tf_cut)
    pl.savefig('calage du modèle (profil course de traffic)', dpi=300,  bbox_inches='tight')
    pl.show()

# renvoie l'indice de trip_id dans horaires0
def recherche_indice_horaire(trip_id):
    n=len(horaires0)
    for k in range(n):
        if trip_id==horaires0[k][2]:
            return k
    print('erreur de trip_id, inexistant dans horaires0')


## utilisation dans la db
#on remplit avec notre modèle stop_times_modif
#il y a des petites approximations sur l'heure dans la conversion en h_m c'est pourquoi la db n'est utilisée que pour du debug et pas pour le calcul de l'écart pour la descente de gradient
def reecriture_DB():
    conn = sql.connect(r"F:/informatique/TIPE/database/produit exploitable/GTFS.db")
    c = conn.cursor()
    c.execute('delete from stop_times_modif')
    for train in stop_times_stock:
        trip_id = train[1]
        for passage in train[2]:
            [id_groupe, departure_time, voy, tps]=passage
            departure_time = conversion_h_m(departure_time)
            c.execute('''
            insert into stop_times_modif(route_id, trip_id, departure_time, id_groupe, stop_sequence)
            values(?,?,?,?,?)''', (route_id, trip_id, departure_time, id_groupe, stop_sequence))
    conn.commit()
    conn.close()

def ecart_db(affichage=False):
    if affichage:
        ec_fig, ax=pl.subplots(figsize=[23, 16])
    conn = sql.connect(r"F:/informatique/TIPE/database/produit exploitable/GTFS.db")
    c = conn.cursor()
    c.execute('''
    select s1.departure_time, s2.departure_time, s1.trip_id
    from stop_times as s1
    join stop_times_modif as s2
        on s1.trip_id = s2.trip_id and s1.stop_sequence = s2.stop_sequence
    ''')
    carre_ecart = 0; ecart_flat = 0
    for h1, h2, trip_id in c:
        h1 = conversion_minutes(h1); h2 = conversion_minutes(h2)
        carre_ecart+=(h2-h1)**2 #écart en min^2
        ecart_flat+=h2-h1
        if affichage:
            ax.scatter(h1, 0); pl.scatter(h2, 1)
            ax.annotate(s='', xy=(h1, 0), xytext=(h2, 1),  arrowprops=dict(arrowstyle="->", lw=0.5, mutation_scale=1))
    conn.close()
    print('écart réduit ', np.sqrt(carre_ecart))
    if affichage:
        pl.savefig('régulation', dpi=300, bbox_inches='tight')
        pl.show()




##perturbation
def execute_arret_force(numtrain, tfin):
    arret_force.append([numtrain, tfin])
    if trains[numtrain][4]!=-1: #pas en gare
        execute_arret(numtrain)
    #si en gare, le train ne redémarre tout simplement pas tant qu'il est en arret_force

def gestion_arret_force(heure):
    n=len(arret_force)
    for k in range(n-1, -1, -1):
        if arret_force[k][1]<heure:
            arret_force.remove(arret_force[k])

# teste si numtrain est en arret forcé
def arret_force_de(numtrain):
    for train in arret_force:
        if train[0]==numtrain:
            return True
    return False


##régulation
#on détermine l'horizon de régulation en supposant connaitre la durée de la régulation

#heure est en heures
# 0.2 h = 12 min
def variables_de_pert(duree, tm, g1, g2):
    global duree_pert_est, tmin, gare_pert1, gare_pert2, horizon_retarde
    duree_pert_est=duree
    tmin=tm
    facteur_pert=4
    gare_pert1=g1; gare_pert2=g2
    horizon_retarde = duree_pert_est*facteur_pert

#on veut que le dernier train à partir ne soit pas retardé
def retarder_departs(heure):
    global trains_retardes
    [name1, name2] = correspondance_traffic_DB([gare_pert1, gare_pert2], liste_id)
    conv_tmin = conversion_h_m(tmin)
    conv_tmax =  conversion_h_m(tmin+duree_pert_est)
    print("\n \n perturbation se déroulant depuis", conv_tmin, "jusqu'à", conv_tmax, "entre ", name1[1], gare_pert1, "et", name2[1], gare_pert2, '\n \n', 'numtrain course=', numtrain_course)
    print(destinations[numtrain_course])
    trains_retardes = []
    N=len(trains)
    # print(numtrain_course, trains[numtrain_course], destinations[numtrain_course], gare_pert1, gare_pert2)

#on arrete les trains
    for numtrain in range(N):
        ret = retard(heure)
        destin = destinations[numtrain]
        if test_destination(destin):
            trains_retardes.append(numtrain)
            execute_arret_force(numtrain, heure+ret)
    if consigne_pert:
        dessin_regulation()
        pl.plot([heure, heure+ret], [1, 2], marker='o', color='black')
    print('la perturbation a occasioné le retard des trains ', trains_retardes, " qui sont en service ainsi que le retard des départs d'autres trains passant entre ", name1[1], "et", name2[1])
#on retarde directement le départ des trains considérés
    num_h=0; dep_time=0 #initialisation
    print('horizon retardé : ', conversion_h_m(horizon_retarde), ' min \n')
    while dep_time<horizon_retarde + tmin:
        destin = dest[num_h]
        dep_time=horaires[num_h][1]
        # print(correspondance_traffic_DB(destin, liste_id), test_destination(destin), '\n')
        if test_destination(destin):
            ret = retard(dep_time)
            hor_dep = horaires[num_h][1]
            new_hor = hor_dep + ret
            horaires[num_h][1] = new_hor
            if consigne_pert:
                pl.plot([hor_dep, new_hor], [1, 2], marker='o', color='black')

        num_h+=1
    if consigne_pert:
        pl.savefig('retards en consigne des trains', dpi=300,  bbox_inches='tight')
        pl.show()


def retard(heure):
    return (tmin+horizon_retarde-heure)*duree_pert_est/horizon_retarde

#on regarde si la direction définie par gare_pert1 et gare_pert2 est dans destination
def test_destination(destin):
    n = len(destin)
    for k in range(n-1):
        if gare_pert1==destin[k] and gare_pert2==destin[k+1]:
            return True
    return False

def mise_en_place_pert(heure):
    if heure<tmin and heure+dt/60>=tmin:
        retarder_departs(heure)

def dessin_regulation():
    pl.figure(figsize=[23, 16])
    pl.xlabel('temps (heures)')
    pl.title('retards en consigne des trains')
    pl.plot([tmin, horizon_retarde + tmin], [1, 1])
    pl.plot([tmin, horizon_retarde + tmin], [2, 2])




##paramètres du modèle et calage
'''Tout est mis en m/min
vit commerciale : 50 km.h-1
vit_nominale : 65 kmh
capacité totale 1700
on calcule accel nominale=3912
'''

vitesse_nominale = round(65000/60)  #65km/h en m/min
vitesse_arrivee=round(vitesse_nominale/3) #le train doit arriver en gare à allure raisonnable
contenance = 1700
tps_arret = 0.5 #30 sec
tps_population = 1/10000 #pour 10000 voyageurs en gare, le train attendra une minute de plus (à ajuster en fonction des paramètres de flux)
distance_freinage = 300
distance_demarrage = 150 #longueur d'une gare
distance_securite = 800
distance_securite_mini = 500 #deux trains ne doivent jamais être à moins de 500m
accel_nominale=round((vitesse_nominale)**2/2/distance_demarrage) #on choisit accel_dem de facon à satisfaire dist_dem et vit_dem qui sont des params plausibles
freinage_nominal=round(-accel_nominale/1.5)

var=[vitesse_nominale, vitesse_arrivee, contenance, tps_arret, tps_population, distance_freinage, distance_demarrage, accel_nominale, freinage_nominal]
delt0=300 #m/min
delt1=300 #m/min
delt2=300 #passagers
delt3=0.3 #min
delt4=tps_population/2
delt=[delt0, delt1, delt2, delt3, delt4]

bornes=[[var[k]-delt[k], var[k]+delt[k]] for k in range(5)]
variabs=[var[k] for k in range(5)]
infinitesimaux=[50, 0.05, 0.1, 30, 30]
'''' minimum(retour_ecart, var, bornes, infinitesimaux, 1, 0.01, False)
la derive_partielle de tps_population est nul ...

Lorsque dt est petit (9 sec), l'exécution du programme est plus lente et les trains sont en avance sur la théorie
pour dt grand (20 sec), les trains sont en retard sur la théorie
Les paramètres dépendent du dt choisi...

monteCarlo(retour_ecart, bornes, 10)

'''

'résultat du monte carlo :'
# vitesse_nominale = 1150 #contre 1083 m/min dans le modèle original.
#cela corrrespond à 69 km/h
# vitesse_arrivee = 300 #contre 361 m/min --> faible différence
#surtout il y a une forte dispertsion de vitesse arrivée aux différents minima --> peu d'impact dans la modélisation

#trop de fluctuation pour tps_arret

#on ne peut pas conclure sur contenance et sur tps_population car le modèle de minimisation ne prend en compte qu'un seul train --> les flux à bord du train et à quai ne sont donc pas réalistes


## initialisation
try:
    test = horaires0[0]
except NameError: #si c'est la première fois
    donnees_GTFS("810:B")

