import sqlite3 as sql
import numpy as np
import time
import matplotlib.pyplot as pl
from algorithmes_de_minimisation import minimiser, minimum

#nous utiliserons deux connections pour pouvoir effectuer des actions à l'intérieur d'une boucle générée par la première connection
#(écrire avec c2 à l'intérieur de c provoque des erreurs)
conn = sql.connect(r"F:/informatique/TIPE/database/produit exploitable/GTFS.db")
c = conn.cursor()
c2 = conn.cursor()

##outils
def traitement_tuple(L):
    rep=[]
    for tupl in L:
        rep.append(tupl[0])
    return rep

def somme(L, ind):
    s=0
    for l in L:
        s+=l[ind]
    return s

def reverse(liste):
    N = len(liste)
    reverse_liste=[]
    for num in range(N-1,-1,-1):
        reverse_liste.append(liste[num])
    return reverse_liste

def recuperer_id_groupe(x, y):
    c.execute('''
    select id_groupe
    from stops_groupe
    where (stops_groupe.x between {a}- 500 and {a}+ 500) and (stops_groupe.y between {b}- 500 and {b}+ 500)
    '''.format(a=x, b=y))
    return c.fetchall()

def recuperer_liaisons(x, y):
    c.execute('''
    select stops_groupe.stop_name, stops_groupe.id_groupe
    from stops_groupe
    join graphe
        on graphe.to_id_groupe=stops_groupe.id_groupe
    where graphe.from_id_groupe in(
        select id_groupe
        from stops_groupe
            where (stops_groupe.x between {a}- 300 and {a}+ 300) and (stops_groupe.y between {b}- 300 and {b}+ 300))
    '''.format(a=x, b=y))
    return c.fetchall()

def recuperer_x_y(id_groupe):
    c2.execute('select x,y from stops_groupe where id_groupe={}'.format(id_groupe))
    return c2.fetchone()

def calcul_distance(dernier_id_groupe, id_groupe):
    XY=[]
    for id_g in [dernier_id_groupe, id_groupe]:
        c.execute('select x, y from stops_groupe where id_groupe={}'.format(id_g))
        XY.append(c.fetchone())
    [(x1,y1), (x2,y2)]=XY
    d=dist(x1, y1, x2, y2)
    return d

def dist(x1, y1, x2, y2):
    return np.sqrt((x1-x2)*(x1-x2)+(y1-y2)*(y1-y2))

def random_color():
    return [np.random.rand()*0.8 for k in range(3)]

##paramètres du système lambert93
def conversion_rad(angle_degre):
    return angle_degre*np.pi/180

def latitude_iso(lat_rad):  #reparamétrage symétrique de la latitude
    sin=np.sin(lat_rad)
    p1=np.arctanh(sin)
    p2=np.arctanh(e*sin)
    result=p1-e*p2
    return result

xs=700000; ys=12655612.05 #coordonnées du pole nord dans le système lambert93
phi1=conversion_rad(44); ss1=np.sin(phi1); cs1=np.cos(phi1) # Premier parallèle automécoïque ie lattitude de sécance
phi2=conversion_rad(49); ss2=np.sin(phi2); cs2=np.cos(phi2) # Deuxième parallèle automécoïque
phi0=(phi1+phi2)/2
a=6378388 #grand axe de l'ellipse en m de la Terre ie rayon à l'équateur
e=0.08248325676 #excentricité de l'ellipse de la Terre
gN1=a/np.sqrt(1-(e*ss1)**2); gl1=latitude_iso(phi1)  #grande normales aux sécantes (meilleur rayon de courbure)
gN2=a/np.sqrt(1-(e*ss2)**2); gl2=latitude_iso(phi2)
lon0=conversion_rad(3)  #méridien de greenwitch par rapport à celui de Paris en degrés
expo=np.log(gN2/gN1*cs2/cs1) / (gl1-gl2) #exposant de la projection
C=((gN1 * cs1) / expo) * np.exp(expo * gl1)

##conversion WGS84_to_lambert93
def WGS84_to_lambert93(lat, lon):
    lat_rad, lon_rad = conversion_rad(lat), conversion_rad(lon)
    g1=latitude_iso(lat_rad)
    R = C*np.exp(-expo*g1)
    teta = expo*(lon_rad-lon0)
    X = xs + R*np.sin(teta)
    Y = ys - R*np.cos(teta)
    return X, Y

##fonction commune aux flux
#il faut avoir au préalable rempli le cursor
def representation_flux(nb_title, seuil, xn=600, xm=715, yn=6800, ym=6905):
    fig_mob=pl.figure(figsize=[23, 16])
    mob_ax=pl.axes(title='flux de personnes émis par la ville par jour pour motif professionnel(Lambert 93 en km)')
    mob_ax.axis('equal')
    X, Y, S = [], [], []
    compt=0; compt2=0
    for x, y, flux, name in c:
        x_k, y_k = x/1000, y/1000
        if x_k>xn and x_k<xm and y_k>yn and y_k<ym:
            S.append(flux/1000)
            X.append(x/1000)
            Y.append(y/1000)
            compt2+=1
            if flux/1000>seuil:
                compt+=1
                if compt%4==0:
                    mob_ax.text(x=x_k, y=y_k, s=name)
            else:
                if compt2%nb_title==0:
                    mob_ax.text(x=x_k, y=y_k, s=name)
    scat = mob_ax.scatter(X, Y, s=S)
    grid(mob_ax)
    handles, labels = scat.legend_elements(prop="sizes")
    mob_ax.legend(handles, labels, loc='lower right', title="nombre d'habitants \n(en milliers)")



##mobilites pro -flux emis par VILLE
def conversion_mobilites():
    c.execute('select cle, lat, lon from mobilites_pro')
    compt=0
    for cle, lat, lon in c:
        x, y = WGS84_to_lambert93(lat, lon)
        c2.execute('update mobilites_pro set x={x}, y={y} where cle={c}'.format(x=x, y=y, c=cle))
        compt+=1
        if compt%100==0:
            print(compt)
    conn.commit()

def representation_mobilites():
    c.execute('select x, y, flux_emis_jour, name from mobilites_pro')
    representation_flux(45, 20)
    pl.savefig('flux émis par ville (mobilités pro)', dpi=300, bbox_inches='tight')
    pl.show()

##flux emis par POLE-ROUTE_ID (frequentation_m)

'''on connait les flux émis par chaque ville, on répartit ce flux entre les différents poles (frequentation_m)
puis on suppose que le flux dans les transports en commun est proportionnel a ce flux tous transports confondus
on va donc faire en sorte que ce facteur de prop donne un flux émis par jour dans chaque gare le plus proche de la fréquentation de référence donnée par RATP'''

def remplir_nb_trips():
    c.execute('''
    select trips.route_id, count(trips.trip_id)
    from trips
    join calendar
        on calendar.service_id=trips.service_id
        where calendar.start_date<20200522 and calendar.stop_date>20200522
    group by trips.route_id''')
    for route_id, nb_trips in c:
        c2.execute('update routes set nb_trips={n} where route_id="{r}"'.format(n=nb_trips, r=route_id))
    conn.commit()

modes=[0.8, 4.2, 2.7, 3.9] #en millions de déplacements par jour selon les routes_types
# 0	tramway; 1	métro; 2	RER; 3	bus
#on va donc comparer les contenances relatives
remplissage_contenance=[0]*4
def contenances():
    c.execute('select sum(nb_trips) from routes group by route_type')
    repart = c.fetchall()
    for k in range(4):
        remplissage_contenance[k] = modes[k]*10**6/repart[k][0]
#les contenances trouvés sont nettement inférieures à la réalité mais elles donnent un poids relatif
#certaines courses ont propablement été comptées en multiple

def remplir_flux_emis():
    c.execute('update flux_emis set frequentation_m=0')
    contenances()
    c.execute('''
    select mobilites_pro.code_commune, mobilites_pro.flux_emis_jour
    from stops_groupe
    join mobilites_pro
        on mobilites_pro.code_commune=stops_groupe.code_commune
    group by mobilites_pro.code_commune
    ''')
    #on ne sélectionne que les codes communes qui ont des pôles
    compt=0
    for code_commune, flux_ville in c:
        execute_repartition_flux(code_commune, flux_ville)
        compt+=1
        if compt%50==0:
            print(compt)
            conn.commit()
    conn.commit()

def execute_repartition_flux(code_commune, flux_ville, affichage=False, ville='à remplir'):
    if affichage:
        contenances()
    c2.execute('''
    select s.id_groupe, graphe.route_id, graphe.route_type, nb_trips, routes.route_short_name
        from graphe
        join (select id_groupe
                from stops_groupe
                where code_commune={}) as s
            on s.id_groupe=graphe.from_id_groupe
        join routes
            on routes.route_id=graphe.route_id
            where graphe.route_type>=0 and graphe.route_type<=3
        order by graphe.route_type, graphe.route_id'''.format(code_commune))
    #on récupère, toutes les liaisons dont le departure_id est dans la ville de code_commune donné
    weight=[]
    dernier_route_id='init'
    #on répartit le flux_ville entre les différentes lignes de la ville
    #le poids de chaque ligne est nb_trips*contenance
    #fact est donc le nombre de voyageurs qui transitent par la ligne route_id
    for id_groupe, route_id, route_type, nb_trips, name in c2:
        if route_id==dernier_route_id:
            weight[-1][-1]+=(id_groupe,)
        else:
            dernier_route_id=route_id
            fact=remplissage_contenance[int(route_type)]*nb_trips
            weight.append([fact, route_type, name, route_id, (id_groupe,)])
        print(id_groupe, route_id, nb_trips, name, fact)
    tot_modes=somme(weight, 0)
    if affichage:
        print(weight)
        repr_distrib_ville(weight, flux_ville, ville, tot_modes)
    #puis pour chaque ligne, on répartit ce flux entre les poles (de manière équitable puisqu'on suppose que l'attractivité se joue à l'echelle d'une ville)
    for ligne in weight:
        fact, route_type, name, route_id, poles = ligne
        freq_ligne = fact/tot_modes*flux_ville
        freq_pole = freq_ligne/len(poles)
        # print('\n'+name)
        for pol in poles:
            update_freq_m(pol, freq_pole, route_id)

def update_freq_m(pol, freq_pole, route_id):
    c2.execute('select frequentation_m from flux_emis where id_groupe={i} and route_id="{rid}"'.format(i=pol, rid=route_id))
    freq=c2.fetchone()
    if freq==None:
        c2.execute('insert into flux_emis (id_groupe, route_id, frequentation_m, frequentation_c) values (?,?,?,?)', (pol, route_id, freq_pole, 0))
    else:
        freq=freq[0]
        c2.execute('''
        update flux_emis
        set frequentation_m={f}
            where id_groupe={p} and route_id="{rid}"
        '''.format(p=pol, f=freq_pole+freq, rid=route_id))


def repr_distrib_ville(weight, flux_ville, ville, tot_modes):
    sizes=[[] for k in range(4)]
    labels=[[] for k in range(4)]
    labels_globaux=['tramway', 'metro', 'RER', 'bus']
    fig_rep = pl.figure(figsize=[23, 16])
    X = [(1, 3, 1), (2, 3, 2), (2, 3, 3), (2, 3, 5), (2, 3, 6)]
    axes=[]
    for nrows, ncols, plot_number in X:
        sb=fig_rep.add_subplot(nrows, ncols, plot_number)
        axes.append(sb)
    rep_ax=axes[0]
    rep_ax.set_title('répartition des flux émis ('+str(flux_ville)+' voyageurs) en fonctions des différents modes à '+ville)
    for ligne in weight:
        fact, route_type, name, route_id, poles = ligne
        freq_ligne = fact/tot_modes*flux_ville
        print(freq_ligne)
        if route_type<4 and freq_ligne>1:
            sizes[route_type].append(freq_ligne)
            labels[route_type].append(name)
    t=[sum(sizes[k]) for k in range(4)]
    print(sizes, sum(t))
    sizes_globaux=[sum(sizes[k]) for k in range(4)]
    for k in range(3, -1, -1):
        axes[k+1].pie(sizes[k], labels=labels[k], autopct = lambda x: str(int(x)) + '%')
        if len(sizes[k])>0:
            axes[k+1].set_title('répartition par lignes de '+labels_globaux[k])
        if sizes_globaux[k]==0:
            sizes_globaux=sizes_globaux[:k]+sizes_globaux[k+1:]
            labels_globaux=labels_globaux[:k]+labels_globaux[k+1:]
    rep_ax.pie(sizes_globaux, labels=labels_globaux, autopct= lambda x : int(x*flux_ville/100))
    pl.savefig('répartition modale test', dpi=300, bbox_inches='tight')
    pl.show()
#execute_repartition_flux(75105, 16842.0, affichage=True, ville='Paris 05')

def representation_flux_emis():
    c.execute('''
    select x, y, sum(frequentation_m), ""
    from flux_emis
    join stops_groupe
        on stops_groupe.id_groupe=flux_emis.id_groupe
    group by flux_emis.id_groupe''')
    representation_flux(500, 10)
    pl.savefig('flux émis par pôle', dpi=300, bbox_inches='tight')
    pl.show()




##normalisation du flux emis
'''deux approches:
1) on veut minimiser les écarts au carré de flux émis en choisissant le facteur multiplicatuif
2) on veut que la totalité des flux émis par les gares de la RATP corresponde aux données
'''
#valeur de référence donnée par RATP (frequentation_c)

def calcul_ecart(facteur_population):
    c.execute('''
    select sum(frequentation_m), sum(frequentation_c)
    from flux_emis
    group by id_groupe''')
    ecart=0; ecart_flat=0
    for frequentation_m, frequentation_c in c:
        if frequentation_c!=0:
            # ecart_flat+=abs(frequentation_m*facteur_population-frequentation_c)
            ecart+=np.sqrt((frequentation_m*facteur_population-frequentation_c)**2)
    return np.sqrt(ecart)

def calcul_ecart_par_ligne(facteur_population):
    c.execute('''
    select frequentation_m, frequentation_c
    from flux_emis
    where frequentation_c!=0''')
    ecart=0; ecart_flat=0
    for frequentation_m, frequentation_c in c:
        if frequentation_c!=0:
            # ecart_flat+=abs(frequentation_m*facteur_population-frequentation_c)
            ecart+=np.sqrt((frequentation_m*facteur_population-frequentation_c)**2)
    return np.sqrt(ecart_flat)

def calcul_facteur_normalisation():
    c.execute('''
    select sum(frequentation_m), sum(frequentation_c)
    from flux_emis
    group by id_groupe''')
    sum_c=0; sum_m=0
    for frequentation_m, frequentation_c in c:
        if frequentation_c!=0:
            sum_c+=frequentation_c
            sum_m+=frequentation_m
    K = sum_c/sum_m
    return K

# calcul_facteur_normalisation()
# >> 4.169361260527865

#fonction vectorielle pour le gradient descent
def calcul_ecart_vecto(facteur_population):
    f=facteur_population[0]
    return calcul_ecart(f)

def repr_ecarts():
    X=np.arange(0, 5, 0.05)
    Y=[]
    for x in X:
        Y.append(calcul_ecart_par_ligne(x))
    fig, ax = pl.subplots(figsize=[24, 12])
    title='valeur absolue des écarts entre les flux émis selon les données RATP et selon le modèle 1'
    ax.set_title(title)
    ax.plot(X, Y)
    pl.xlabel('facteur_population')
    pl.savefig(title, dpi=300, bbox_inches='tight')
    pl.show()

'''
minimiser(0, 5, 0.0001, calcul_ecart)=1.9097520275672006 par méthode du nombre d'or
minimum(calcul_ecart_vecto, [1], [[0, 5]], [0.001], 0.001, 0.01, False) par descente de gradient
'''

facteur_population=1.9
# facteur_population=2.1 #par ligne carré
# facteur_population=3.34 #carré
# facteur_population=1.9 #flat


def scalairisation_flux_emis():
    c.execute('select id_groupe, route_id, frequentation_m from flux_emis')
    for id_groupe, route_id, frequentation_m in c:
        c2.execute('update flux_emis set frequentation_m={}'.format(facteur_population*frequentation_m))
    conn.commit()

def repr_ecarts_ligne(par_ligne=False, xn=630000, xm=670000, yn=6845000, ym=6875000):
    fig_emis, axes = pl.subplots(1,2, figsize=[24, 12])
    ax_m, ax_c = axes
    ax_m.axis('equal'); ax_c.axis('equal'); grid(ax_m); grid(ax_c)
    ax_c.set_title("flux emis par jour d'après les données RATP \n soit 289 pôles (Lambert 93 en km)")
    ax_m.set_title("flux modélisés sur ces mêmes pôles")
    #1=ax_m (modèle) ; 2=ax_c (théorie)
    if par_ligne:
        c.execute('''
        select stops_groupe.x, stops_groupe.y,  frequentation_m, frequentation_c, stop_name
        from stops_groupe
        join flux_emis
            on flux_emis.id_groupe=stops_groupe.id_groupe
        where (stops_groupe.x between {xn} and {xm} ) and (stops_groupe.y between {yn} and {ym} )
        '''.format(xn=xn, xm=xm, yn=yn, ym=ym))
    else:
        c.execute('''
        select stops_groupe.x, stops_groupe.y,  sum(frequentation_m), sum(frequentation_c), stop_name
        from stops_groupe
        join flux_emis
            on flux_emis.id_groupe=stops_groupe.id_groupe
        where (stops_groupe.x between {xn} and {xm} ) and (stops_groupe.y between {yn} and {ym} )
        group by stops_groupe.id_groupe
        '''.format(xn=xn, xm=xm, yn=yn, ym=ym))
    X, Y, S1, S2 = [], [], [], []
    compt=0
    for x, y, f1, f2, stop_name in c:
        if f2!=0:
            compt+=1
            xs, ys = x/1000, y/1000
            X.append(xs)
            Y.append(ys)
            s1, s2 = f1*facteur_population/1000, f2/1000
            # print(s1, s2)
            S1.append(s1)
            S2.append(s2)
            if s1==0:
                ax_m.scatter(xs, ys, s=4, color='grey')
            if s1>500 or s2>300:
                ax_m.text(xs, ys, s="")
                ax_c.text(xs, ys, s="")
    scat1 = ax_m.scatter(X, Y, s=S1)
    scat2 = ax_c.scatter(X, Y, s=S2)
    handles1, labels1 = scat1.legend_elements(prop="sizes", num=np.arange(10, 60, 10))
    handles2, labels2 = scat2.legend_elements(prop="sizes", num=np.arange(10, 60, 10))
    ax_m.legend(handles1, labels1, loc='lower right', title='flux emis par jour (milliers de déplacements)')
    ax_c.legend(handles2, labels2, loc='lower right', title='flux emis par jour (milliers de déplacements)')
    pl.savefig('flux emis sur la ligne B '.format(facteur_population), dpi=300, bbox_inches='tight')
    pl.show()


## suppression des exceptions
#on prend les trains rajoutés n'assurant pas un trajet régulier et on les supprime
def exceptions():
    c.execute('''
    delete from stop_times
    where stop_times.trip_id in(
    select trips.trip_id
    from trips
    join calendar_dates
        on calendar_dates.service_id=trips.service_id
    where exception_type=1 and calendar_dates.service_id not in(
            select service_id
                from calendar))''')
    c.execute('''
    delete from trips
    where trip_id in(
    select trips.trip_id
    from trips
    join calendar_dates
        on calendar_dates.service_id=trips.service_id
    where exception_type=1 and calendar_dates.service_id not in(
            select service_id
                from calendar))''')
    conn.commit()

##creation de groupe
#dans la DB, plusieurs stop_id designent la même gare (un stop_id par ligne de la gare + un stop_id par stop_area (=entrée de la gare) )
def enlever_proche(liste):
    compt = 0
    non_traite = liste
    stopid_groupeid=[]
    while len(non_traite)>1:
        stop_id1, lat1, lon1 = non_traite[0]; stop_id2, lat2, lon2 = non_traite[1]
        non_traite.remove(non_traite[0])
        stopid_groupeid.append([compt, stop_id1])
        indice = 0
        while  coordonnee_proche(lat1, lat2, eps_lat) and indice<len(non_traite)-1:  #la liste est classée par latitude
            if coordonnee_proche(lon1, lon2, eps_lon):  #je ne compare longitude que pour deux latitude proches
                non_traite.remove((stop_id2, lat2, lon2))
                stopid_groupeid.append([compt, stop_id2])  #meme id_groupe (=compt) si proches
            else:
                indice+=1
            stop_id2, lat2, lon2 = non_traite[indice]
        compt+=1
        if compt%1000==0:
            print(compt)
    stopid_groupeid.append([compt, non_traite[0][0]])
    return stopid_groupeid, compt


def coordonnee_proche(lat_long1, lat_long2, eps):
    if abs(lat_long1-lat_long2)<eps:
        return True
    return False

lat_ref, lon_ref = 48.8, 2.3
rayon_reel=100 #distance en m
def choix_eps(distance):
    x_ref, y_ref = WGS84_to_lambert93(lat_ref, lon_ref)
    x, y = x_ref, y_ref
    lat, lon = lat_ref, lon_ref
    while dist(x, y, x_ref, y_ref)<distance:
        lat+=0.00001
        x, y = WGS84_to_lambert93(lat, lon_ref)
    eps_lat=lat-lat_ref

    x, y = x_ref, y_ref
    lat, lon = lat_ref, lon_ref
    while dist(x, y, x_ref, y_ref)<distance:
        lon+=0.00001
        x, y = WGS84_to_lambert93(lat_ref, lon)
    eps_lon=lon-lon_ref
    return eps_lat, eps_lon

#on trouve
eps_lat, eps_lon = choix_eps(rayon_reel)

def regroupement():
    c.execute('delete from groupe')
    conn.commit()
    c.execute('select stop_id, stop_lat, stop_lon from stops order by stop_lat')
    liste = c.fetchall()
    stopid_groupeid, compt = enlever_proche(liste)
    print('il y a', compt, 'id_groupe')
    compt = 0
    for id_groupe, stop_id in stopid_groupeid:
        c.execute('insert into groupe(id_groupe, stop_id) values (?,?)', (id_groupe, stop_id))
        if compt%3000==0:
            print(compt)
        compt+=1
    conn.commit()
    print('done\n')
    remplissage_stops_groupe()


## validité du regroupement
#on regarde le nombre de stop_id regroupés et à quelle gare ça correspond
def verification_nombre_par_regroupement(condition):
    c.execute('select count(stop_id) as compt, id_groupe from groupe group by id_groupe having compt>3 order by compt desc limit 5')
    liste=c.fetchall()
    print(liste)
    time.sleep(3)
    if condition:
        for count, id_groupe in liste:
            c.execute('''
            select stop_name
            from stops
            join groupe
                on stops.stop_id = groupe.stop_id
            where groupe.id_groupe={}'''.format(id_groupe))
            print(id_groupe, traitement_tuple(c.fetchall()), '\n')

# on regarde s'il y a des transfers (=correspondance) de moins d'une minute entre deux stop_id non regroupés
def verification_regroupements_transfers():
    c.execute('''select g1.id_groupe, t1.stop_name, g2.id_groupe, t2.stop_name, transfers.transfer_time
    from transfers
    join groupe as g1
        on g1.stop_id=transfers.from_stop_id
    join groupe as g2
        on g2.stop_id=transfers.to_stop_id

    join stops_groupe as t1
        on t1.id_groupe=g1.id_groupe
    join stops_groupe as t2
        on t2.id_groupe=g2.id_groupe

    where g1.id_groupe!=g2.id_groupe and transfers.transfer_time<=60
    group by g1.id_groupe, g2.id_groupe
    order by transfer_time
    limit 50''')
    return c.fetchall()



def validation_regroupement(xmin=660000, xmax=662000, ymin=6870000, ymax=6872000):
    fig_GTFS, axes = pl.subplots(1,2, figsize=[24, 12])
    ax2, ax3 = axes
    ax2.axis('equal'); ax3.axis('equal')
    c.execute('select count(stop_id) from stops')
    N1 = c.fetchone()[0]
    c.execute('select count(id_groupe) from stops_groupe')
    N2 = c.fetchone()[0]
    c.execute('select stop_lat, stop_lon from stops')
    n1=0
    for lat, lon in c:
        x, y = WGS84_to_lambert93(lat, lon)
        if x<xmax and x>xmin and y<ymax and y>ymin:
            ax2.plot(x/1000, y/1000, marker='o')
            n1+=1
    c.execute('''
    select x, y
    from stops_groupe
    where x<{xmax} and x>{xmin} and y<{ymax} and y>{ymin}'''.format(xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin))
    n2=0
    for x,y in c:
        n2+=1
        ax2.plot(x/1000, y/1000, marker='+')
        ax3.plot(x/1000, y/1000, marker='o', markersize=8)
        centre=x/1000, y/1000
        creation_cercle(centre, (rayon_reel+10)/1000, ax2)
    grid(ax3)
    ax2.set_title('position de ' + str(n1) + ' pôles avant regroupement dans une zone de 2 km \n (lambert93 en km)\n total: ' + str(N1))
    ax3.set_title('position de ' + str(n2) + ' pôles après regroupement dans cette même zone \n total: ' + str(N2))
    pl.savefig('efficacité du regroupement spatial', dpi=300, bbox_inches='tight')
    pl.show()

def creation_cercle(centre, rayon, ax):
    x, y = centre
    X=[x+rayon*np.cos(teta) for teta in np.linspace(0, 2*np.pi, 30)]
    Y=[y+rayon*np.sin(teta) for teta in np.linspace(0, 2*np.pi, 30)]
    ax.plot(X, Y)

##Stops_groupe
#on utilise les coordonnes de lambert (projection conique) pour pouvoir représenter le réseau
def remplissage_stops_groupe():
    c.execute('delete from stops_groupe')
    c.execute('''
    select stops.stop_lat, stops.stop_lon, groupe.id_groupe, stop_name, sum(frequentation_c)
        from stops
        join groupe
            on stops.stop_id = groupe.stop_id
        group by groupe.id_groupe''')
    for lat, lon, id_groupe, stop_name, freq in c:
        x, y = WGS84_to_lambert93(lat, lon)
        c2.execute('insert into stops_groupe(id_groupe, stop_name, x, y, frequentation_c) values (?,?,?,?,?)', (id_groupe, stop_name, x, y, freq))
    conn.commit()
    print('done \n')

#on se donne un carré n*n que l'on fait grandir jusqu'a trouver une ville correspondant à la gare
def remplissage_code_commune():
    c.execute('select id_groupe, x, y, n_maillage from stops_groupe')
    compt=0
    for id_groupe, x, y, n_maillage in c:
        data=trouver_ville(n_maillage)
        code_commune=plus_proche(data, x, y)
        c2.execute('update stops_groupe set code_commune={c} where id_groupe={i}'.format(c=code_commune, i=id_groupe))
        compt+=1
        if compt%1000==0:
            print(compt)
    conn.commit()


def trouver_ville(n_maillage):
    carre=carre_n_maillage(n_maillage-3*m-3, 7)
    c2.execute('select x, y, code_commune, n_maillage from mobilites_pro where n_maillage in {}'.format(carre))
    data=c2.fetchall()
    if len(data)==0:
        print('non trouvé') #on agrandit suffisement la séléction
        carre=carre_n_maillage(n_maillage-4*m-4, 9)
        c2.execute('select x, y, code_commune, n_maillage from mobilites_pro where n_maillage in {}'.format(carre))
        data=c2.fetchall()
    return data

def plus_proche(data, x, y):
    d=np.inf
    for x1, y1, code_commune, n_maillage in data:
        d1=dist(x, y, x1, y1)
        if d1<d:
            d=d1
            code=code_commune
    return code

def repr_attribution_ville(xmin=660000, xmax=675000, ymin=6870000, ymax=6885000):
    fig_comm=pl.figure(figsize=[22, 15])
    comm_ax=pl.axes(title='attribution des pôles aux villes adjacentes dans un rayon de 15 km (Lambert 93 en km)')
    comm_ax.axis('equal')
    c.execute('''
    select stops_groupe.x, stops_groupe.y, mobilites_pro.x, mobilites_pro.y, mobilites_pro.name
    from stops_groupe
    join mobilites_pro
        on mobilites_pro.code_commune=stops_groupe.code_commune
    where stops_groupe.x<{xmax} and stops_groupe.x>{xmin} and stops_groupe.y<{ymax} and stops_groupe.y>{ymin}
    '''.format(xmax=xmax, xmin=xmin, ymax=ymax, ymin=ymin))
    for x1, y1, x2, y2, ville_name in c:
        x1k, y1k, x2k, y2k = x1/1000, y1/1000, x2/1000, y2/1000
        comm_ax.plot(x1k, y1k, marker='o')
        comm_ax.plot(x2k, y2k, marker='s', markersize=10, color='red')
        comm_ax.text(x=x2k, y=y2k, s=str(ville_name))
        comm_ax.annotate(s='', xy=(x2k, y2k), xytext=(x1k,y1k),  arrowprops=dict(arrowstyle="->", lw=0.5, mutation_scale=1))
    grid(comm_ax)
    pl.savefig('attribution_ville', dpi=300, bbox_inches='tight')
    pl.show()

##maillage
#pour limiter le coût des calculs de gares proches, on associe à chaque id_groupe un numero correspondant à un cadrillage n*m
#chaque case est de coté 1000m
# n:lignes, m:colonnes --> numerotation par ligne en partant de bas à gauche

def limites():
    global minx, miny, maxx, maxy, n, m
    c.execute('select min(x), min(y), max(x), max(y) from mobilites_pro')
    [(minx1, miny1, maxx1, maxy1)] = c.fetchall()
    c.execute('select min(x), min(y), max(x), max(y) from stops_groupe')
    [(minx2, miny2, maxx2, maxy2)] = c.fetchall()
    minx=min(minx1, minx2); miny=min(miny1, miny2); maxx=max(maxx1, maxx2); maxy=max(maxy1, maxy2)
    n=(maxy-miny)//1000+2; m=(maxx-minx)//1000+2
    n, m = int(n), int(m)
    c.execute('delete from global_data')
    c.execute('insert into global_data (n_ligne, m_colonne, minx, miny, maxx, maxy) values (?,?,?,?,?,?)',(n,m, minx, miny, maxx, maxy ))
    conn.commit()


def maillage():
    limites()
    compt=0
    print("maillage jusqu'à", n*m)
    for lig in range(n):
        for col in range(m): # case de 0 --> n*m-1
            x=(minx//1000)*1000+col*1000
            y=(miny//1000)*1000+lig*1000
            c.execute('''update stops_groupe
            set n_maillage={num}
            where x between {x0} and {x0}+1000
            and y between {y0} and {y0}+1000
            '''.format(x0=x, y0=y, num=compt))
            c.execute('''update mobilites_pro
            set n_maillage={num}
            where x between {x0} and {x0}+1000
            and y between {y0} and {y0}+1000
            '''.format(x0=x, y0=y, num=compt))
            compt+=1
            if compt%1000==0:
                print(compt)
    conn.commit()


##maj de l'importance
#on prend le carré 3*3 pour etre plus exact
#le centre est donc n_maillage+m+1
def importance(liste, n_maillage):
    c.execute('insert into importance(n_maillage, n_gares) values(?,?)', (n_maillage+m+1, len(liste)))

#on veut aussi remplir les bordures
def completer_importance():
    for col in range(m):
        c.execute('insert into importance(n_maillage, n_gares) values(?,?)', (col, 0))
        c.execute('insert into importance(n_maillage, n_gares) values(?,?)', (col+(n-1)*m, 0))
    for lig in range(1, n-1):
        c.execute('insert into importance(n_maillage, n_gares) values(?,?)', (lig*m, 0))
        c.execute('insert into importance(n_maillage, n_gares) values(?,?)', (lig*m-1, 0))

##graphe du reseau
"il est plus rapide de tout récuperer depuis stop_times et de sélectionner progressivement les infos intéressantes. La requête SQL trop complexe n'aboutissait pas"

#on retrouve le graphe du réseau à partir de stop_times (=horaires de chaque train)
#on traite par route_id pour supprimer les trip directs
#calcul des distances seulement si il y a une liaison (=voie) ou si les gares sont proches (à pied)
def selection_graphe():
    print('recuperation de stop_times')
    c.execute('''
    select routes.route_type, selection_trip.route_id, groupe.id_groupe, stop_times.stop_sequence
    from stop_times
    join (
            select trips.trip_id, trips.route_id, trips.service_id
            from (
                    select service_id, count(service_id) as nombre
                    from trips
                    group by service_id) as selection_ser
            join trips
                on trips.service_id=selection_ser.service_id
            where selection_ser.nombre>5) as selection_trip
        on selection_trip.trip_id = stop_times.trip_id
    join groupe
        on groupe.stop_id=stop_times.stop_id
    join calendar
        on calendar.service_id=selection_trip.service_id
    join routes
        on routes.route_id=selection_trip.route_id
    where selection_trip.route_id!="800:TER" and calendar.start_date<20200522 and calendar.stop_date>20200522
    order by selection_trip.route_id, selection_trip.trip_id, stop_times.stop_sequence
    ''')
    print('liste recupere')


# regroupe est ordonne selon cette forme
# liste_dest=[[route_id, dest], [route_id2, dest2]]
# dest=[[id_groupe1, id_groupe2, id_groupe3], [id_groupe7, id_groupe8]]
#c est params car cette fonction est réutilisée dans données_traffic
def creation_liste_dest(c):
    route_id_en_cours="init"
    route_type_en_cours="init"
    dernier_stop_seq="init"
    compt = 0
    liste_dest=[]
    dest=[]
    for route_type, route_id, id_groupe, stop_sequence in c:
        if route_id==route_id_en_cours:
            if stop_sequence!=dernier_stop_seq+1:  #nouveau trip
                dest.append([id_groupe])
                dernier_stop_seq=stop_sequence
            else: #je continue un trip_id
                dest[-1].append(id_groupe)
                dernier_stop_seq+=1
        else: #fin de la ligne (route_id)
            if compt!=0:
                liste_dest.append([route_type_en_cours, route_id_en_cours, dest])
            dest=[[id_groupe]] #on réinitialise et on ajoute le depart
            route_id_en_cours=route_id
            route_type_en_cours=route_type
            dernier_stop_seq=stop_sequence
        compt+=1
        if compt%100000==0:
            print(compt)
    liste_dest.append([route_type_en_cours, route_id_en_cours, dest]) #terminer
    print('liste_dest crée')
    return liste_dest


def remplissage_graphe():
    c.execute('delete from graphe')
    conn.commit()
    selection_graphe()  #je recupere les stop_times
    liste_dest=creation_liste_dest(c)
    liste_dest=prep_suppr_double(liste_dest)
    liste_dest=traitement_directs(liste_dest)
    print('directs traités', len(liste_dest), 'lignes')
    for route_type, route_id, dest in liste_dest:
        print(route_id)
        for destination in dest:
            insertion_une_destination(destination, route_type, route_id)
    conn.commit()
    suppression_doublons_graphe()
    print('graphe recopie \n')

def insertion_une_destination(destination, route_type, route_id):
    compt=0
    dernier_id_groupe='init'
    for id_groupe in destination:
        if compt!=0:
            dist=calcul_distance(dernier_id_groupe, id_groupe)
            c.execute('insert into graphe(from_id_groupe, to_id_groupe, route_type, route_id, distance) values (?,?,?,?,?)', (dernier_id_groupe, id_groupe, route_type, route_id, dist))
        dernier_id_groupe = id_groupe
        compt+=1

#on veut que le graphe soit symetrique, si les gares sont reliées elles le sont dans les deux sens
#redondance des informations dans la DB mais requetes SQL plus faciles
def retablir_symetrie():
    print('symetrisation')
    c.execute('''
    insert into graphe (from_id_groupe, to_id_groupe, route_type, route_id, distance)
        select graphe.to_id_groupe, graphe.from_id_groupe, route_type, route_id, distance from graphe
        join (select from_id_groupe, to_id_groupe from graphe
                except
            select to_id_groupe, from_id_groupe from graphe) as selection
            on selection.from_id_groupe=graphe.from_id_groupe and selection.to_id_groupe=graphe.to_id_groupe
            ''')
    conn.commit()

def prep_suppr_double(liste_dest):
    liste_dest_corrige=[]
    for route_type, route_id, dest in liste_dest:
        liste_dest_corrige.append([route_type, route_id, suppr_double(dest)])
    return liste_dest_corrige

def suppr_double(dest):
    new_dest=[]
    for destination in dest:
        if destination not in new_dest:
            new_dest.append(destination)
    return new_dest

##traitement directs
'''je ne veut pas dans le graphe de liaisons virtuelles créées par un train direct
je regarde pour chaque liaison si il n'y en a pas une plus longue'''
#j'applique correction_directs_ligne à toutes les lignes
#parfois un direct est emboité dans un plus direct encore et il faut plusieurs itérations de correction_directs_ligne
def traitement_directs(liste_dest):
    liste_dest_corrige=[]
    for route_type, route_id, dest in liste_dest:
        nombre_chgmts=1
        if route_type!=3: #les bus n'ont pas ce problème
            while nombre_chgmts!=0:
                dest, nombre_chgmts = correction_directs_ligne(dest, False)
        liste_dest_corrige.append([route_type, route_id, dest])
    return liste_dest_corrige

#remplace dans dest (pour un route_id donc) les trajets directs par les omnibus
#ce programme est utilisé dans donnees traffic pour recuperer directs, dans ce cas cond_creation_directs=True et on peut renseigner directs
def correction_directs_ligne(dest, cond_creation_directs, directs=[]):
    nombre_chgmts=0
    N=len(dest)
    for numdest in range(N):
        feuille_route=dest[numdest]
        for k in range(len(feuille_route)-2, -1, -1):
            gare1=feuille_route[k]; gare2=feuille_route[k+1]
            liste_recherche=dest[:numdest]+dest[numdest+1:]
            intermediaires=recherche_trajet_equivalent(gare1, gare2, liste_recherche)
            if verification_boucle(intermediaires, feuille_route):
                feuille_route=feuille_route[:k+1]+intermediaires+feuille_route[k+1:]  #compteur descendant
                if cond_creation_directs:
                    directs[numdest].extend(intermediaires)
                nombre_chgmts+=len(intermediaires)
        dest[numdest] = feuille_route
    if cond_creation_directs:
        return dest, nombre_chgmts, directs
    else:
        return dest, nombre_chgmts

#il ne faut pas créer de boucle --> infini sinon
#cette situation arrive sur les lignes ou il y a des boucles (RER C par ex)
def verification_boucle(intermediaires, destination):
    for id_groupe in intermediaires:
        if id_groupe in destination:
            return False
    return True

#je recherche s'il y a un train non-direct entre gare1, gare2
def recherche_trajet_equivalent(gare1, gare2, liste_dest):
    parties_changer=[]
    for k in range(len(liste_dest)):
        destination = liste_dest[k]
        resultats=[-1, -1]
        for index in range(len(destination)):
            if destination[index]==gare1:
                resultats[0]=index
            elif destination[index]==gare2:
                resultats[1]=index
        if resultats[0]!=-1 and resultats[1]!=-1 and abs(resultats[0]-resultats[1])>1: #il existe un trajet moins direct
            # print(gare1, gare2, k)
            parties_changer = remetre_ordre(destination, resultats, parties_changer)
    if len(parties_changer)==0:
        return []
    else:
        plus_long = selection_plus_long(gare1, gare2, parties_changer)
    return plus_long

#on veut le plus long (=le moins direct) et pas de boucle
def selection_plus_long(gare1, gare2, parties_changer):
    plus_long=[]
    for liste in parties_changer:
        if len(liste)>len(plus_long) and gare1 not in liste and gare2 not in liste:
            plus_long = liste
    return plus_long

#le trajet intermediaire trouvé peut etre de gare2-->gare1 et pas dans le bon sens
def remetre_ordre(destination, resultats, parties_changer):
    if resultats[0]<resultats[1]: #bon ordre
        parties_changer.append(destination[min(resultats)+1:max(resultats)])
    else:
        parties_changer.append(reverse(destination[min(resultats)+1:max(resultats)]))  #ordre inverse
    return parties_changer

#après remplacement des directs par omnibus on a des redondances
def suppression_doublons_graphe():
    print('suppression doublons')
    c.execute('''
    delete from graphe
    where cle not in (
    select  cle
        from graphe
        where from_id_groupe!=to_id_groupe
        group by from_id_groupe, to_id_groupe, route_type, route_id )''')
    conn.commit()
    print('sans doublons graphe \n')

##gares accessibles à pied
'''route_id=-1 si à moins de 500m
route_id=-2 si à moins de 1km
le maillage permet de reduire le cout n*n en séquencant l'espace
on en profite pour calculer importance'''
def marche():
    print('ligne', n, 'colonne', m, 'total', n*m)
    for lig in range(n-2):
        for col in range(m-2):
            n_maillage=lig*m + col
            carre=carre_n_maillage(n_maillage, 3)
            c.execute('select id_groupe, x, y from stops_groupe where n_maillage in {}'.format(carre))
            liste=c.fetchall()
            traitement_graphe_marche(liste)
            importance(liste, n_maillage)
            if n_maillage%1000==0:
                print(n_maillage)
    suppression_doublons_graphe()  #plus efficace de le faire une fois à la fin
    completer_importance()
    conn.commit()
    print('marche rempli\n')

#on renseigne en bas à gauche et renvoie le carrée de k*k
def carre_n_maillage(n_maillage, k):
    rep=()
    for ligne in range(k):
        for col in range(k):
            rep+=(n_maillage+ligne*m+col,)
    return rep

#il faudrait ne remplir qu'un demi et recopier pour gagner du temps
def traitement_graphe_marche(liste):
    for id_groupe1, x1, y1 in liste:
        for id_groupe2, x2, y2 in liste:
            execute_marche(id_groupe1, x1, y1, id_groupe2, x2, y2)

def execute_marche(id_groupe1, x1, y1, id_groupe2, x2, y2):
    deltx=abs(x1-x2); delty=abs(x1-x2)
    distance=np.sqrt(deltx**2+delty**2)
    if distance<rayon_reel*3:
        if distance<rayon_reel+10:
            route_type=-1
        else:
            route_type=-2
        c.execute('insert into graphe (from_id_groupe, to_id_groupe, route_type, route_id, distance) values (?,?,?,?,?)', (id_groupe1, id_groupe2, route_type, "marche", distance))


##représentation graphe
def grid(ax):
    xmin, xmax, ymin, ymax = ax.axis()
    x_ticks=np.arange(int(xmin), xmax+1, 1); y_ticks=np.arange(int(ymin), ymax+1, 1)
    ax.set_xticks(x_ticks, minor=True); ax.set_yticks(y_ticks, minor=True)
    ax.grid(which='both', alpha=0.2)

titles=['tramway', 'métro', 'RER', 'bus']
def representation(route_type, condition_global):  #on choisis de tracer ligne par ligne ou tout d'un coup
    fig_maill=pl.figure(figsize=[23, 16])
    maill_ax=pl.axes()
    maill_ax.axis('equal')
    c.execute('select count(route_id) from routes where route_type={}'.format(route_type))
    N=c.fetchone()[0]
    title = str(N) + ' lignes de ' + titles[route_type] + ' positionnés en km dans le système de coordonnées Lambert'
    maill_ax.set_title(title)
    c.execute('''
    select from_id_groupe, to_id_groupe, graphe.route_type, graphe.route_id, routes.route_long_name, s1.x, s1.y, s2.x, s2.y
    from graphe
    join routes
        on routes.route_id=graphe.route_id
    join stops_groupe as s1
        on s1.id_groupe=from_id_groupe
    join stops_groupe as s2
        on s2.id_groupe=to_id_groupe
    where graphe.route_type={} and from_id_groupe<to_id_groupe
    order by graphe.route_id, s1.x, s1.y'''.format(route_type))
    dernier_route_id='init'
    compt=0
    for id1, id2, route_type, route_id, route_long_name, x1, y1, x2, y2 in c:
        if route_id!=dernier_route_id:
            compt+=1
            if not condition_global:
                pl.show()
            if route_type!=3:
                color=random_color()
                maill_ax.plot([x1/1000, x2/1000], [y1/1000, y2/1000], color=color, linewidth=3, label=route_long_name)
                print(dernier_route_id)
            else:
                color=[0, 0, 1]
                if compt%20==0:
                    print(compt)
        else:
            maill_ax.plot([x1/1000, x2/1000], [y1/1000, y2/1000], color=color, linewidth=3)
        dernier_route_id = route_id
    if route_type!=3:
        maill_ax.legend(loc='lower right', title='nom des lignes')
    grid(maill_ax)
    print(dernier_route_id)
    pl.savefig('maillage ' + titles[route_type], dpi=400,  bbox_inches='tight')
    pl.show()

def representation_une_ligne(route_id, ax):
    c.execute('''select s1.x, s1.y, s2.x, s2.y, s1.stop_name, s1.id_groupe
    from graphe
    join stops_groupe as s1
        on s1.id_groupe=graphe.from_id_groupe
    join stops_groupe as s2
        on s2.id_groupe=graphe.to_id_groupe
    where graphe.route_id="{}" and s1.id_groupe>s2.id_groupe
    group by s1.stop_name, s2.stop_name
    '''.format(route_id))
    ax.axis('equal')
    for x1, y1, x2, y2, name, id_groupe in c:
        ax.plot([x1/1000, x2/1000], [y1/1000, y2/1000], color='black')
        # ax.annotate(name+' '+str(id_groupe), xy=(x1, y1), size=6)

## outils de debug
def liaisons(route_id):
    c.execute('''
    select s1.stop_name, s1.id_groupe, s2.stop_name, s2.id_groupe
    from graphe
    join stops_groupe as s1
        on s1.id_groupe=graphe.from_id_groupe
    join stops_groupe as s2
        on s2.id_groupe=graphe.to_id_groupe
    where graphe.route_id="{}"
    group by s1.stop_name, s2.stop_name
    order by s1.stop_name'''.format(route_id))
    return c

def trip_id_passant_par(id_groupe):
    c.execute('''
    select trips.route_id, trips.trip_id, trips.service_id
    from stop_times
    join trips
        on trips.trip_id=stop_times.trip_id
    where stop_times.stop_id
    in (select stops.stop_id
            from stops
            join groupe
                on groupe.stop_id=stops.stop_id
            where groupe.id_groupe="{}")
    group by trips.route_id'''.format(id_groupe))
    return c


def representation_maillage():
    c.execute('select stop_lat, stop_lon from stops')
    for lat, lon in c:
        x,y = WGS84_to_lambert93(lat, lon)
        pl.plot(x, y, marker='o')
    pl.show()
    c.execute('select x,y from stops_groupe')
    for x,y in c:
        pl.plot(x, y, marker='o')
    pl.show()

##global
def reseau():
    print('remplissage de nb_trips\n')
    remplir_nb_trips()
    print('conversion Lambert93 mobilites\n')
    conversion_mobilites()
    print('suppression des exceptions\n')
    exceptions()
    print('regroupement\n')
    regroupement()
    maillage()
    print('remplissage_graphe\n')
    remplissage_graphe()
    print('symétrisation du graphe\n')
    retablir_symetrie()
    print('attribution des villes\n')
    remplissage_code_commune()
    print('frequentations modèlisés\n')
    remplir_flux_emis()
    print('liaisons marche\n')
    marche()
    for route_type in range(3, -1, -1):
        representation(route_type, True)
    #il faut aussi actualiser les flux du RERB pour le modèle
