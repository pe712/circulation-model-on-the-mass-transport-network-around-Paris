import numpy as np



##méthode du nombre d'or
phi=(np.sqrt(5)+1)/2
def minimiser(xmin, xmax, seuil_x, f):
    print(xmin, xmax)
    x2=xmin+(xmax-xmin)/(phi+1)
    if abs(x2-xmin)<seuil_x or abs(x2-xmax)<seuil_x:
        return x2
    else:
        y2=f(x2)
        x3 = x2+(xmax-x2)/(phi+1)
        y3=f(x3)
        if y3>y2: #on réduit l'intervalle
            return  minimiser(xmin, x2, seuil_x, f)
        else:
            return  minimiser(x3, xmax, seuil_x, f)

##algorithme de gradient descent
'''variables contient les N paramètres de f
infinitésimaux contient des variations de chacun des paramètres, suffisantes pour approximer la dérivée
le test de convergence se fait sur la norme du gradient
'''


# approximation de la dérivée partielle par rapport à la variable d'index num_deriv à partir de la tangente
def derive_partielle(f, variables, infinitesimaux, num_deriv):
    X1=variables.copy(); X2=variables.copy()
    X1[num_deriv]-=infinitesimaux[num_deriv]; X2[num_deriv]+=infinitesimaux[num_deriv]
    f1=f(X1); f2=f(X2)
    return (f2-f1)/(2*infinitesimaux[num_deriv])

def gradient(f, variables, infinitesimaux, N):
    grad=[]
    for num_deriv in range(N):
        grad.append(derive_partielle(f, variables, infinitesimaux, num_deriv))
    return grad

def norme(vect):
    somme=0
    for direc in vect:
        somme+=direc**2
    return np.sqrt(somme)

def diff_vect(U, V):
    W=[]
    for i in range(len(U)):
        W.append(U[i]-V[i])
    return W

def prod_vect(alpha, U):
    W=[]
    for u in U:
        W.append(alpha*u)
    return(W)

#dans le changement du pas, si les variables changent peu sans que pour autant le grad soit petit, c'est que l'une des variable est limitée par les bornes du modèle
def minimum(f, variables0, bornes, infinitesimaux, pas_cv_grad, pas, dessin):
    variables=variables0.copy()
    N=len(variables)
    iteration=1; grad=gradient(f, variables, infinitesimaux, N)
    if dessin:
        X, Y= [], []
    while iteration<15 and norme(grad)>pas_cv_grad:
        if dessin:
            X.append(variables[0])
            Y.append(variables[1])
        print('itération ', iteration, variables, grad, pas, '\n')
        avancement=prod_vect(pas, grad)
        variables2=diff_vect(variables, avancement)
        for num_dir in range(N):
            borne_dir=bornes[num_dir]
            variation_var=0
            if variables2[num_dir]<borne_dir[1] and variables2[num_dir]>borne_dir[0]: #on ne change la valeur du paramètre que s'il reste dans les limites du modèle
                variation_var+=abs(variables[num_dir]-variables2[num_dir])
                variables[num_dir]=variables2[num_dir]
        #amélioration du taux s'apprentissage:
        # if variation_var<pas:
        #     pas=pas/10
        iteration+=1
        grad=gradient(f, variables, infinitesimaux, N)
    if dessin:
        return X, Y
    else:
        return variables

##exemple
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as pl

def fonction_test(variables):
    x, y = variables
    return 10*(np.sin(x)**2 + 0.8*np.sin(y)**2)**2

def fction_non_vect(x, y):
    return 10*(np.sin(x)**2 + 0.8*np.sin(y)**2)**2

def grid():
    g=np.vectorize(fction_non_vect)
    X=np.arange(-1.5, 0.9, 0.1)
    Y=np.arange(-1.5, 0.7, 0.1)
    X, Y = np.meshgrid(X, Y)
    Z = g(X, Y)
    return X, Y, Z

def plot_3D():
    X, Y, Z = grid()
    fig=pl.figure()
    ax=Axes3D(fig)
    ax.plot_surface(X, Y, Z)
    pas = 0.01
    X, Y = minimum(fonction_test, [-1, -1.5], [[-10, 10], [-10, 10]], [0.0001, 0.0001], 0.001, pas, True)
    n = len(X)
    decal = -1
    for k in range(n):
        x, y = X[k], Y[k]
        z = fction_non_vect(x, y)
        ax.quiver(x, y, z+decal, 0, 0, -decal, color = 'black', linewidth=1, arrow_length_ratio=0)
        ax.scatter(xs=x, ys=y, zs=z+decal, color='red')
    pl.show()

def plot_contour():
    X, Y, Z = grid()
    pl.grid()
    pl.contour(X, Y, Z, range(30), colors =['blue']*30, alpha=0.7)
    pas = 0.01
    X, Y = minimum(fonction_test, [-1, -1.5], [[-10, 10], [-10, 10]], [0.0001, 0.0001], 0.001, pas, True)
    n = len(X)
    for k in range(n):
        x, y = X[k], Y[k]
        pl.scatter(x, y, color='red', s=20)
        if k<n-1:
            pl.annotate('', xy=(x, y), xytext=(X[k+1], Y[k+1]),  arrowprops=dict(arrowstyle="->", lw=3, mutation_scale=1))
    pl.savefig('contour 2D gradient', dpi=300,  bbox_inches='tight')
    pl.show()

# minimum(fonction_test, [1, 1], [[-2, 2], [-2, 2]], [0.01, 0.01], 0.001, 0.3, True)


##monte Carlo
#n est le nombre de valeur prise par variable d'entrée de f
# si f est une fonction de 3 variables, alors il y aura n^^3 valeurs prises
#on représente chaque point testé par une liste (écriture en base n)
def monteCarlo(f, bornes, n):
    N = len(bornes)
    mini = np.inf; meilleur_var='init'
    pas = [(bornes[k][1] - bornes[k][0])/n for k in range(N)]
    pos = [0]*N #point de départ
    for k in range(n**N):
        if k%200==0:
            print(pos)
        var = [bornes[j][0]+pos[j]*pas[j] for j in range(N)]
        resultat = f(var)
        if resultat<mini:
            print('minimum de ', resultat, ' obtenu en ', var, 'correspondant à', pos)
            mini = resultat
            meilleur_var = var
        dizaine = True
        j = 0
        while dizaine and j<N:
            if pos[j]==n-1:
                pos[j] = 0
                j+=1
            else:
                pos[j]+=1
                dizaine = False
    return mini, meilleur_var