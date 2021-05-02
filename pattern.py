import numpy as np
from numpy import cos, pi, exp
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import scipy.signal as scsg
from scipy.integrate import simps

r = 0.2
L = 100
N = 1024
dt = 0.05
T = 200.01
M = int(T/dt) + 1
h = L/N

x_range = np.linspace(0, L-(L/N), N)
k_range = np.arange(-N/2,N/2,1)

def u_0(x):
    gamma = 0.1
    return 1*(cos(2*pi*x/L) + gamma*cos(4*pi*x/L))
def noise_u_0(x):
    gamma = 0.1
    N = np.shape(x)[0]
    return 1*(cos(2*pi*x/L) + gamma*np.random.random(N)*cos(4*pi*x/L))

def v_0(x):
    return (0.5 - 0.25*cos(2*pi*x/L))

def w_0(x):
    return exp(-(x-L/2)**2)

def f_L(k):
    return r - 1 + 2*((2*pi*k/L)**2) - ((2*pi*k/L)**4)

fL_range = f_L(k_range)
"""plt.plot(k_range,fL_range)
plt.show()
plt.clf()"""

coef_1 = (1+(fL_range*dt/2))/(1-(fL_range*dt/2))
coef_2 = dt/(1-(fL_range*dt/2))
"""plt.plot(k_range, coef_1, label ="coef 1")
plt.plot(k_range, coef_2, label ="coef 2")
plt.legend()
plt.show()
plt.clf()"""

print("Résolution numérique avec une grille spatiale de {} points".format(N))
print("Résolution numérique avec une grille temporelle de {} points".format(M))
print("Paramètres numérique : L = {}, T = {}s, h = {}, k = {}, r = {}".format(L, T, h, dt, r))

def SH(f0):
    pattern = False
    precision = 1e-4
    ground = 0.4
    t_pattern = '/'
    nbr_ok = []
    u_range = f0(x_range)
    U = np.zeros((N,M))
    U[:,0] = u_range

    DFT_0 = (1/N)*np.fft.fftshift(np.fft.fft(u_range))
    DFT = np.zeros((N,M), dtype = complex)
    U[:,0] = u_range
    DFT[:,0] = DFT_0

    DFTc_0 = (1/N)*np.fft.fftshift(np.fft.fft(u_range**3))
    DFTc = np.zeros((N,M), dtype = complex)
    U[:,0] = u_range
    DFTc[:,0] = DFTc_0

    for j in range(1,M):
        d1 = DFT[:,j-1]
        dc1 = DFTc[:,j-1]
        if j == 1 :
            dc2 = dc1
        else :
            dc2 = DFTc[:,j-2]

        uk = (coef_1*d1) - (coef_2*((3/2)*dc1 - (1/2)*dc2))

        DFT[:,j] = uk
        u = np.real(np.fft.ifft(np.fft.ifftshift(N*uk)))
        U[:,j] = u
        ukc = (1/N)*np.fft.fftshift(np.fft.fft(u**3))
        DFTc[:,j] = ukc

        if any(U[:,j])>ground:
            diff = U[:,j]-U[:,j-1]
            bolean_diff = max(diff)<precision
            if bolean_diff:
                nbr_ok.append(1)
            else:
                nbr_ok.append(0)
        else:
            nbr_ok.append(0)
    loop = 50
    #print(len(nbr_ok))
    #print(nbr_ok)
    for j in range(len(nbr_ok)):        # J'essaye d'implémenter un test pour déterminer si on a un pattern ou non, je vois deux options 
                                        # une physiquement + correcte qui consiste à tester qu'on ait une périodicité dans l'espace de notre fonction 
                                        # l'autre moins juste physiquement mais officiellement plus simple à écrire qui est de calculer la différence entre deux itérations
                                        # et si on a quelque chose qui a convergé ET qui n'est pas la valeur de champ uniforme constante du début on a un pattern.
                                        
                                        # On pourrait aussi intégrer la valeur absolue de la fonction au temps final, si ça dépasse une certaine valeur à fixer 
                                        # c'est que l'aire sous la courbe est importante et qu'il y a un pattern
                                        # Je viens de voir ta fonction bifurcation, je vais ajouter ce à quoi je pensais comme ça on pourra en discuter
        if nbr_ok[j] == 1:
            if j+loop < T-1:
                if nbr_ok[j+loop] == 1:     # Je ne comprends pas trop à quoi sert le loop ici, c'est une périodicité dans le temps ?
                    for i in range(loop):
                        if nbr_ok[j+i]==1:
                            pattern = True
                            t_pattern = j*dt
            #else: print("Le test d'apparition des motifs n'est pas concluant. Peut-être que pour un T plus grand des motifs vont apparaitre.")

    return U,DFT, pattern, t_pattern

def bifurcation(U):
    u_range = U[:,0]
    #A_c = (1/L)*simps(U[:,-1]**2,u_range)
    # return A_c
    
    # Test :
    seuil = 0.2
    Départ = True
    for i in range(M):
        A_c = (1/L)*simps(abs(U[:,i]), x_range)
        if A_c > seuil:
            if Départ :
                continue
            else :
                print("Apparition de patterns à t = {} s".format(i*dt))
                break
        else :
            if Départ :
                Départ = False
        if i == M-1:
            print("Pas d'apparition de patterns avec ces paramètres.")
    # Ça a l'air de marcher pas trop mal pour les paramètres actuels, il faut peut-être adapter le seuil






time_ev = SH(u_0)
U = time_ev[0]
D = time_ev[1]
print('Apparition de motifs : {} après {}s'.format(time_ev[2],time_ev[3]))

A = bifurcation(U)
#print('Calcul de A^2 = {}'.format(A))

plt.plot(x_range, U[:,0], label = "t = 0s")
plt.plot(x_range, U[:,1], label = "t = {}s".format(dt))
plt.plot(x_range, U[:,int(M/4)], label = "t = {:2.2f}s".format((M/4)*dt))
plt.plot(x_range, U[:,-1], label = "t = {:2.2f}s".format(T))
plt.xlabel("x")
plt.ylabel("Intensité")
plt.title('Instantanés de l\'équation de Swift-Hohenberg\n r = {}, dt = {}, N = {}, L = {}'.format(r,dt,N,L))
plt.legend()
plt.show()
plt.clf()

t_range = np.arange(0,T, dt)
[xx,tt]=np.meshgrid(x_range,t_range)
plt.contourf(xx,tt, U.T, cmap = "plasma", levels = 100) # cmap = "jet" dans les consignes
plt.xlabel("x")
plt.ylabel("t")
plt.title("Simulation numérique de l'équation de Swift-Hohenberg \n r = {}, dt = {}, N = {}, L = {}".format(r,dt,N,L))
plt.colorbar()
plt.show()

"""
[kk,tt]=np.meshgrid(k_range,t_range)
plt.contourf(kk,tt, np.real(D).T)
plt.colorbar()
plt.show()
"""
