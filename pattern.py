import numpy as np
from numpy import cos, pi, exp, sqrt
import matplotlib.pyplot as plt
from matplotlib.colors import Colormap
import scipy.signal as scsg
from scipy.integrate import simps


#########################################Conditions initiales#############################################

def u_0(x,L):
    gamma = 0
    return 1*(cos(2*pi*x/L) + gamma*cos(4*pi*x/L))

def noise_u_0(x,L):
    gamma = 0.1
    N = np.shape(x)[0]
    return 1*(cos(2*pi*x/L) + gamma*np.random.random(N)*cos(4*pi*x/L))

def noise(x,L):
    gamma = 1e-4
    N = np.shape(x)[0]
    return gamma*np.random.random(N)

def cste(x,L):
    X = np.full(np.shape(x)[0], 1)
    return X

def step(x,L):
    X1 = np.full(int(np.shape(x)[0]/2), 1)
    X2 = np.full(int(np.shape(x)[0]/2), 0)
    X = np.concatenate((X1,X2))
    return X

def v_0(x,L):
    return (0.5 - 0.25*cos(2*pi*x/L))

def w_0(x,L):
    return exp(-(x-L/2)**2)



#########################################Ex 1 (a), Simulation de référence (Fig3)#############################################
def SH(f0,r,L):
    N = 1024
    dt = 0.05
    T = 200.01
    M = int(T/dt) + 1
    h = L/N

    x_range = np.linspace(0, L-(L/N), N)
    k_range = np.arange(-N/2,N/2,1)

    print("Résolution numérique avec une grille spatiale de {} points".format(N))
    print("Résolution numérique avec une grille temporelle de {} points".format(M))
    print("Paramètres numérique : L = {}, T = {}s, h = {}, k = {}, r = {}".format(L, T, h, dt, r))

    fL_range = r - 1 + 2*((2*pi*k_range/L)**2) - ((2*pi*k_range/L)**4)
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

    u_range = f0(x_range,L)
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
    cbar = plt.colorbar()
    cbar.set_label('Intensité', rotation=270)
    plt.show()


    [kk,tt]=np.meshgrid(k_range,t_range)
    plt.contourf(kk,tt, np.abs(DFT.T), cmap = "plasma", levels = 100) # cmap = "jet" dans les consignes
    plt.xlabel("k")
    plt.ylabel("t")
    plt.title("Simulation numérique de l'équation de Swift-Hohenberg \n r = {}, dt = {}, N = {}, L = {}".format(r,dt,N,L))
    cbar = plt.colorbar()
    cbar.set_label('Intensité', rotation=270)
    plt.show()


    return U, x_range, [L, T, h, dt, r, T, M, N]

SH(u_0, 0.2, 100)

##################################### Ex1 (b), mesure du temps d'apparition des motifs en fonction de r et L #########################

def tl_mesure(f0,r,L):
    N = 1024
    dt = 0.05
    T = 300.01
    M = int(T/dt) + 1
    h = L/N

    x_range = np.linspace(0, L-(L/N), N)
    k_range = np.arange(-N/2,N/2,1)

    fL_range = r - 1 + 2*((2*pi*k_range/L)**2) - ((2*pi*k_range/L)**4)
    coef_1 = (1+(fL_range*dt/2))/(1-(fL_range*dt/2))
    coef_2 = dt/(1-(fL_range*dt/2))

    u_range = f0(x_range,L)
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


    seuil = 0.2
    Départ = True
    t_pattern = 0
    lamb_f = 0

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

        # Temps d'apparition
        if not j%int(1/dt):
            continue

        A_c = (1/L)*simps(abs(u), x_range)
        if A_c > seuil:
            if Départ :
                continue
            else :
                t_pattern = j*dt
                # Longueur d'onde
                ua = np.abs(uk)
                k_f = abs(ua.tolist().index(max(ua))) + N/36
                lamb_f = 2*pi/k_f

                break
        else :
            if Départ :
                Départ = False

    return t_pattern, lamb_f


def rL_effect(u0, r_range, L_range):
    time = np.zeros((len(r_range), len(L_range)))
    wavelength = np.zeros((len(r_range), len(L_range)))

    for n,r in enumerate(r_range):
        for m,L in enumerate(L_range):
            t_,l_ = tl_mesure(u0, r, L)
            time[n,m] = t_
            wavelength[n,m] = l_

    plt.imshow(time)
    plt.xlabel("L = {}".format(L_range))
    plt.ylabel("r = {}".format(r_range))
    cbar.set_label('Temps d\'apparition t*', rotation=270)
    plt.title('Temps d\'apparition des motifs en fonction de $r$ et $L$.')
    plt.show()
    plt.clf()

    plt.imshow(wavelength)
    plt.xlabel("L = {}".format(L_range))
    plt.ylabel("r = {}".format(r_range))
    cbar = plt.colorbar()
    cbar.set_label('Intensité', rotation=270)
    plt.title('Longueur d\'onde des motifs en fonction de $r$ et $L$.')
    plt.show()
    plt.clf()


def r_effect(u0, r_range):
    L = 100
    time = np.zeros((len(r_range)))
    wavelength = np.zeros((len(r_range)))

    for n,r in enumerate(r_range):
        t_,l_ = tl_mesure(u0, r, L)
        time[n] = t_
        wavelength[n] = l_

    plt.plot(r_range, time)
    plt.xlabel("$r$")
    plt.ylabel("Temps")
    plt.title("Temps d'apparition des motifs en fonction de $r$")
    plt.show()
    plt.clf()

    plt.plot(r_range, wavelength)
    plt.xlabel("$r$")
    plt.ylabel("Longueur d'onde")
    plt.title("Longueur d'onde des motifs en fonction de $r$")
    plt.show()
    plt.clf()

def L_effect(u0, L_range):
    r = 0.2
    time = np.zeros((len(L_range)))
    wavelength = np.zeros((len(L_range)))

    for n,l in enumerate(L_range):
        t_,l_ = tl_mesure(u0, r, l)
        time[n] = t_
        wavelength[n] = l_

    plt.plot(L_range, time)
    plt.xlabel("$L$")
    plt.ylabel("Temps")
    plt.title("Temps d'apparition des motifs en fonction de $L$")
    plt.show()
    plt.clf()

    plt.plot(L_range, wavelength)
    plt.xlabel("$L$")
    plt.ylabel("Longueur d'onde")
    plt.ylim([0,1.1*max(wavelength)])
    plt.title("Longueur d'onde des motifs en fonction de $L$")
    plt.show()
    plt.clf()



rL_effect(u_0, np.arange(-0.05,0.25,0.05), [25,50,100,150,200])
#r_effect(u_0, np.arange(-0.05, 0.25, 0.01))
#L_effect(u_0, np.arange(30,200, 30))


########################################## Ex2, mesure de A^2 et diagrame de bifurcation##############################################

def A_mesure(f0,r,L):
    N = 1024
    dt = 0.03
    T = 800.01
    M = int(T/dt) + 1
    h = L/N

    x_range = np.linspace(0, L-(L/N), N)
    k_range = np.arange(-N/2,N/2,1)

    fL_range = r - 1 + 2*((2*pi*k_range/L)**2) - ((2*pi*k_range/L)**4)
    coef_1 = (1+(fL_range*dt/2))/(1-(fL_range*dt/2))
    coef_2 = dt/(1-(fL_range*dt/2))

    u_range = f0(x_range,L)
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


    seuil = 0.2
    Départ = True
    t_pattern = 0

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

    A = sqrt((1/L)*simps(U[:,-1]**2, x_range))
    return A


def r_bifurcation(u0, r_range):
    integ = np.zeros((len(r_range)))
    L = 100

    for n,r in enumerate(r_range):
        A_ = A_mesure(u0, r, L)
        integ[n] = A_

    plt.plot(r_range, integ)
    plt.xlabel("r")
    plt.ylabel("A")
    plt.show()
    plt.clf()


    integ_pos = [i for i in integ if i > 0.0001]
    r_pos = r_range[-len(integ_pos):]

    lr_pos = np.log(r_pos)
    lint_pos = np.log(integ_pos)
    plt.plot(lr_pos, lint_pos)
    plt.xlabel("ln(r)")
    plt.ylabel("ln(A)")
    plt.show()
    plt.clf()

    slope = (lint_pos[1:] - lint_pos[:-1])/(lr_pos[1:] - lr_pos[:-1])
    plt.plot(lr_pos[:-1], slope)
    plt.xlabel("ln(r)")
    plt.ylabel("d/dr(ln(A))")
    plt.show()
    plt.clf()

    return integ

#r_bifurcation(u_0, np.arange(-0.01,0.1,0.005))
