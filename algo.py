import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt


r = 0.2
L = 100


def u_0(x):
    return cos(2*pi*x/L) + 0.1*cos(4*pi*x/L)
    
def f_L(k):
    return r - 1 - 2*((2*pi*k/L)**2) + ((2*pi*k/L)**4)
    
    
N = 1024
dt = 0.05
t_max = 0.51
T = int(t_max/dt)+1

x_range = np.linspace(0, L-(L/N), N)
k_range = np.arange(-N/2,N/2,1)
fL_range = f_L(k_range)
coef_1 = (1+(fL_range*dt/2))/(1-(fL_range*dt/2)) 
coef_2 = dt/(1-(fL_range*dt/2))

    
def SH(f0):
    u_range = f0(x_range)
    U = np.zeros((N,T))
    U[:,0] = u_range
    
    DFT_0 = (1/N)*np.fft.fftshift(np.fft.fft(u_range))
    DFT = np.zeros((N,T))
    DFT[:,0] = DFT_0
    
    uc_range = [u**3 for u in u_range]
    DFTc_0 = (1/N)*np.fft.fftshift(np.fft.fft(uc_range))
    DFTc = np.zeros((N,T))
    DFTc[:,0] = DFTc_0

    for t in range(1,T):
        d1 = DFT[:,t-1]
        dc1 = DFTc[:,t-1]
        if t == 1 :
            dc2 = dc1
        else :
            dc2 = DFTc[:,t-2]
        
        uk = coef_1*d1 + coef_2*((3/2)*dc1 - (1/2)*dc2)
        
        DFT[:,t] = uk
        u = np.fft.ifft(np.fft.ifftshift(N*uk))
        U[:,t] = u
        ukc = (1/N)*np.fft.fftshift(np.fft.fft(u**3))
        DFTc[:,t] = ukc
        
        t += 1
        
    return U,DFT
    

time_ev = SH(u_0)
U = time_ev[0]
D = time_ev[1]

plt.plot(x_range, U[:,0], label = "t = 0")
plt.plot(x_range, U[:,1], label = "t = {}".format(dt))
plt.plot(x_range, U[:,-1], label = "dernier temps")
plt.legend()
plt.show()
plt.clf()

t_range = np.arange(0,t_max, dt)
[xx,tt]=np.meshgrid(x_range,t_range)
plt.contourf(xx,tt, np.real(U).T)
plt.colorbar()
plt.show()


[kk,tt]=np.meshgrid(k_range,t_range)
plt.contourf(kk,tt, np.real(D).T)
plt.colorbar()
plt.show()
