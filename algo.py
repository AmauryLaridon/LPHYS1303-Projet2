import numpy as np
from numpy import cos, pi
import matplotlib.pyplot as plt


r = 0.1
L = 100


def u_0(x):
    return cos(2*pi*x/100) + 0.1*cos(4*pi*x/100)
    
def f_L(k):
    return r - 1 - 2*(2*pi*k/L)**2 + (2*pi*k/L)**4
    
    
N = 1024
dt = 0.05
t_max = 1
x_range = np.linspace(0, L-(L/N), N)
k_range = np.arange(-N/2,N/2,1)
fL_range = [f_L(k) for k in k_range]
coef_1 = [ (1+(flk*dt/2))/(1-(flk*dt/2)) for flk in fL_range]
coef_2 = [dt/(1-(flk*dt/2)) for flk in fL_range]
#plt.plot(k_range, fL_range, label = "fL")
plt.plot(k_range, coef_1, label = "C1")
plt.plot(k_range, coef_2, label = "C2")
plt.legend()
plt.show()
    
def SH(f0):
    u_range = [f0(x) for x in x_range]
    
    
    U = []
    U.append(np.array(u_range))
    
    DFT_0 = (1/N)*np.fft.fftshift(np.fft.fft(u_range))
    DFT = []
    DFT.append(np.array(DFT_0))
    
    uc_range = [u**3 for u in u_range]
    DFTc_0 = (1/N)*np.fft.fftshift(np.fft.fft(uc_range))
    DFTc = []
    DFTc.append(np.array(DFTc_0))

    
    
    t = dt
    while t < t_max:
        d1 = DFT[-1]
        dc1 = DFTc[-1]
        
        if t == dt :
            dc2 = dc1
        else :
            dc2 = DFTc[-2]
        
        uk = []
        for n in range(len(k_range)):
            nex = coef_1[n]*d1[n] + coef_2[n]*((3/2)*dc1[n] - (1/2)*dc2[n])
            uk.append(nex)
        
        DFT.append(np.array(uk))
        
        u = np.real(np.fft.ifft(np.fft.ifftshift(N*uk)))
        U.append(np.array(u))
        
        uc = [uu**3 for uu in u]
        ukc = (1/N)*np.fft.fftshift(np.fft.fft(uc))
        DFTc.append(np.array(ukc))
        
        t += dt
        
    return np.array(U),np.array(DFT)
    

time_ev = SH(u_0)
U = time_ev[0]
D = time_ev[1]
print(U)

plt.plot(x_range, U[1])
plt.show()
plt.clf()

plt.imshow(U)
plt.show()
plt.clf()

plt.imshow(D)
plt.show()
plt.clf()

t_range = np.arange(0,t_max, dt)
[xx,tt]=np.meshgrid(x_range,t_range)
plt.contourf(xx,tt, U)
plt.colorbar()
plt.show()


[kk,tt]=np.meshgrid(k_range,t_range)
plt.contourf(kk,tt, D)
plt.colorbar()
plt.show()
