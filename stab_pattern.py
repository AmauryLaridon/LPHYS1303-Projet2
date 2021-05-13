import matplotlib.pyplot as plt
import numpy as np


r = np.arange(-1,2,0.1)
r_plus = np.arange(1,2,0.001)
u = np.zeros(len(r))

u_zero = np.zeros(len(r))


u_plus = np.sqrt(r_plus-1)
u_moins = -np.sqrt(r_plus-1)


plt.plot(r, u_zero, label='u_0')
plt.plot(r_plus, u_moins, label='u_')
plt.plot(r_plus, u_plus, label='u+')
plt.xlabel('$r$')
plt.ylabel('$u_s$')
plt.title('Distribution uniforme et stationnaire, $u_s$ en fonction de $r$.')
plt.legend()
plt.show()

N = len(r_plus)
k = np.arange(-N/2,N/2,1)

#omega_k = -2*u_plus+2*(k**2)-(k**4)
"""
omega_k = np.zeros((len(k),len(u_plus)))
for i in k:
    omega_k [i,:] = -2*u_plus+2*(i**2)-(i**4)
"""




[kk,uu]=np.meshgrid(k,u_plus)
omega_k = -2*uu+2*(kk**2)-(kk**4)
#condition = omega_k>0
#print(np.shape(omega_k[condition]))
#omega_k_plus = np.zeros(len(condition))
plt.pcolor(kk,uu, omega_k.T)
#plt.plot(k, omega_k[condition])
plt.xlabel("k")
plt.ylabel("u_s")
plt.title("Simulation numérique de l'équation de Swift-Hohenberg")
plt.colorbar()
plt.show()

for i,ka in enumerate(omega_k) :
    for j,k in enumerate(ka):
        if k > 0:
            omega_k[i][j] = 1000
        else:
            omega_k[i][j] = 0

levels = [0,1,1001]
plt.contourf(kk,uu, omega_k.T, levels)
plt.xlabel("$k$")
plt.ylabel("$u_s$")
plt.title("Simulation numérique de l'équation de Swift-Hohenberg")
plt.colorbar()
plt.show()


"""
x = np.linspace(-3, 3, 51)
y = np.linspace(-2, 2, 41)
X, Y = np.meshgrid(x, y)

Z = (1 - X/2 + X**5 + Y**3) * np.exp(-X**2 - Y**2) # calcul du tableau des valeurs de Z

plt.pcolor(X, Y, Z)

plt.show()"""
