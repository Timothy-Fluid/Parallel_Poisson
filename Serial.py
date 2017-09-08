import numpy as np
import sys
import matplotlib.pyplot as plt
import time

# smoother
# now jacobi G-S SOR are available


def smoother_Jacobi(phi, phi_0, f, h):
    h2 = h**2
    phi[1:-1, 1:-1] = 0.25 * (phi_0[:-2, 1:-1] + phi_0[1:-1,
                                                       :-2] + phi_0[2:, 1:-1]
                              + phi_0[1:-1, 2:] + h2 * f[1:-1, 1:-1])


def smoother_GS(phi, f, n_nodes):
    h2 = (1. / (n_nodes - 1.))**2
    for i in range(1, n_nodes - 1):
        for j in range(1, n_nodes - 1):
            phi[i, j] = 0.25 * \
                (phi[i - 1, j] + phi[i, j - 1] +
                 phi[i + 1, j] + phi[i, j + 1] + h2 * f[i, j])


def smoother_SOR(phi, f, n_nodes):
    omega = 1.5
    h2 = (1. / (n_nodes - 1.))**2
    for i in range(1, n_nodes - 1):
        for j in range(1, n_nodes - 1):
            phi[i, j] = (1. - omega) * phi[i, j]\
                + omega * 0.25 * \
                (phi[i - 1, j] + phi[i, j - 1] +
                 phi[i + 1, j] + phi[i, j + 1] + h2 * f[i, j])


n_nodes = int(sys.argv[1])
type_smoother = int(sys.argv[2])
ntmax = int(sys.argv[3])

h = 1. / (n_nodes - 1)
# Initialize the array
x = np.linspace(0, 1, n_nodes, dtype=np.float64)
y = np.linspace(0, 1, n_nodes, dtype=np.float64)
X, Y = np.meshgrid(x, y)


# Force the boundary conditions
phi = np.exp(X) * np.exp(-2. * Y)
phi[1:-1, 1:-1] = 0.0


# Calculate the force source
f = -5. * np.exp(X) * np.exp(-2. * Y)
res = np.empty(0, dtype=np.float64)
begin_time = time.time()
if type_smoother == 1:
    for i in range(ntmax):
        phi_0 = phi.copy()
        smoother_Jacobi(phi, phi_0, f, h)
        res = np.append(res, np.linalg.norm(phi - phi_0))
        print("Iteration " + str(i) + "Res = " + str(res[-1]))
        if res[-1] <= 1.e-5:
            print("Converge")
            break
elif type_smoother == 2:
    for i in range(ntmax):
        phi_0 = phi.copy()
        smoother_GS(phi, f, n_nodes)
        res = np.linalg.norm(phi - phi_0)
        print("Iteration " + str(i) + "Res = " + str(res))
        if res <= 1.e-5:
            print("Converge")
            break
elif type_smoother == 3:
    for i in range(ntmax):
        phi_0 = phi.copy()
        smoother_SOR(phi, f, n_nodes)
        res = np.linalg.norm(phi - phi_0)
        print("Iteration " + str(i) + "Res = " + str(res))
        if res <= 1.e-5:
            print("Converge")
            break

end_time = time.time()
print("Time taken: " + str(end_time - begin_time))
# Plot
plt.figure(1)
surf = plt.contourf(X, Y, phi)
plt.colorbar(surf)
plt.title(r'$\phi$')
plt.savefig("Serial.png")

# Plot the residual
t = np.linspace(0, res.size - 1, res.size, dtype=np.int)
plt.figure(2)
plt.plot(t, res)
plt.yscale('log')
plt.xlabel("Iteration")
plt.ylabel(r"$L^2$Residual")
plt.savefig("Residual.png")
