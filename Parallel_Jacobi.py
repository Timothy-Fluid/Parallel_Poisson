import numpy as np
import mpi4py.MPI as MPI
import sys
import time


def smoother_Jacobi(phi, phi_0, f, h):
    h2 = h**2
    phi[1:-1, 1:-1] = 0.25 * (phi_0[:-2, 1:-1] + phi_0[1:-1, :-2] +
                              phi_0[2:, 1:-1] + phi_0[1:-1, 2:] + h2 * f[1:-1, 1:-1])


# Initialize the parallel environment
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

n_nodes = int(sys.argv[1])
ntmax = int(sys.argv[2])

local_size = int((n_nodes - 1) / size + 1)
local_length = 1. / size
h = 1. / (n_nodes - 1)

x = np.linspace(0, 1, n_nodes, dtype=np.float64)
y = np.linspace(rank * local_length, (rank + 1.) *
                local_length, local_size, dtype=np.float64)
X, Y = np.meshgrid(x, y)
local_phi = np.zeros((local_size + 2, n_nodes), dtype=np.float64)

# left and right boundary
local_phi[1:-1, 0] = np.exp(X[:, 0]) * np.exp(-2. * Y[:, 0])
local_phi[1:-1, -1] = np.exp(X[:, -1]) * np.exp(-2 * Y[:, -1])

# Force the boundary to Procs 0 and size-1
if rank == 0:
    local_phi[1, :] = np.exp(X[0, :]) * np.exp(-2. * Y[0, :])
if rank == size - 1:
    local_phi[-2, :] = np.exp(X[-1, :]) * np.exp(-2. * Y[-1, :])

# Force source setting
local_f = np.zeros((local_size + 2, n_nodes), dtype=np.float64)
local_f[1:-1, :] = -5. * np.exp(X) * np.exp(-2. * Y)

res = np.empty(0, dtype=np.float64)
res_t = np.empty(1, dtype=np.float64)
# Decide the domain
begin_row = 0
end_row = local_size + 2
if rank == 0:
    begin_row = 1
if rank == size - 1:
    end_row = local_size + 1

begin_time = time.time()
for nt in range(ntmax):
    # Receive data from blocks below
    if rank < (size - 1):
        comm.Recv([local_phi[-1, :], MPI.DOUBLE], source=rank + 1, tag=1)
    # Send data to blocks above
    if rank > 0:
        comm.Send([local_phi[2, :], MPI.DOUBLE], dest=rank - 1, tag=1)
    # Send data to blocks below
    if rank < (size - 1):
        comm.Send([local_phi[-3, :], MPI.DOUBLE], dest=rank + 1, tag=2)
    # Receive data from blocks above
    if rank > 0:
        comm.Recv([local_phi[0, :], MPI.DOUBLE], source=rank - 1, tag=2)
    local_phi0 = local_phi.copy()
    smoother_Jacobi(local_phi[begin_row:end_row, :],
                    local_phi0[begin_row:end_row, :], local_f[begin_row:end_row, :], h)
    if rank == size - 1:
        local_res = np.array(
            [np.sum((local_phi[1:-1, :] - local_phi0[1:-1, :])**2)])
    else:
        local_res = np.array(
            [np.sum((local_phi[1:-2, :] - local_phi0[1:-2, :])**2)])

    comm.Allreduce(local_res, res, op=MPI.SUM)
    res = np.sqrt(res)
    if rank == 0:
        print("Iteration " + str(nt) + " Res " + str(res[0]))

    if res[0] <= 1.e-5:
        print("The solution has converged")
        break

end_time = time.time()
if rank == 0:
    print("Time taken: " + str(end_time - begin_time))
    # Residual Plot
