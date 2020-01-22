"""Calculate non-interacting imaginary time Green's function.

For this example gftool version 0.6.0 from PyPi was used.
The functions (or at least most of them...) act like numpy gu-functions.

"""
import numpy as np
import gftool as gt
import gftool.matrix
from gftool import fourier, pade


import matplotlib.pyplot as plt

# example hamiltonion: 3x3 with open boundary conditions
BETA = 4
t = 1
mu = 0
hamilton = np.array([
    [mu, -t, 0.],
    [-t, mu, -t],
    [0., -t, mu]
])


hamil_dec = gt.matrix.decompose_hamiltonian(hamilton)
tau = np.linspace(0, BETA, num=2049)
# append axis, as we don't wan't the sum here
diag_gf_tau = gt.pole_gf_tau(tau, hamil_dec.xi[..., np.newaxis], weights=1, beta=BETA)
# I always put shape (site, site, tau)
gf_tau = np.moveaxis(hamil_dec.reconstruct(diag_gf_tau, kind='full'), 0, -1)

for site, linestd in zip(range(3), ['-', '--', ':']):
    plt.plot(tau, gf_tau[site, site], label=f"site {site}", linestyle=linestd)
plt.xlabel('tau')
plt.legend()
plt.show()