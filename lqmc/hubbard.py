# coding: utf-8
"""
Created on 13 Jan 2020

project: LatticeQMC
version: 1.0
"""
import numpy as np
from .lattice import Lattice


class HubbardModel:

    def __init__(self, u=4, t=1, mu=None):
        self.lattice = Lattice.square()
        self.u = u
        self.t = t
        self.mu = mu if mu is not None else u/2  # Half filling if None

        self.graph = None

    def param_str(self):
        return f"u={self.u}_t={self.t}_mu={self.mu}"

    def __str__(self):
        return f"HubbardModel(u={self.u}, t={self.t}, mu={self.mu})"

    @property
    def n_sites(self):
        return self.lattice.n_sites

    def build(self, width, height=1):
        self.lattice.build((width, height))

    def ham_kinetic(self, cycling=False):
        n = self.lattice.n_sites

        # Create hamiltonian with diagonal elements
        energy = self.u/2 - self.mu
        ham = energy * np.eye(n, dtype=np.float64)

        # Add hopping terms
        # i is the index of the current site. The lattice
        # returns the second index, j, which corresponds to
        # the nearest neighbors of the site:
        # H_ij = t, H_ji = t^*
        for i in range(n):
            for j in self.lattice.nearest_neighbours(i):
                ham[i, j] = -self.t
                ham[j, i] = -np.conj(self.t)

        if cycling:
            i, j = 0, -1
            ham[i, j] = -self.t
            ham[j, i] = np.conj(-self.t)
        return ham
