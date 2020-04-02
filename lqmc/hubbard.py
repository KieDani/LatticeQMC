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
        """ Object representing the Hubbard model.

        Parameters
        ----------
        u: float, optional
            Interaction energy of the Hubbard model.
        t: float, optional
            Hopping energy of the Hubbard model.
        mu: float, optional
            Chemical potential of the Hubbard model.
            If `None` the chemical potential is set to ensure half filling.
        """
        self.lattice = Lattice.square()
        self.u = u
        self.t = t
        self.mu = mu if mu is not None else u/2

    def param_str(self):
        return f"u={self.u}_t={self.t}_mu={self.mu}"

    def __str__(self):
        return f"HubbardModel(u={self.u}, t={self.t}, mu={self.mu})"

    @property
    def n_sites(self):
        """ int: Total number of lattice sites of the model"""
        return self.lattice.n_sites

    def build(self, width, height=1, cycling=0):
        """ Builds the lattice of the Hubbard model with the given shape.

        Parameters
        ----------
        width: int
            The number of sites in the `x`-direction.
        height: int, optional
            The number of sites in the `y`-direction.
            The default is `1`(1D lattice).
        cycling: int or array_like of int, optional
            Axis that will be set to periodic boundary conditions.
            The default is periodic boundary conditions along the `x`-axis.
        """
        self.lattice.build((width, height))
        if cycling is not None:
            self.lattice.set_periodic_boundary(cycling)

    def build_square(self, size, cycling=(0, 1)):
        """ Builds a square lattice with the given size.

        Parameters
        ----------
        size: int
            The number of sites in both directions.
        cycling: int or array_like of int, optional
            Axis that will be set to periodic boundary conditions.
            The default is periodic boundary conditions along both axes.
        """
        self.build(size, size, cycling)

    def ham_kinetic(self):
        """ Constructs the kinetic Hamiltonian matrix of the Hubbard model.

        Returns
        -------
        ham_kin: (N, N) np.ndarray
        """
        # Create hamiltonian with diagonal elements
        energy = -self.mu
        ham = energy * np.eye(self.lattice.n_sites, dtype=np.float64)
        # Add hopping terms
        # i is the index of the current site. The lattice
        # returns the second index, j, which corresponds to
        # the nearest neighbors of the site:
        # H_ij = t, H_ji = t^*
        for i in range(self.lattice.n_sites):
            for j in self.lattice.nearest_neighbours(i):
                if i > j:
                    ham[i, j] = -self.t
                    ham[j, i] = -np.conj(self.t)
        return ham
