# coding: utf-8
"""
Created on 13 Jan 2020

project: LatticeQMC
version: 1.0
"""
import numpy as np


class Configuration:
    """ Configuration class representing the hubbard-Stratonovich (HS) field."""

    def __init__(self, n_sites, time_steps):
        """ Constructor of the Configuration class

        Parameters
        ----------
        n_sites: int
            Number of spatial lattice sites.
        time_steps: int
            Number of time slices (per site).
        """
        self.n_sites = n_sites
        self.time_steps = time_steps
        self.config = np.zeros((n_sites, time_steps), dtype=np.int8)
        self.initialize()

    @property
    def dtype(self):
        return self.config.dtype

    def copy(self):
        """ Copies the configuration instance

        Returns
        -------
        config: Configuration
        """
        config = Configuration(self.n_sites, self.time_steps)
        config.config = np.copy(self.config)
        return config

    def initialize(self):
        """ Initializes the configuration with a random distribution of -1 and +1 """
        # Create an array of random 0 and 1.
        config = np.random.randint(0, 2, size=(self.n_sites, self.time_steps))
        # Scale array to -1 and 1
        self.config = 2*config - 1

    def update(self, i, t):
        """ Update element of array by flipping its spin-value

        Parameters
        ----------
        i: int
            Site index.
        t: int
            Time slice index.
        """
        self.config[i, t] *= -1

    def get(self):
        """ (n, m) np.ndarray: Spin-configuration"""
        return self.config

    def get_element(self, i, t):
        """ Return a element of the array

        Parameters
        ----------
        i: int
            Site index.
        t: int
            Time slice index.

        Returns
        -------
        s: int
        """
        return self.config[i, t]

    def __getitem__(self, item):
        return self.config[item]

    def __str__(self):
        delim = " "
        rows = [r"i\l  " + delim.join([f"{i:^3}" for i in range(self.time_steps)])]
        for site in range(self.n_sites):
            row = delim.join([f"{x:^3}" for x in self.config[site]])
            rows.append(f"{site:<3} [{row}]")
        return "\n".join(rows)


