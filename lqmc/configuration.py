# coding: utf-8
"""
Created on 13 Jan 2020

project: LatticeQMC
version: 1.0
"""
import numpy as np


class Configuration:
    """ Configuration class representing the Hubbard-Stratonovich (HS) field."""

    dtype = np.int8

    def __init__(self, n_sites, time_steps, array=None):
        """ Constructor of the Configuration class

        Parameters
        ----------
        n_sites: int
            Number of spatial lattice sites.
        time_steps: int
            Number of time slices (per site).
        array: np.ndarray, optional
            Existing configuration to use.
        """
        self.n_sites = n_sites
        self.time_steps = time_steps
        self.config = np.ndarray
        if array is not None:
            self.config = array.astype(self.dtype)
        else:
            self.initialize()

    def copy(self):
        """ Creates a (deep) copy of the 'Configuration' instance

        Returns
        -------
        config: Configuration
        """
        return Configuration(self.n_sites, self.time_steps, array=self.config.copy())

    def initialize(self):
        """ Initializes the configuration with a random distribution of -1 and +1 """
        # Create an array of random 0 and 1 and scale array to -1 and 1
        config = 2 * np.random.randint(0, 2, size=(self.n_sites, self.time_steps)) - 1
        self.config = config.astype(self.dtype)

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

    def get(self, i, t):
        """ Returns a specific element of the configuration

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

    def __eq__(self, other):
        return np.all(self.config == other.config)

    def __getitem__(self, item):
        return self.config[item]

    def string_header(self, delim=" "):
        return r"i\l  " + delim.join([f"{i:^3}" for i in range(self.time_steps)])

    def string_bulk(self, delim=" "):
        rows = list()
        for site in range(self.n_sites):
            row = delim.join([f"{x:^3}" for x in self.config[site]])
            rows.append(f"{site:<3} [{row}]")
        return "\n".join(rows)

    def __str__(self):
        delim = " "
        string = self.string_header(delim) + "\n"
        string += self.string_bulk(delim)
        return string
