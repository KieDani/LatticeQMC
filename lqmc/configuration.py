# coding: utf-8
"""
Created on 13 Jan 2020

project: LatticeQMC
version: 1.0
"""
import numpy as np
from .tools import Plot


class ConfigStatPlot(Plot):

    def __init__(self, xlabel="Sweeps"):
        super().__init__()
        self.ax.set_xlabel(xlabel)
        self.mean = None
        self.var = None
        self.x = list()

    @classmethod
    def empty(cls, xlabel="Sweeps"):
        self = cls(xlabel)
        self.plot([], [])
        return self

    def plot(self, mean, var):
        self.x = list(range(len(mean)))
        self.mean = self.ax.plot(self.x, mean, label="MC Mean")[0]
        self.var = self.ax.plot(self.x, var, label="MC Var")[0]
        self.ax.legend()
        self.ax.grid()

    def update(self, mean, var):
        self.x.append(len(self.x))
        self.mean.set_data(self.x, np.append(self.mean.get_ydata(), mean))
        self.var.set_data(self.x, np.append(self.var.get_ydata(), var))
        self.autoscale()


class ConfigPlot(Plot):

    def __init__(self, config):
        super().__init__()
        self.im = self.ax.matshow(config.config.T, cmap="binary")
        self.ax.xaxis.tick_bottom()
        self.ax.invert_yaxis()
        self.ax.set_xlabel("Sites")
        self.ax.set_ylabel(r"$\tau$")

    def update(self, config):
        self.im.set_data(config.config.T)


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
        array: np.ndarray of np.int8, optional
            Existing configuration to use.
        """
        self.n_sites = n_sites
        self.time_steps = time_steps
        self.config = np.ndarray
        if array is not None:
            self.config = array
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

    def mean(self):
        """ float: Computes the Monte-Carlo sample mean """
        return np.mean(self.config)

    def var(self):
        """ float: Computes the Monte-Carlo sample variance """
        return np.var(self.config)

    def __eq__(self, other):
        return np.all(self.config == other.config)

    def __getitem__(self, item):
        return self.config[item]

    def string_header(self, delim=" "):
        return r"i\l  " + delim.join([f"{i:^3}" for i in range(self.time_steps)])

    def string_bulk(self, delim=" "):
        rows = list()
        for site in range(self.n_sites):
            row = delim.join([f"{x:^3}" for x in self.config[site, :]])
            rows.append(f"{site:<3} [{row}]")
        return "\n".join(rows)

    def __str__(self):
        delim = " "
        string = self.string_header(delim) + "\n"
        string += self.string_bulk(delim)
        return string

    def show(self, show=True):
        plot = ConfigPlot(self)
        if show:
            plot.show()
        return plot
