# coding: utf-8
"""
Created on 15 Jan 2020

project: LatticeQMC
version: 1.0

To Do
-----
- Make console output thread-safe
- Green's function time slices
- Efficiency
"""
import logging
import itertools
import numpy as np
from scipy.linalg import expm
from lqmc import HubbardModel, Configuration


def updateln(string):
    """ Rewrites the current console line

    Parameters
    ----------
    string: str
        String to display.
    """
    print("\r" + string, end="", flush=True)


def compute_lambda(u, dtau):
    r""" Computes the factor '\lambda' used in the computation of 'M'.

    Parameters
    ----------
    u: float
        Hubbard interaction :math:`U`.
    dtau: float
        Time slice size of the HS-field.

    Returns
    -------
    lamb: float
    """
    return np.arccosh(np.exp(u * dtau / 2.))


def compute_m(ham_kin, config, lamb, dtau, sigma):
    r""" Computes the matrix :math:'M' used in the Metropolis ratio

    The matrix :math:'M' is defined as
    .. math::
       M_\sigma(h) = I + B_{L, \sigma}(h_{L}) B_{L-1, \sigma}(h_{L-1}) \dots B_{1, \sigma}(h_{1})

    with
    .. math::
       B_{l, \sigma}(h_l) = e^{\Delta\tau K} e^{\sigma \lambda V_l(h_l)}

    .math'V_l(h_l) = diag(h_{1l}, \dots h_{Nl})' is the diagonal matrix of a time slice
    of the HS-field .math'h_{il}'

    Parameters
    ----------
    ham_kin: (N, N) np.ndarray
        Kinetic hamiltonian .math'K'.
    config: Configuration
        The current HS-field configuration object.
        This contains the field .math'h_{il}'
    lamb: float
        Factor .math'\lambda'.
    dtau: float
        Time slice size of the HS-field.
    sigma: int
        Spin index.

    Returns
    -------
    m: (N, N) np.ndarray
    """
    n = ham_kin.shape[0]

    # Calculate the first matrix exp of B. This is a static value.
    # check if there is a better way to calculate the matrix exponential
    exp_k = expm(dtau * ham_kin)
    # Create the V_l matrix
    exp_v = np.zeros((n, n), dtype=config.dtype)

    # fill diag(V_l) with values of last time slice and compute B-product
    lmax = config.time_steps - 1

    np.fill_diagonal(exp_v, (sigma * lamb * config[:, lmax]))
    b = np.dot(exp_k, exp_v)

    b_prod = b
    for l in reversed(range(1, lmax)):
        # Fill V_l with new values, compute B(l) and multiply with total product
        np.fill_diagonal(exp_v, (sigma * lamb * config[:, l]))
        b = np.dot(exp_k, exp_v)
        b_prod = np.dot(b_prod, b)

    # compute M matrix
    return np.eye(n) + b_prod


def compute_gf_tau(config, ham_kin, g_beta, lamb, dtau, sigma):
    r""" Computes the Green's function for all time slices

    Parameters
    ----------
    config: Configuration
        The current HS-field configuration object.
        This contains the field .math'h_{il}'
    ham_kin: (N, N) np.ndarray
        Kinetic hamiltonian .math'K'.
    g_beta: (N, N) np.ndarray
        The Green's function matrix for all spatial sites N at time .math'\beta'
    lamb: float
        Factor .math'\lambda'.
    dtau: float
        Time slice size of the HS-field.
    sigma: int
        Spin index.

    Returns
    -------
    gf: (M, N, N) np.ndarray
        The Green's function for N sites and M time slices.
    """
    n = ham_kin.shape[0]

    # Calculate the first matrix exp of B. This is a static value.
    # check if there is a better way to calculate the matrix exponential
    exp_k = expm(dtau * ham_kin)
    exp_min_k = expm(-1 * dtau * ham_kin)

    exp_v = np.zeros((n, n), dtype=config.dtype)
    exp_min_v = np.zeros((n, n), dtype=config.dtype)

    # g[0, :, :] is Greensfunction at time beta, g[1, :, :] is Greensfunction one step before, etc
    g = np.zeros((config.time_steps, n, n), dtype=np.float64)
    g[0, :, :] = g_beta
    for l in range(1, config.time_steps):
        # Create the V_l matrix
        np.fill_diagonal(exp_v, (sigma * lamb * config[:, l]))
        np.fill_diagonal(exp_min_v, (-1 * sigma * lamb * config[:, l]))

        b = np.dot(exp_k, exp_v)
        b_min = np.dot(exp_min_k, exp_min_v)
        g[l, :, :] = np.dot(np.dot(b, g[l-1, :, :]), np.linalg.inv(b_min))
    return g


def warmup_loop(model, config, dtau, sweeps=200, fast=True):
    """ Runs the warmup lqmc-loop

    Parameters
    ----------
    model: HubbardModel
        The Hubbard model instance.
    config: Configuration
        The current HS-field configuration object after warmup.
        This contains the field .math'h_{il}'
    dtau: float
        Time slice size of the HS-field.
    sweeps: int, optional
        Number of sweeps through the HS-field.
    fast: bool, optional
        Flag if the fast algorithm should be used. The default is True.

    Returns
    -------
    config: Configuration
        The updated HS-field configuration after the warmup loop.
    """
    ham_kin = model.ham_kinetic()
    # Calculate factor
    lamb = compute_lambda(model.u, dtau)
    # Calculate M matrices
    m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
    m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)
    # Store copy of current configuration (only used for slow algorith)
    old_config = config.copy()

    # QMC loop
    acc = False
    ratio = 0
    # updateln("Warmup sweep")
    for sweep in range(sweeps):
        for i, l in itertools.product(range(model.n_sites), range(config.time_steps)):
            updateln(f"Warmup sweep: {sweep+1}/{sweeps}, accepted: {acc} (ratio={ratio:.2f})")
            if fast:
                # Calculate m-matrices and ratio of the configurations
                # Accept move with metropolis acceptance ratio.
                m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
                m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
                d_up = 1 + (1 - np.linalg.inv(m_up)[i, i]) * (np.exp(-2 * lamb * config[i, l]) - 1)
                d_dn = 1 + (1 - np.linalg.inv(m_dn)[i, i]) * (np.exp(+2 * lamb * config[i, l]) - 1)
                ratio = d_up * d_dn
                r = np.random.rand()  # Random number between 0 and 1
                if r < ratio:
                    # Move accepted: Update configuration
                    acc = True
                    config.update(i, l)
                else:
                    # Move not accepted!
                    acc = False
            else:
                # Update Configuration
                config.update(i, l)
                # Calculate m-matrices and ratio of the configurations
                # Accept move with metropolis acceptance ratio.
                m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
                m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
                new = np.linalg.det(m_up) * np.linalg.det(m_dn)
                ratio = new / old
                r = np.random.rand()  # Random number between 0 and 1
                if r < ratio:
                    # Move accepted: Continue using the new configuration
                    acc = True
                    old = new
                    old_config = config.copy()
                else:
                    # Move not accepted: Revert to the old configuration
                    acc = False
                    config = old_config

            # logging.info(f"[Warmup] Sweep={sweep} i={i}, l={l} - ratio={ratio:.3f}, accepted={acc}")
    print()
    return config


def measure_loop(model, config, dtau, sweeps=800, fast=True):
    r""" Runs the measurement lqmc-loop and returns the measured Green's function

    Parameters
    ----------
    model: HubbardModel
        The Hubbard model instance.
    config: Configuration
        The current HS-field configuration object.
        This contains the field .math'h_{il}'
    dtau: float
        Time slice size of the HS-field.
    sweeps: int, optional
        Number of sweeps through the HS-field.
    fast: bool, optional
        Flag if the fast algorithm should be used. The default is True.

    Returns
    -------
    gf_dn: (M, N, N) np.ndarray
        Measured spin-up Green's function .math'G_\uparrow(\tau)' for all time slices.
    gf_dn: (M, N, N) np.ndarray
        Measured spin-down Green's function .math'G_\downarrow(\tau)' for all time slices.
    """
    ham_kin = model.ham_kinetic()
    n = model.n_sites
    # Calculate factor
    lamb = compute_lambda(model.u, dtau)
    # Calculate M matrices
    m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
    m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
    old = np.linalg.det(m_up) * np.linalg.det(m_dn)
    # Initialize total and temp greens functions
    gf_up, gf_dn = 0, 0
    g_beta_up = np.linalg.inv(m_up)
    g_beta_dn = np.linalg.inv(m_dn)
    g_tmp_up = compute_gf_tau(config, ham_kin, g_beta_up, lamb, dtau, sigma=+1)
    g_tmp_dn = compute_gf_tau(config, ham_kin, g_beta_dn, lamb, dtau, sigma=-1)
    # Store copy of current configuration (only used for slow algorith)
    old_config = config.copy()

    # QMC loop
    acc = False
    ratio = 0
    number = 0
    updateln("Measurement sweep")
    for sweep in range(sweeps):
        for i, l in itertools.product(range(model.n_sites), range(config.time_steps)):
            updateln(f"Measurement sweep: {sweep+1}/{sweeps}, accepted: {acc} (ratio={ratio:.2f})")

            if fast:
                # Calculate m-matrices and ratio of the configurations
                # Accept move with metropolis acceptance ratio.
                m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
                m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
                d_up = 1 + (1 - np.linalg.inv(m_up)[i, i]) * (np.exp(-2 * lamb * config[i, l]) - 1)
                d_dn = 1 + (1 - np.linalg.inv(m_dn)[i, i]) * (np.exp(+2 * lamb * config[i, l]) - 1)
                ratio = d_up * d_dn
                r = np.random.rand()  # Random number between 0 and 1
                if r < ratio:
                    # Move accepted: Update temp greens function and update configuration
                    c_up = np.zeros(n, dtype=np.float64)
                    c_dn = np.zeros(n, dtype=np.float64)
                    c_up[i] = np.exp(-2 * lamb * config[i, l]) - 1
                    c_dn[i] = np.exp(+2 * lamb * config[i, l]) - 1
                    c_up = -1 * (np.exp(-2 * lamb * config[i, l]) - 1) * g_beta_up[i, :] + c_up
                    c_dn = -1 * (np.exp(+2 * lamb * config[i, l]) - 1) * g_beta_dn[i, :] + c_dn

                    b_up = g_beta_up[:, i] / (1. + c_up[i])
                    b_dn = g_beta_dn[:, i] / (1. + c_dn[i])

                    g_beta_up = g_beta_up - np.outer(b_up, c_up)
                    g_beta_dn = g_beta_dn - np.outer(b_dn, c_dn)
                    g_tmp_up = compute_gf_tau(config, ham_kin, g_beta_up, lamb, dtau, sigma=+1)
                    g_tmp_dn = compute_gf_tau(config, ham_kin, g_beta_dn, lamb, dtau, sigma=-1)

                    # Update Configuration
                    acc = True
                    config.update(i, l)
                else:
                    # Move not accepted!
                    acc = False
            else:
                # Update Configuration
                config.update(i, l)
                # Calculate m-matrices and ratio of the configurations
                # Accept move with metropolis acceptance ratio.
                m_up = compute_m(ham_kin, config, lamb, dtau, sigma=+1)
                m_dn = compute_m(ham_kin, config, lamb, dtau, sigma=-1)
                new = np.linalg.det(m_up) * np.linalg.det(m_dn)
                ratio = new / old
                r = np.random.rand()  # Random number between 0 and 1
                if r < ratio:
                    # Move accepted:
                    # Update temp greens function and continue using the new configuration
                    g_beta_up = np.linalg.inv(m_up)
                    g_beta_dn = np.linalg.inv(m_dn)
                    g_tmp_up = compute_gf_tau(config, ham_kin, g_beta_up, dtau, lamb, sigma=+1)
                    g_tmp_dn = compute_gf_tau(config, ham_kin, g_beta_dn, dtau, lamb, sigma=-1)

                    acc = True
                    old = new
                    old_config = config.copy()
                else:
                    # Move not accepted:
                    # Revert to the old configuration
                    acc = False
                    config = old_config

            # Add temp greens function to total gf after each step
            gf_up += g_tmp_up
            gf_dn += g_tmp_dn
            number += 1

            # logging.info(f"[Measure] Sweep={sweep} i={i}, l={l} - ratio={ratio:.3f}, accepted={acc}")
    print()
    # Return the normalized gfs for each spin
    return np.array([gf_up, gf_dn]) / number


class LatticeQMC:

    def __init__(self, model, beta, time_steps, sweeps=1000, warmup_ratio=0.2):
        """ Initialize the Lattice Quantum Monte-Carlo solver.

        Parameters
        ----------
        model: HubbardModel
            The Hubbard model instance.
        beta: float
            The inverse temperature .math'\beta = 1/T'.
        time_steps: int
            Number of time steps from .math'0' to .math'\beta'
        sweeps: int, optional
            Total number of sweeps (warmup + measurement)
        warmup_ratio: float, optional
            The ratio of sweeps used for warmup. The default is '0.2'.
        """
        self.model = model
        self.dtau = beta / time_steps
        self.warm_sweeps = int(sweeps * warmup_ratio)
        self.meas_sweeps = sweeps - self.warm_sweeps

        self.config = Configuration(model.n_sites, time_steps)
        self.ham_kin = self.model.ham_kinetic()
        self.lamb = compute_lambda(self.model.u, self.dtau)

        self.status = ""
        self.it = 0

    def log_iterstep(self, sweep, i, l, ratio, acc):
        logging.debug(f"{self.status} {sweep} i={i}, l={l} - ratio={ratio:.3f}, accepted={acc}")

    def loop_generator(self, sweeps):
        """ Generates the indices of the LQMC loop.

        This is mainly used to saving total iteration number and
        for hooking logging and printing events into the loops easily.

        Parameters
        ----------
        sweeps: int
            Number of sweeps of current loop.

        Yields
        -------
        indices: tuple of int
            Indices of current iteration step consisting of .term'sweep', .term'i' and .term'l'.
        """
        for sweep in range(sweeps):
            self.it = sweep
            for i, l in itertools.product(range(self.model.n_sites), range(self.config.time_steps)):
                print(f"\r{self.status} Sweep {sweep} [{i}, {l}]", end="", flush=True)
                yield sweep, i, l

    def _compute_m(self):
        """ Computes the 'M' matrices for both spins.

        Returns
        -------
        m_up: (N, N) np.ndarray
        m_dn: (N, N) np.ndarray
        """
        m_up = compute_m(self.ham_kin, self.config, self.lamb, self.dtau, sigma=+1)
        m_dn = compute_m(self.ham_kin, self.config, self.lamb, self.dtau, sigma=-1)
        return m_up, m_dn

    def _compute_gf_tau(self, g_beta_up, g_beta_dn):
        """ Computes the time dependend Green's function for both spins

        Returns
        -------
        gf_up: (M, N, N) np.ndarray
            The spin-up Green's function for N sites and M time slices.
        gf_dn: (M, N, N) np.ndarray
            The spin-down Green's function for N sites and M time slices.
        """
        g_tau_up = compute_gf_tau(self.config, self.ham_kin, g_beta_up, self.lamb, self.dtau, sigma=+1)
        g_tau_dn = compute_gf_tau(self.config, self.ham_kin, g_beta_dn, self.lamb, self.dtau, sigma=-1)
        return g_tau_up, g_tau_dn

    # def _ratio(self, i, l, m_up, m_dn):
    #     """ Not used """
    #     d_up = 1 + (1 - np.linalg.inv(m_up)[i, i]) * (np.exp(-2 * self.lamb * self.config[i, l]) - 1)
    #     d_dn = 1 + (1 - np.linalg.inv(m_dn)[i, i]) * (np.exp(+2 * self.lamb * self.config[i, l]) - 1)
    #     return d_up * d_dn

    def warmup_loop_det(self):
        """ Runs the slow version of the LQMC warmup-loop """
        self.status = "Warmup"
        m_up, m_dn = self._compute_m()  # Calculate M matrices for both spins
        old = np.linalg.det(m_up) * np.linalg.det(m_dn)
        old_config = self.config.copy()  # Store copy of current configuration
        # QMC loop
        for sweep, i, l in self.loop_generator(self.warm_sweeps):
            # Update Configuration
            self.config.update(i, l)

            m_up, m_dn = self._compute_m()  # Calculate M matrices for both spins
            new = np.linalg.det(m_up) * np.linalg.det(m_dn)
            ratio = new / old
            acc = np.random.rand() < ratio
            if acc:
                # Move accepted: Continue using the new configuration
                old = new
                old_config = self.config.copy()
            else:
                # Move not accepted: Revert to the old configuration
                self.config = old_config

    def measure_loop_det(self):
        r""" Runs the slow version of the LQMC measurement-loop and returns the Green's function.

        Returns
        -------
        gf_dn: (M, N, N) np.ndarray
            Measured spin-up Green's function .math'G_\uparrow(\tau)' for all M time slices.
        gf_dn: (M, N, N) np.ndarray
            Measured spin-down Green's function .math'G_\downarrow(\tau)' for all M time slices.
        """
        self.status = "Measurement"

        m_up, m_dn = self._compute_m()  # Calculate M matrices for both spins
        old = np.linalg.det(m_up) * np.linalg.det(m_dn)
        old_config = self.config.copy()  # Store copy of current configuration

        # Initialize total and temp greens functions
        gf_up, gf_dn = 0, 0
        g_beta_up = np.linalg.inv(m_up)
        g_beta_dn = np.linalg.inv(m_dn)
        g_tmp_up, g_tmp_dn = self._compute_gf_tau(g_beta_up, g_beta_dn)

        # QMC loop
        number = 0
        for sweep, i, l in self.loop_generator(self.meas_sweeps):
            # Update Configuration
            self.config.update(i, l)

            m_up, m_dn = self._compute_m()  # Calculate M matrices for both spins
            new = np.linalg.det(m_up) * np.linalg.det(m_dn)
            ratio = new / old
            acc = np.random.rand() < ratio
            if acc:
                # Move accepted:
                # Update temp greens function and continue using the new configuration
                g_beta_up = np.linalg.inv(m_up)
                g_beta_dn = np.linalg.inv(m_dn)
                g_tmp_up, g_tmp_dn = self._compute_gf_tau(g_beta_up, g_beta_dn)
                old = new
                old_config = self.config.copy()
            else:
                # Move not accepted
                self.config = old_config

            # Add temp greens function to total gf after each step
            gf_up += g_tmp_up
            gf_dn += g_tmp_dn
            number += 1
        # Return the normalized gfs for each spin
        return np.array([gf_up, gf_dn]) / number

    def warmup_loop(self):
        """ Runs the fast version of the LQMC warmup-loop """
        self.status = "Warmup"
        # QMC loop
        for sweep, i, l in self.loop_generator(self.warm_sweeps):
            m_up, m_dn = self._compute_m()  # Calculate M matrices for both spins
            d_up = 1 + (1 - np.linalg.inv(m_up)[i, i]) * (np.exp(-2 * self.lamb * self.config[i, l]) - 1)
            d_dn = 1 + (1 - np.linalg.inv(m_dn)[i, i]) * (np.exp(+2 * self.lamb * self.config[i, l]) - 1)
            ratio = d_up * d_dn
            acc = np.random.rand() < ratio
            if acc:
                self.config.update(i, l)

    def measure_loop(self):
        r""" Runs the fast version of the LQMC measurement-loop and returns the Green's function.

        Returns
        -------
        gf_dn: (M, N, N) np.ndarray
            Measured spin-up Green's function .math'G_\uparrow(\tau)' for all M time slices.
        gf_dn: (M, N, N) np.ndarray
            Measured spin-down Green's function .math'G_\downarrow(\tau)' for all M time slices.
        """
        self.status = "Measurement"
        n = self.model.n_sites
        # Initialize total and temp greens functions
        gf_up, gf_dn = 0, 0
        m_up, m_dn = self._compute_m()  # Calculate M matrices for both spins
        g_beta_up = np.linalg.inv(m_up)
        g_beta_dn = np.linalg.inv(m_dn)
        g_tmp_up, g_tmp_dn = self._compute_gf_tau(g_beta_up, g_beta_dn)

        # QMC loop
        number = 0
        for sweep, i, l in self.loop_generator(self.meas_sweeps):
            m_up, m_dn = self._compute_m()  # Calculate M matrices for both spins
            d_up = 1 + (1 - np.linalg.inv(m_up)[i, i]) * (np.exp(-2 * self.lamb * self.config[i, l]) - 1)
            d_dn = 1 + (1 - np.linalg.inv(m_dn)[i, i]) * (np.exp(+2 * self.lamb * self.config[i, l]) - 1)
            ratio = d_up * d_dn
            acc = np.random.rand() < ratio
            if acc:
                # Move accepted: Update temp greens function and update configuration
                c_up = np.zeros(n, dtype=np.float64)
                c_dn = np.zeros(n, dtype=np.float64)
                c_up[i] = np.exp(-2 * self.lamb * self.config[i, l]) - 1
                c_dn[i] = np.exp(+2 * self.lamb * self.config[i, l]) - 1
                c_up = -1 * (np.exp(-2 * self.lamb * self.config[i, l]) - 1) * g_beta_up[i, :] + c_up
                c_dn = -1 * (np.exp(+2 * self.lamb * self.config[i, l]) - 1) * g_beta_dn[i, :] + c_dn
                b_up = g_beta_up[:, i] / (1. + c_up[i])
                b_dn = g_beta_dn[:, i] / (1. + c_dn[i])

                g_beta_up = g_beta_up - np.outer(b_up, c_up)
                g_beta_dn = g_beta_dn - np.outer(b_dn, c_dn)
                g_tmp_up, g_tmp_dn = self._compute_gf_tau(g_beta_up, g_beta_dn)

                # Update Configuration
                self.config.update(i, l)

            # Add temp greens function to total gf after each step
            gf_up += g_tmp_up
            gf_dn += g_tmp_dn
            number += 1
        # Return the normalized gfs for each spin
        return np.array([gf_up, gf_dn]) / number
