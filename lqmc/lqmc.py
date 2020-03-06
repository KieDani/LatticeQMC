# coding: utf-8
"""
Created on 15 Jan 2020

project: LatticeQMC
version: 1.0
"""
import time
import numpy as np
from scipy.linalg import expm
from lqmc import HubbardModel, Configuration
from lqmc.configuration import ConfigStatPlot
from lqmc.logging import get_logger, DEBUG


class LatticeQMC:

    def __init__(self, model, time_steps, warmup=300, sweeps=2000, det_mode=False):
        """ Initialize the Lattice Quantum Monte-Carlo solver.

        Parameters
        ----------
        model: HubbardModel
            The Hubbard model instance.
        time_steps: int
            Number of time steps from .math'0' to .math'\beta'.
        warmup int, optional
            Number of warmup sweeps.
        sweeps: int, optional
            Number of measurement sweeps.
        det_mode: bool, optional
            Flag for the calculation mode. If 'True' the slow algorithm via
            the determinants is used. The default i9s 'False' (faster).
        """
        self.logger = get_logger()
        self.logger.setLevel(DEBUG)
        self.logger.debug(f"INIT")

        # Constant attributes
        self.model = model
        self.config = Configuration(model.n_sites, time_steps)
        self.n_sites = model.n_sites
        self.time_steps = time_steps
        self.warm_sweeps = warmup
        self.meas_sweeps = sweeps

        # Iteration and mode attributes
        self.det_mode = det_mode
        self.status = ""
        self.it = 0
        self.ratio = 0.0
        self.acc = False

        # Cached and temperature-dependend attributes
        self.ham_kin = self.model.ham_kinetic()
        self.beta = 0.
        self.dtau = 0.
        self.lamb = 0.
        self.exp_k = None
        self.exp_k_inv = None

        self.logger.debug(f"u=          {self.model.u}")
        self.logger.debug(f"t=          {self.model.t}")
        self.logger.debug(f"mu=         {self.model.mu}")
        self.logger.debug(f"sites=      {self.n_sites}")
        self.logger.debug(f"time_steps= {self.time_steps}")
        self.logger.debug(f"det_mode=   {self.det_mode}")
        self.logger.info(f"Warmup=     {self.warm_sweeps}")
        self.logger.info(f"Measurement={self.meas_sweeps}")
        self.logger.debug(f"END INIT")

    def set_temperature(self, temp):
        """ Sets the temperature and initializes the calculation.

        Parameters
        ----------
        temp: float
        """
        self.logger.debug(f"SETUP")
        beta = 1 / temp
        self.dtau = beta / self.time_steps
        self.beta = beta

        self.lamb = np.arccosh(np.exp(self.model.u * self.dtau / 2.)) if self.model.u else 0
        self.exp_k = expm(+1 * self.dtau * self.ham_kin)
        self.exp_k_inv = expm(-1 * self.dtau * self.ham_kin)
        check_val = self.model.u * self.model.t * self.dtau**2

        self.logger.debug(f"beta=       {self.beta}")
        self.logger.debug(f"dtau=       {self.dtau}")
        self.logger.debug(f"lambda=     {self.lamb}")
        if check_val < 0.1:
            self.logger.info(f"Check-value {check_val:.2} is smaller than 0.1!")
        else:
            self.logger.warning(f"Check-value {check_val:.2} should be smaller than 0.1!")
        self.logger.debug(f"END SETUP")

    # =========================================================================

    def get_exp_v(self, l, sigma, inv=False, matrix_exp=False):
        r""" Computes the Matrix exponential of 'V_\sigma(l)'

        Notes
        -----
        Since .math:'V_\sigma(l) = diag(h_{l, 1}, \dots, h_{l, N}' is a diagonal matrix,
        the numerical matrix exponential is not needed. The exponential of the diagonal elements
        can be computed directly.

        Parameters
        ----------
        l: int
            Time-slice index.
        sigma: int
            Spin value.
        inv: bool, optional
            Flag if inverse should be computed.
        matrix_exp: bool, optional
            Flag if the matrix-exponential should be computed numerically.

        Returns
        -------
        exp_v: (N, N) np.ndarray
        """
        sign = -1 if inv else +1
        diag = sigma * self.lamb * self.config[:, l]
        if matrix_exp:
            return expm(np.diagflat(sign * diag))
        else:
            return np.diagflat(np.exp(sign * diag))

    def get_m(self, l0, sigma):
        r""" Computes the 'M' matrices for spin '\sigma'

        Notes
        -----
        In the warmup loop of the determinant-mode the cyclic permutation is not needed.
        The timeslice 'l0' can be left at 0 to skip the first loop in the computation
        of the matrix-product.

        Parameters
        ----------
        l0: int
            Time-slice index used for cyclic permutation.
        sigma: int
            Spin value.

        Returns
        -------
        m: (N, N) np.ndarray
        """
        # Initialize time slices in cyclic permutation:
        # l0, l0-1, ..., 0, L-1, L-2, ..., l0+1
        l0 = l0 % self.time_steps
        indices = list(reversed(range(self.config.time_steps)))
        time_indices = indices[-l0:] + indices[:-l0]
        # compute A=prod(B_l)
        exp_v = self.get_exp_v(time_indices[0], sigma)
        b_prod = self.exp_k.dot(exp_v)
        for l in time_indices[1:]:
            exp_v = self.get_exp_v(l, sigma)
            b = self.exp_k.dot(exp_v)
            b_prod = np.dot(b_prod, b)
        # Assemble M=I+prod(B)
        return np.eye(self.n_sites) + b_prod

    def _gf_tau(self, g_beta, sigma):
        """ Computes the Green's function for all time slices recursively.

        Returns
        -------
        gf_tau: (M, N, N) np.ndarray
            The Green's function for N sites and M time slices.
        """
        # First index of g[0, :, :] represents the time slices
        g = np.zeros((self.time_steps, self.n_sites, self.n_sites), dtype=np.float64)
        g[0, :, :] = g_beta
        for l in range(1, self.time_steps):
            exp_v = self.get_exp_v(l, sigma)
            exp_v_inv = self.get_exp_v(l, sigma, inv=True)
            b = np.dot(exp_v, self.exp_k)
            # b_inv = np.linalg.inv(b)
            b_inv = np.dot(exp_v_inv, self.exp_k_inv)
            g[l, ...] = np.dot(np.dot(b_inv, g[l - 1, ...]), b)
        return g[::-1]

    def _compute_gf_tau(self, g_beta_up, g_beta_dn):
        """ Computes the time dependend Green's function for both spins

        Returns
        -------
        gf_up: (M, N, N) np.ndarray
            The spin-up Green's function for N sites and M time slices.
        gf_dn: (M, N, N) np.ndarray
            The spin-down Green's function for N sites and M time slices.
        """
        # compute_gf_tau(self.config, self.ham_kin, g_beta_up, self.lamb, self.dtau, sigma=+1)
        # compute_gf_tau(self.config, self.ham_kin, g_beta_dn, self.lamb, self.dtau, sigma=-1)
        g_tau_up = self._gf_tau(g_beta_up, sigma=+1)
        g_tau_dn = self._gf_tau(g_beta_dn, sigma=-1)
        return g_tau_up, g_tau_dn

    # =========================================================================

    def _debug(self, i, l):
        """ Logs the attributes of the iteration.

        Format of log is:
        <Status> <iteration> -- <time_slice> <site> -- <ratio> (<accepted>) -- <mc mean> <mc var>

        Parameters
        ----------
        i: int
            Site index.
        l: int
            Time-slice index.
        """
        log = f"{self.status} {self.it + 1} -- {l:>2} {i:>2} -- {self.ratio:.1f} ({self.acc})"
        log += f" -- {self.config.mean():.3f} {self.config.var():.3f}"
        self.logger.debug(log)

    def iter_sweeps(self, n, console_updates=200):
        self.logger.debug(self.status.upper())
        print_interval = max(1, int(n/console_updates))
        for it in range(n):
            count = it + 1
            if it < n and count % print_interval == 0:
                string = f"{self.status} Sweep {it + 1} ({100 * count / n:.1f}%)"
                string += f" [Mean: {self.config.mean():5.2f}, Var: {self.config.var():5.2f}]"
                print("\r" + string, end="", flush=True)
            self.it = it
            yield it
        print(f"\r{self.status} Sweep {n} (100.0%)")
        self.logger.debug("END " + self.status.upper())

    def _warmup_step_det(self, old_det):
        # Iterate over all time-steps, starting at the end (.math:'\beta')
        for l in reversed(range(self.time_steps)):
            # Iterate over all lattice sites
            for i in range(self.n_sites):
                # Update Configuration by flipping spin
                self.config.update(i, l)
                # Compute updated M matrices after config-update
                m_up = self.get_m(0, sigma=+1)
                m_dn = self.get_m(0, sigma=-1)
                # Compute the new determinant for both matrices for the acceptance ratio
                new_det = np.linalg.det(m_up) * np.linalg.det(m_dn)
                self.ratio = new_det / old_det
                self.acc = np.random.rand() <= min(self.ratio, 1)
                if self.acc:
                    # Move accepted:
                    # Continue using the new configuration
                    old_det = new_det
                else:
                    # Move not accepted:
                    # Revert to the old configuration by updating again
                    self.config.update(i, l)
                self._debug(i, l)
        return old_det

    def warmup_loop_det(self):
        """ Runs the slow version of the LQMC warmup-loop """
        self.status = "Warmup"
        # Calculate M matrices for both spins
        m_up = self.get_m(0, sigma=+1)
        m_dn = self.get_m(0, sigma=-1)
        # Initialize the determinant for both matrices
        old_det = np.linalg.det(m_up) * np.linalg.det(m_dn)
        # Warmup-sweeps
        for _ in self.iter_sweeps(self.warm_sweeps):
            old_det = self._warmup_step_det(old_det)

    def warmup_loop_det_animated(self, plot_updates=1):
        """ Runs the slow version of the LQMC warmup-loop """
        self.status = "Warmup"
        # Calculate M matrices for both spins
        m_up = self.get_m(0, sigma=+1)
        m_dn = self.get_m(0, sigma=-1)
        # Initialize the determinant for both matrices
        old_det = np.linalg.det(m_up) * np.linalg.det(m_dn)

        stat_plot = ConfigStatPlot.empty()
        config_plot = self.config.show(False)

        # Warmup-sweeps
        for sweep in self.iter_sweeps(self.warm_sweeps):
            old_det = self._warmup_step_det(old_det)

            stat_plot.update(self.config.mean(), self.config.var())
            if sweep % plot_updates == 0:
                config_plot.update(self.config)
                stat_plot.draw()
                config_plot.draw()

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

        # Calculate M matrices for both spins
        m_up = self.get_m(0, sigma=+1)
        m_dn = self.get_m(0, sigma=-1)
        # Initialize the determinant for both matrices
        old_det = np.linalg.det(m_up) * np.linalg.det(m_dn)

        # Initialize greens functions
        gf_total_up = np.zeros((self.time_steps, self.n_sites, self.n_sites), dtype=np.float64)
        gf_total_dn = np.zeros((self.time_steps, self.n_sites, self.n_sites), dtype=np.float64)
        gf_tmp_up = np.linalg.inv(m_up)
        gf_tmp_dn = np.linalg.inv(m_dn)

        number = 0
        # Measurement-sweeps
        for _ in self.iter_sweeps(self.meas_sweeps):
            # Iterate over all time-steps, starting at the end (.math:'\beta')
            for l in reversed(range(self.time_steps)):
                # Iterate over all lattice sites
                for i in range(self.n_sites):
                    # Update Configuration by flipping spin
                    self.config.update(i, l)
                    # Compute updated M matrices (cyclic) after config-update
                    m_up = self.get_m(l, sigma=+1)
                    m_dn = self.get_m(l, sigma=-1)
                    # Compute the new determinant for both matrices for the acceptance ratio
                    new_det = np.linalg.det(m_up) * np.linalg.det(m_dn)
                    self.ratio = new_det / old_det
                    self.acc = np.random.rand() < self.ratio
                    if self.acc:
                        # Move accepted:
                        gf_tmp_up = np.linalg.inv(m_up)
                        gf_tmp_dn = np.linalg.inv(m_dn)
                        # Continue using the new configuration
                        old_det = new_det
                    else:
                        # Move not accepted:
                        # Revert to the old configuration by updating again
                        self.config.update(i, l)

                    self._debug(i, l)

                    # Add result to total results
                    gf_total_up += gf_tmp_up
                    gf_total_dn += gf_tmp_dn
                    number += 1
                # Update greens function for calculation at the next time slice
                # This is not yet needed, because the greens function is calculated
                # every time explicitly. If the fast updating without the inversion
                # of the M matrix is used, the wrapping step with the B matrices
                # needs to be done here.
                # If using this change back to the old _compute_m function, because
                # order of the B matrices does not matter then as they are used 
                # only within a determinant.
        # Return the normalized total green functions
        return np.array([gf_total_up, gf_total_dn]) / number * self.time_steps

    def warmup_loop(self):
        """ Runs the fast version of the LQMC warmup-loop """
        self.status = "Warmup"

        # Warmup-sweeps
        for _ in self.iter_sweeps(self.warm_sweeps):
            # Iterate over all time-steps, starting at the end (.math:'\beta')
            for l in reversed(range(self.time_steps)):
                # Iterate over all lattice sites
                for i in range(self.n_sites):
                    # Compute updated M matrices (cyclic)
                    m_up = self.get_m(l, sigma=+1)
                    m_dn = self.get_m(l, sigma=-1)
                    arg = 2 * self.lamb * self.config[i, l]
                    d_up = 1 + (1 - np.linalg.inv(m_up)[i, i]) * (np.exp(-arg) - 1)
                    d_dn = 1 + (1 - np.linalg.inv(m_dn)[i, i]) * (np.exp(+arg) - 1)
                    self.ratio = d_up * d_dn
                    self.acc = np.random.rand() < self.ratio
                    if self.acc:
                        self.config.update(i, l)

                    self._debug(i, l)

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
        # Compute updated M matrices (cyclic) after config-update
        m_up = self.get_m(0, sigma=+1)
        m_dn = self.get_m(0, sigma=-1)
        g_beta_up = np.linalg.inv(m_up)
        g_beta_dn = np.linalg.inv(m_dn)
        g_tmp_up, g_tmp_dn = self._compute_gf_tau(g_beta_up, g_beta_dn)

        number = 0
        # Measurement-sweeps
        for _ in self.iter_sweeps(self.meas_sweeps):
            # Iterate over all time-steps, starting at the end (.math:'\beta')
            for l in reversed(range(self.time_steps)):
                # Iterate over all lattice sites
                for i in range(self.n_sites):
                    # Compute updated M matrices (cyclic)
                    m_up = self.get_m(l, sigma=+1)
                    m_dn = self.get_m(l, sigma=-1)
                    arg = 2 * self.lamb * self.config[i, l]

                    d_up = 1 + (1 - np.linalg.inv(m_up)[i, i]) * (np.exp(-arg) - 1)
                    d_dn = 1 + (1 - np.linalg.inv(m_dn)[i, i]) * (np.exp(+arg) - 1)
                    self.ratio = d_up * d_dn
                    self.acc = np.random.rand() < self.ratio
                    if self.acc:
                        # Move accepted: Update temp greens function and update configuration
                        c_up = np.zeros(n, dtype=np.float64)
                        c_dn = np.zeros(n, dtype=np.float64)
                        c_up[i] = np.exp(-arg) - 1
                        c_dn[i] = np.exp(+arg) - 1
                        c_up = -1 * (np.exp(-arg) - 1) * g_beta_up[i, :] + c_up
                        c_dn = -1 * (np.exp(+arg) - 1) * g_beta_dn[i, :] + c_dn
                        b_up = g_beta_up[:, i] / (1. + c_up[i])
                        b_dn = g_beta_dn[:, i] / (1. + c_dn[i])

                        g_beta_up = g_beta_up - np.outer(b_up, c_up)
                        g_beta_dn = g_beta_dn - np.outer(b_dn, c_dn)
                        g_tmp_up, g_tmp_dn = self._compute_gf_tau(g_beta_up, g_beta_dn)

                        # Update Configuration
                        self.config.update(i, l)

                    self._debug(i, l)

                    # Add temp greens function to total gf after each step
                    gf_up += g_tmp_up
                    gf_dn += g_tmp_dn
                    number += 1

        # Return the normalized gfs for each spin
        return np.array([gf_up, gf_dn]) / number

    def run_lqmc(self):
        if self.det_mode:
            self.warmup_loop_det()
            gf = self.measure_loop_det()
        else:
            self.warmup_loop()
            gf = self.measure_loop()
        return gf

    def run(self):
        t0 = time.time()
        gf_tau = self.run_lqmc()
        t = time.time() - t0
        mins, secs = divmod(t, 60)
        self.logger.info(f"Total time: {int(mins):0>2}:{int(secs):0>2} min")
        return gf_tau
