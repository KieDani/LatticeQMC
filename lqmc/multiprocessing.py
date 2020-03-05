# coding: utf-8
"""
Created on 16 Jan 2020

project: LatticeQMC
version: 1.0
"""
import time
import numpy as np
import itertools
import multiprocessing
from .lqmc import LatticeQMC


class LqmcProcess(LatticeQMC, multiprocessing.Process):

    INDEX = itertools.count()

    def __init__(self, i, iters, pipe, *args, **kwargs):
        multiprocessing.Process.__init__(self)
        LatticeQMC.__init__(self, *args, **kwargs)
        self.lock = multiprocessing.Lock()
        self.idx = i
        self.iters = iters
        self.pipe = pipe

    def loop_generator(self, sweeps):
        for sweep in range(sweeps):
            self.it += 1
            # self.lock.acquire()
            self.iters[self.idx] = self.it
            # self.lock.release()
            for i in range(self.model.n_sites):
                for l in range(self.config.time_steps):
                    yield sweep, i, l

    def run(self):
        # Set seed here to be sure pid is correct
        np.random.seed(self.pid)
        self.config.initialize()
        # Run LQMC
        gf = self.run_lqmc()
        # Send results to main thread
        self.pipe.send(gf)


class LqmcProcessManager:

    CORE_COUNT = multiprocessing.cpu_count()

    def __init__(self, processes=None):
        """ Initialize process manager for lqmc multiprocessing

        Parameters
        ----------
        processes: int, optional
            Number of processes to use. If value is 'None' or '0' the number of  cores of the system-cpu is used.
            If the value is negative it will be subtracted from the core number.
        """
        if processes is None:
            n_procs = multiprocessing.cpu_count()
        elif processes < 0:
            n_procs = multiprocessing.cpu_count() + processes
        else:
            n_procs = processes
        self.cores = n_procs
        self.iters = multiprocessing.Array("i", self.cores)
        self.processes = list()
        self.pipes = list()
        self.warm_sweeps = 0
        self.measure_sweeps = np.zeros(self.cores)

    def get_total_sweeps(self, i):
        return self.measure_sweeps[i] + self.warm_sweeps

    def set_cores(self, cores=None):
        self.cores = multiprocessing.cpu_count() if cores is None else cores
        self.iters = multiprocessing.Array("i", self.cores)

    def processes_alive(self):
        return [self.processes[i].is_alive() for i in range(self.cores)]

    def processes_done(self):
        return [self.iters[i] >= self.get_total_sweeps(i) for i in range(self.cores)]

    def all_done(self):
        return all(self.processes_done()) or not any(self.processes_alive())

    def init(self, model, temp, time_steps,  warm_sweeps=300, meas_sweeps=2000, det_mode=False):
        self.warm_sweeps = warm_sweeps
        self.measure_sweeps = np.full(self.cores, meas_sweeps / self.cores, dtype="int")
        self.measure_sweeps[0] += meas_sweeps - np.sum(self.measure_sweeps)
        self.processes = list()
        self.pipes = list()
        for i in range(self.cores):
            recv_end, send_end = multiprocessing.Pipe(False)
            meas_sweeps = self.measure_sweeps[i]
            p = LqmcProcess(i, self.iters, send_end, model, time_steps, self.warm_sweeps, meas_sweeps, det_mode)
            p.set_temperature(temp)
            self.processes.append(p)
            self.pipes.append(recv_end)

    def start(self):
        # print(f"Running {self.total_sweeps()} Sweeps on {self.cores} processes ({self.sweeps})")
        print(f"Starting {self.cores} processes (Total: {np.sum(self.measure_sweeps)} measurement sweeps)")
        for p in self.processes:
            p.start()

    def join(self):
        for p in self.processes:
            p.join()

    def terminate(self):
        for p in self.processes:
            p.terminate()

    def recv(self, i):
        return self.pipes[i].recv()

    def recv_all(self):
        return np.array([self.recv(i) for i in range(self.cores)])

    def run(self, sleep=0.5):
        print("Warmup:     ", str([p.warm_sweeps for p in self.processes]))
        print("Measurement:", str([p.meas_sweeps for p in self.processes]))
        print()

        delim = " | "
        width = len(str(max(self.measure_sweeps)))
        row = delim.join([f"{str(i):>{width}}" for i in range(self.cores)])
        print("Process     " + row)
        t0 = time.time()
        while not self.all_done():
            row = delim.join([f"{str(it):>{width}}" for it in self.iters])
            print(f"\r" + "Iteration   " + row, end="", flush=True)
            time.sleep(sleep)

        mins, secs = divmod(time.time() - t0, 60)
        row = delim.join([f"{str(it):>{width}}" for it in self.iters])
        print(f"\r" + "Iteration   " + row, flush=True)
        print(f"Total time: {int(mins):0>2}:{int(secs):0>2} min")
