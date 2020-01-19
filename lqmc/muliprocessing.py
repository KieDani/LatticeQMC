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

    def __init__(self, i, iters, pipe, model, beta, time_steps, sweeps=1000, warmup_ratio=0.2):
        multiprocessing.Process.__init__(self)
        LatticeQMC.__init__(self, model, beta, time_steps, sweeps, warmup_ratio)
        self.lock = multiprocessing.Lock()
        self.idx = i
        self.iters = iters
        self.pipe = pipe

    def loop_generator(self, sweeps):
        for sweep in range(sweeps):
            self.it += 1
            self.lock.acquire()
            self.iters[self.idx] = self.it
            self.lock.release()
            for i in range(self.model.n_sites):
                for l in range(self.config.time_steps):
                    yield sweep, i, l

    def run(self):
        # Set seed here to be sure pid is correct
        np.random.seed(self.pid)
        self.config.initialize()
        # Run LQMC
        self.warmup_loop()
        gf = self.measure_loop()
        # Send results to main thread
        self.pipe.send(gf)


class LqmcProcessManager:

    CORE_COUNT = multiprocessing.cpu_count()

    def __init__(self, cores=None):
        self.cores = multiprocessing.cpu_count() if cores is None else cores
        self.iters = multiprocessing.Array("i", self.cores)
        self.processes = list()
        self.pipes = list()
        self.sweeps = np.zeros(self.cores)

    def total_sweeps(self):
        return np.sum(self.sweeps)

    def set_cores(self, cores=None):
        self.cores = multiprocessing.cpu_count() if cores is None else cores
        self.iters = multiprocessing.Array("i", self.cores)

    def processes_alive(self):
        return [self.processes[i].is_alive() for i in range(self.cores)]

    def processes_done(self):
        return [self.iters[i] == self.sweeps[i] for i in range(self.cores)]

    def all_done(self):
        return all(self.processes_done()) or not any(self.processes_alive())

    def _init_processes(self, model, beta, time_steps, warmup_ratio=0.2):
        self.processes = list()
        self.pipes = list()
        for i in range(self.cores):
            recv_end, send_end = multiprocessing.Pipe(False)
            p = LqmcProcess(i, self.iters, send_end, model, beta, time_steps, self.sweeps[i], warmup_ratio)
            self.processes.append(p)
            self.pipes.append(recv_end)

    def init(self, model, beta, time_steps, sweeps, warmup_ratio=0.2):
        self.sweeps = np.full(self.cores, sweeps / self.cores, dtype="int")
        self.sweeps[0] += sweeps - np.sum(self.sweeps)
        self._init_processes(model, beta, time_steps, warmup_ratio)

    def init_per_core(self, model, beta, time_steps, sweeps_per_core, warmup_ratio=0.2):
        self.sweeps = np.full(self.cores, sweeps_per_core, dtype="int")
        self._init_processes(model, beta, time_steps, warmup_ratio)

    def start(self):
        # print(f"Running {self.total_sweeps()} Sweeps on {self.cores} processes ({self.sweeps})")
        print(f"Starting {self.cores} processes (Total: {self.total_sweeps()})")
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
        width = len(str(max(self.sweeps)))
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
