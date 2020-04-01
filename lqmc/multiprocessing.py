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
        LatticeQMC.__init__(self, *args, **kwargs, log_lvl=None)
        self.idx = i
        self.iters = iters
        self.pipe = pipe

    def is_done(self):
        return not self.is_alive()

    def iter_sweeps(self, n, console_updates=0):
        for it in range(n):
            self.it = it
            self.iters[self.idx] += 1
            yield it

    def run(self):
        # Set seed here to be sure pid is correct
        np.random.seed(self.pid)
        self.config.initialize()
        # Run LQMC
        gf = self.run_lqmc()
        # Send results to main thread
        self.pipe.send(gf)


class ProcessManager:

    def __init__(self, procs=None, **default_kwargs):
        """ Manager for handling multiple `LqmcProcess`-instances.

        Parameters
        ----------
        procs: int, optional
            Number of processes to use. If value is 'None' or '0' the number of  cores of the system-cpu is used.
            If the value is negative it will be subtracted from the core number.
        **default_kwargs
            The default keyword arguments for the `LatticeQMC`-object.
            These parameters are static and will be passed to each process.
        """
        if procs is None:
            n_procs = multiprocessing.cpu_count()
        elif procs < 0:
            n_procs = multiprocessing.cpu_count() + procs
        else:
            n_procs = procs
        self.max_procs = n_procs

        self.processes = list()
        self.lock = multiprocessing.Lock()

        self.iters = multiprocessing.Array("i", self.max_procs)
        self.idx = 0
        self.total = 0
        self.result = None
        self.default_kwargs = default_kwargs
        self.var_kwargs = dict()
        self.t0 = 0

    def set_jobs(self, **kwargs):
        """ Sets the jobs of the process manager.

        Parameters
        ----------
        **kwargs
            The variable arguments for the `LatticeQMC`-object. which will be added
            to the default arguments. Each value must be an array-like object
            contain the same number of parameters, one for each job.
        """
        key = list(kwargs.keys())[0]
        varlist = kwargs[key]
        total = len(varlist)
        for key, item in kwargs.items():
            varlist = kwargs[key]
            if len(varlist) != total:
                raise ValueError('All variable lists must have the same length!')

        self.var_kwargs = kwargs
        self.idx = 0
        self.total = total
        self.result = [None for _ in range(total)]
        self.iters = multiprocessing.Array("i", total)

    def __str__(self):
        string = self.__class__.__name__
        var_keys = list(self.var_kwargs.keys())
        return string + f'(Vars: {var_keys}, Jobs: {self.total}, Processes: {self.max_procs})'

    @property
    def all_done(self):
        return self.idx == self.total and not self.processes

    @property
    def jobs_running(self):
        return len(self.processes)

    @property
    def jobs_pending(self):
        return self.total - self.idx

    @property
    def jobs_done(self):
        return self.idx - self.jobs_running

    @property
    def free_processes(self):
        return self.max_procs - len(self.processes)

    def get_result(self):
        return np.array(self.result)

    def get_progress(self):
        # Get progress of active processes
        prog, num = 0, 0
        for p, _ in self.processes:
            total = p.meas_sweeps + p.warm_sweeps
            prog += self.iters[p.idx] / total
            num += 1

        # Add Pending and done Processes
        prog += 1.0 * self.jobs_done
        return prog / self.total

    def join(self):
        for p in self.processes:
            p.join()

    def terminate(self):
        for p in self.processes:
            p.terminate()

    def _start_process(self):
        # Get next index
        self.lock.acquire()
        idx = self.idx
        self.idx += 1
        self.lock.release()

        # Create new process
        kwargs = self.default_kwargs.copy()
        for key, item in self.var_kwargs.items():
            kwargs.update({key: item[idx]})
        pipe, send_pipe = multiprocessing.Pipe(False)
        p = LqmcProcess(idx, self.iters, send_pipe, **kwargs)
        self.processes.append((p, pipe))

        # Start the new process
        p.start()

    def handle_processes(self):
        # Check for finished processes and store the data in result
        for item in self.processes:
            p, pipe = item
            if p.is_done():
                idx = p.idx
                data = pipe.recv()
                self.result[idx] = np.array(data)
                self.lock.acquire()
                self.processes.remove(item)
                self.lock.release()

        # Check if jobs are left and start new processes
        while self.jobs_pending and self.free_processes:
            self._start_process()

    def start_str(self, *args, **kwargs):
        return f'Starting {self}'

    def start(self):
        startstr = self.start_str()
        if startstr:
            print(startstr)
        return []

    def update_str(self, *args, **kwargs):
        info = f'Alive: {self.jobs_running}, Done: {self.jobs_done}, Pending: {self.jobs_pending}'
        string = f'Progress: {100 * self.get_progress():5.1f}% ({info})'
        return string

    def update(self, *args):
        print('\r' + self.update_str(), end='', flush=True)
        return args

    def end(self, *args):
        print()
        mins, secs = divmod(time.time() - self.t0, 60)
        print(f"Total time: {int(mins):0>2}:{int(secs):0>2} min")

    def run(self, sleep=0.5):
        self.t0 = time.time()
        args = self.start()
        while not self.all_done:
            self.handle_processes()
            time.sleep(sleep)
            args = self.update(*args)
        self.end(args)


# =========================================================================


class SerialProcessManager(ProcessManager):

    CORE_COUNT = multiprocessing.cpu_count()

    def __init__(self, model, time_steps, warmup=300, sweeps=2000, det_mode=False, procs=None):
        super().__init__(procs, model=model, time_steps=time_steps,
                         warmup=warmup, sweeps=sweeps, det_mode=det_mode)

    def set_jobs(self, betas):
        super().set_jobs(beta=betas)

    def start_str(self, *args, **kwargs):
        string = super().start_str()
        string += f"\nWarmup     ={self.default_kwargs['warmup']}"
        string += f"\nMeasurement={self.default_kwargs['sweeps']}"
        return string


class ParallelProcessManager(ProcessManager):

    CORE_COUNT = multiprocessing.cpu_count()

    def __init__(self, model, beta, time_steps, warmup=300, det_mode=False, procs=None):
        super().__init__(procs, model=model, beta=beta, time_steps=time_steps,
                         warmup=warmup, det_mode=det_mode)

    def set_jobs(self, sweeps):
        sweeplist = np.full(self.max_procs, sweeps / self.max_procs, dtype="int")
        sweeplist[0] += sweeps - np.sum(sweeplist)
        super().set_jobs(sweeps=sweeplist)

    def get_result(self):
        gf_data = super().get_result()
        return np.sum(gf_data, axis=0) / self.max_procs

    @staticmethod
    def _frmt_items(items, delim, width):
        return delim.join([f"{item:>{width}}" for item in items])

    def start(self):
        print(self.start_str())
        delim = " | "
        sweeplist = list(self.var_kwargs['sweeps'])
        warmuplist = [self.default_kwargs['warmup'] for _ in range(self.max_procs)]
        width = len(str(max(sweeplist)))

        print(f"Warmup:      [{str(self._frmt_items(warmuplist, ',  ', width))}]")
        print(f"Measurement: [{str(self._frmt_items(sweeplist, ',  ', width))}]")
        print("Process       " + self._frmt_items(range(self.max_procs), delim, width))
        return [delim, width]

    def update(self, delim, width):
        row = self._frmt_items(self.iters, delim, width)
        row += f'   Progress: {100 * self.get_progress():.1f}%'
        print(f"\r" + "Iteration     " + row, end="", flush=True)
        return delim, width
