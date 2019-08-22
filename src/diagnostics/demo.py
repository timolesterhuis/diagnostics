import datetime as dt
import random

from .classes import Report


class TimeGenerator(object):
    min = 0
    max = None
    mean = 0
    stdev = 1

    def __init__(self):
        pass

    def run(self):
        value = random.gauss(self.mean, self.stdev)
        if self.min is not None:
            if value < self.min:
                return self.min
        if self.max is not None:
            if value > self.max:
                return self.max
        else:
            return value


class CodeGenerator(object):
    def __init__(self, code_mu, code_sigma, pause_mu, pause_sigma, name=""):

        self.code_mu = code_mu
        self.code_sigma = code_sigma
        self.pause_mu = pause_mu
        self.pause_sigma = pause_sigma
        self.name = name

        self.code_gen = TimeGenerator()
        self.pause_gen = TimeGenerator()

        self.create()

    def create(self):
        self.code_gen.min = 1
        self.code_gen.mean = self.code_mu
        self.code_gen.stdev = self.code_sigma

        self.pause_gen.min = 1
        self.pause_gen.mean = self.pause_mu
        self.pause_gen.stdev = self.pause_sigma

    def run_once(self):
        return self.code_gen.run(), self.pause_gen.run()

    def run_n(self, n):
        return [self.run_once() for _ in range(n)]

    def run_for_t(self, t):
        total = 0
        array = []
        while total < t:
            code, pause = self.run_once()
            total += code + pause
            array.append((code, pause))
        return array

    def run_as_event(self, start, start_mu=0):
        code, pause = self.run_once()
        if start_mu:
            offset = random.gauss(0, start_mu)
            code_start = start + dt.timedelta(seconds=offset)
        else:
            code_start = start

        code_end = start + dt.timedelta(seconds=code)
        pause_start = code_end
        pause_end = pause_start + dt.timedelta(seconds=pause)
        return (code_start, code_end), (pause_start, pause_end)
        pass

    def run_as_events_n(self, start, n):
        array = []
        for _ in range(n):
            code, pause = self.run_as_event(start)
            array.append(code)
            start = pause[1]
        return array

    def run_as_reports_n(self, *args, **kwargs):
        array = self.run_as_events_n(*args, **kwargs)
        reports = [Report(t0=s, te=e, name=self.name) for s, e in array]
        return reports

    def run_as_events_for_t(self, start, t, start_mu=0):
        array = []
        if start_mu:
            offset = random.gauss(0, start_mu)
            start += dt.timedelta(seconds=offset)
        total = start + dt.timedelta(seconds=t)
        while start < total:
            code, pause = self.run_as_event(start)
            array.append(code)
            start = pause[1]
        return array

    def run_as_reports_for_t(self, *args, **kwargs):
        array = self.run_as_events_for_t(*args, **kwargs)
        reports = [Report(t0=s, te=e, name=self.name) for s, e in array]
        return reports
