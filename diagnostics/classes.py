import datetime

import numpy as np
import matplotlib.pyplot as plt

from .log import modlog
from .log import logged

from .errors import DataLossError

# TODO: think about rounding (data) and possible rounding errors when rounding data
# TODO: implement interpolation to match channel properties (t0, fs, len(data))
# TODO: think about displaying timeseries using datetime (UTC or LOCAL)


class TimeSerie(object):
    def __init__(self, data, t0=0, name="", fs=1):

        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self._data = data

        if isinstance(t0, datetime.datetime):
            t0 = t0.timestamp()
        self._t0 = t0

        if not isinstance(fs, int):
            fs = int(fs)
        self._fs = fs

        self.name = name

        self.mod_func = {
            "default": lambda x: x,
            "zero_negatives": lambda x: x * (x > 0),
            "correct_negatives": lambda x: x - min(x * (x < 0)),
        }
        self._reprconfig = {"threshold": 10, separator=" "}

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        self._data = value

    @property
    def t0(self):
        return self._t0

    @t0.setter
    def t0(self, value):
        assert value >= 0, "Can't have a negative value for time!"
        self._t0 = value

    @property
    def fs(self):
        return self._fs

    @fs.setter
    def fs(self, value):
        assert value > 0, "Can't have a fs <= 0!"
        self._fs = value

    @property
    def channel(self):
        return self.t0, self.fs, self.__len__()

    @channel.setter
    def channel(self, value):
        raise ValueError("Can't set channel-properties of channel!")

    @property
    def hz(self):
        return self.fs

    @hz.setter
    def hz(self, value):
        self.fs = value

    def __repr__(self):
        return "TimeSerie({}, t0={}, name={}, fs={})".format(
            np.array2string(self.data, **self._reprconfig), *map(repr, [self.t0, self.name, self.fs])
        )

    #        return "TimeSerie({}, t0={}, name={}, fs={})".format(np.array2str(self.data, threshold=self._threshold), *map(repr, [self.t0, self.name, self.fs]))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.__getitem__(index)

    def __eq__(self, other):
        if isinstance(other, TimeSerie):
            if self.channel != other.channel:
                raise ValueError(
                    "TimeSerie channels inconsistent! ({} & {})".format(
                        self.channel, other.channel
                    )
                )
            return BooleanTimeSerie(
                self.data == other.data,
                t0=self.t0,
                fs=self.fs,
                name="({} == {})".format(self.name, other.name),
            )
        else:
            return BooleanTimeSerie(self.data == other, t0=self.t0, fs=self.fs, name="")

    def __ne__(self, other):
        if isinstance(other, TimeSerie):
            if self.channel != other.channel:
                raise ValueError(
                    "TimeSerie channels inconsistent! ({} & {})".format(
                        self.channel, other.channel
                    )
                )
            return BooleanTimeSerie(
                self.data != other.data,
                t0=self.t0,
                fs=self.fs,
                name="({} != {})".format(self.name, other.name),
            )
        else:
            return BooleanTimeSerie(self.data != other, t0=self.t0, fs=self.fs, name="")

    def __lt__(self, other):
        if isinstance(other, TimeSerie):
            if self.channel != other.channel:
                raise ValueError(
                    "TimeSerie channels inconsistent! ({} & {})".format(
                        self.channel, other.channel
                    )
                )
            return BooleanTimeSerie(
                self.data < other.data,
                t0=self.t0,
                fs=self.fs,
                name="({} < {})".format(self.name, other.name),
            )
        else:
            return BooleanTimeSerie(self.data < other, t0=self.t0, fs=self.fs, name="")

    def __gt__(self, other):
        if isinstance(other, TimeSerie):
            if self.channel != other.channel:
                raise ValueError(
                    "TimeSerie channels inconsistent! ({} & {})".format(
                        self.channel, other.channel
                    )
                )
            return BooleanTimeSerie(
                self.data > other.data,
                t0=self.t0,
                fs=self.fs,
                name="({} > {})".format(self.name, other.name),
            )
        else:
            return BooleanTimeSerie(self.data > other, t0=self.t0, fs=self.fs, name="")

    def __le__(self, other):
        if isinstance(other, TimeSerie):
            if self.channel != other.channel:
                raise ValueError(
                    "TimeSerie channels inconsistent! ({} & {})".format(
                        self.channel, other.channel
                    )
                )
            return BooleanTimeSerie(
                self.data <= other.data,
                t0=self.t0,
                fs=self.fs,
                name="({} <= {})".format(self.name, other.name),
            )
        else:
            return BooleanTimeSerie(self.data <= other, t0=self.t0, fs=self.fs, name="")

    def __ge__(self, other):
        if isinstance(other, TimeSerie):
            if self.channel != other.channel:
                raise ValueError(
                    "TimeSerie channels inconsistent! ({} & {})".format(
                        self.channel, other.channel
                    )
                )
            return BooleanTimeSerie(
                self.data >= other.data,
                t0=self.t0,
                fs=self.fs,
                name="({} >= {})".format(self.name, other.name),
            )
        else:
            return BooleanTimeSerie(self.data >= other, t0=self.t0, fs=self.fs, name="")

    def __add__(self, other):
        if isinstance(other, TimeSerie):
            if self.channel != other.channel:
                raise ValueError(
                    "TimeSerie channels inconsistent! ({} & {})".format(
                        self.channel, other.channel
                    )
                )
            return TimeSerie(
                self.data + other.data,
                t0=self.t0,
                fs=self.fs,
                name="({} + {})".format(self.name, other.name),
            )
        else:
            return TimeSerie(self.data + other, t0=self.t0, fs=self.fs, name="")

    def __radd__(self, other):
        return TimeSerie(other + self.data, t0=self.t0, fs=self.fs, name="")

    def __sub__(self, other):
        if isinstance(other, TimeSerie):
            if self.channel != other.channel:
                raise ValueError(
                    "TimeSerie channels inconsistent! ({} & {})".format(
                        self.channel, other.channel
                    )
                )
            return TimeSerie(
                self.data - other.data,
                t0=self.t0,
                fs=self.fs,
                name="({} - {})".format(self.name, other.name),
            )
        else:
            return TimeSerie(self.data - other, t0=self.t0, fs=self.fs, name="")

    def __rsub__(self, other):
        return TimeSerie(other - self.data, t0=self.t0, fs=self.fs, name="")

    def __and__(self, other):
        if self.channel != other.channel:
            raise ValueError(
                "TimeSerie channels inconsistent! ({} & {})".format(
                    other.channel, self.channel
                )
            )
        if not (self.is_bool() and other.is_bool()):
            raise ValueError(
                "Can't perform bitwise operation on non-boolean TimeSeries!"
            )
        return BooleanTimeSerie(
            self.data & other.data,
            t0=self.t0,
            name="({} & {})".format(self.name, other.name),
            fs=self.fs,
        )

    def __or__(self, other):
        if self.channel != other.channel:
            raise ValueError(
                "TimeSerie channels inconsistent! ({} & {})".format(
                    other.channel, self.channel
                )
            )
        if not (self.is_bool() and other.is_bool()):
            raise ValueError(
                "Can't perform bitwise operation on non-boolean TimeSeries!"
            )
        return BooleanTimeSerie(
            self.data | other.data,
            t0=self.t0,
            name="({} | {})".format(self.name, other.name),
            fs=self.fs,
        )

    def __xor__(self, other):
        if self.channel != other.channel:
            raise ValueError(
                "TimeSerie channels inconsistent! ({} & {})".format(
                    other.channel, self.channel
                )
            )
        if not (self.is_bool() and other.is_bool()):
            raise ValueError(
                "Can't perform bitwise operation on non-boolean TimeSeries!"
            )
        return BooleanTimeSerie(
            self.data ^ other.data,
            t0=self.t0,
            name="({} ^ {})".format(self.name, other.name),
            fs=self.fs,
        )

    def __invert__(self):
        if not self.is_bool():
            raise ValueError(
                "Can't perform bitwise operation on non-boolean TimeSerie!"
            )
        return BooleanTimeSerie(
            ~self.data, t0=self.t0, name="(~{})".format(self.name), fs=self.fs
        )

    @classmethod
    def empty(cls, t0, te, fs, name="", inclusive=False):
        if isinstance(t0, datetime.datetime):
            t0 = t0.timestamp()
        t0 = round(t0, len(str(fs)) - 1)
        if isinstance(te, datetime.datetime):
            te = te.timestamp()
        te = round(te, len(str(fs)) - 1)
        k = int(((te - t0) * fs))
        if inclusive:
            k += 1
        data = np.zeros(k)
        return cls(data, t0=t0, fs=fs, name=name)

    @logged()
    def reset_t0(self):
        self.t0 = 0

    @logged()
    def round_t0(self):
        self.t0 = round(self.t0, len(str(self.fs)) - 1)

    def _t(self):
        return self.t0 + (np.arange(self.__len__()) / self.fs)

    @property
    def t(self):
        return self._t()

    @property
    def te(self):  # TODO: optimize this function
        return self.t[-1]

    def iter(self):
        for t, v in zip(self.t, self.data):
            yield t, v

    def plot(self, **kwargs):
        show = kwargs.pop("show", True)
        plt.plot(self._t(), self.data, label=self.name, **kwargs)
        if show:
            plt.show()  # TODO: this freezes the terminal

    def to_channel(self, c):
        if self.fs != c.fs:
            raise ValueError(
                "can't modify channel to a different fs! please interpolate first"
            )
        if c.t0 > self.t0:
            raise DataLossError(
                "channel data window does not fully overlap! (c.t0 > self.t0)"
            )
        if c.te < self.te:
            raise DataLossError(
                "channel data window does not fully overlap! (c.te < self.te)"
            )
        # TODO: check that self.t0 and c_t0 can be exact t values for given fs
        self.t0 - c.t0 * fs % 1
        # TODO: modify data

    @logged()
    def modify(self, method, inplace=False):
        if inplace:
            self.data = self.mod_func[method](self.data)
        else:
            from copy import deepcopy

            b = deepcopy(self)
            b.data = self.mod_func[method](self.data)
            return b

    @logged()
    def interpolate(self, t_new, inplace=False):
        data = np.interp(t_new, self.t, self.data)
        fs = np.mean(1 / np.diff(t_new))

        if self.is_bool():
            data = data.astype(np.bool)
            classtype = BooleanTimeSerie
        else:
            classtype = TimeSerie

        if inplace:
            self.data = data
            self.fs = fs
            self.t0 = t_new[0]
        else:
            return classtype(data, fs=fs, t0=t_new[0], name=self.name)

    def is_bool(self):
        return self.data.dtype == np.bool

    @logged()
    def to_bool(self, inplace=False):

        data = self.data.astype(np.bool)
        if inplace:
            self.data = data
        else:
            return BooleanTimeSerie(data, t0=self.t0, fs=self.fs, name=self.name)

    @logged()
    def to_events(self):
        return list(self.events())
        # pass # TODO: create events (state changes) from data

    @logged()
    def events(self):
        state = None
        for t, v in self.iter():
            if v != state:
                state = v
                e = Event(value=v, t=t, name=self.name)
                yield e

    @logged()
    def to_statechangearray(self):
        events = self.to_events()
        return StateChangeArray.from_events(events)
        # pass # TODO: create statechangearray from data

    @logged()
    def from_events(self, events):
        pass  # TODO: implement creation of TimeSeries from event (not boolean, since the validity of single events are not always known) (also return 'validity' BooleanTimeSerie!)


class BooleanTimeSerie(TimeSerie):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)  # TODO: create check for boolean values data
        if self.data.dtype != np.bool:
            raise ValueError("data is not of dtype 'bool'")

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, value):
        if not isinstance(value, np.ndarray):
            value = np.array(value)
        if value.dtype != np.bool:
            raise ValueError("data is not of dtype 'bool'")
        self._data = value

    def __repr__(self):
        return "BooleanTimeSerie({}, t0={}, name={}, fs={})".format(
            self.data, *map(repr, [self.t0, self.name, self.fs])
        )

    @logged()
    def from_reports(self, reports):
        pass  # TODO: implement creation of BooleanTimeSerie from (multiple) reports. Might use report.to_timeserie() method and then combines multiple timeseries


class StateChangeArray(object):
    def __init__(self, data, t, name=""):

        if len(data) != len(t):
            raise ValueError("data & t should be of the same length")
        if not isinstance(data, np.ndarray):
            self.data = np.array(data)
        else:
            self.data = data
        if not isinstance(t, np.ndarray):
            self.t = np.array(t)
        else:
            self.t = t
        self.name = name

    def __repr__(self):
        return "StateChangeArray({}, t={}, name={})".format(
            self.data, self.t, repr(self.name)
        )

    def iter(self):
        for t, v in zip(self.t, self.data):
            yield t, v

    @classmethod
    def from_events(cls, events):
        data = []
        t = []
        for e in events:
            data.append(e.value)
            t.append(e.t)
        return cls(data=data, t=t, name=e.name)

    def events(self):
        for t, v in self.iter():
            e = Event(data=v, t=t, name=self.name)
            yield e

    def to_events(self):
        return list(self.events())

    def duration(self):
        return np.diff(self.t)

    def to_timeseries(self, fs, method="default", tol=1e-4, tail=0):
        t0 = self.t[0]
        name = self.name
        if method == "default":
            t0 = self.t[0]
            name = self.name
            duration = self.duration()
            if any(((duration * fs) % 1) > tol):
                raise ValueError("chosen fs does not match state change timings!")
            samples = (fs * duration).astype(np.int)
            data = np.array([])
            for s, d in zip(samples, self.data[:-1]):
                data = np.append(data, np.repeat(d, s))
            data = np.append(data, np.repeat(self.data[-1], (tail * fs) or 1))
        elif method == "interpolate":
            t_new = np.arange(t0, (self.t[-1] + 1 / fs), 1 / fs)
            data = np.interp(t_new, self.t, self.data)
        else:
            raise NameError("unknown method '{}'".format(method))
        return TimeSerie(data, t0=t0, fs=fs, name=name)

    def to_reports(self):
        if not self.is_bool():
            ValueError("Can't create reports from non-boolean statechangearray!")

    def is_bool(self):
        return self.data.dtype == np.bool


class Report(object):
    def __init__(self, t0, te, name=""):

        self.t0 = t0
        self.te = te
        self.name = name

    @logged()
    def to_events(self):
        event_t0 = Event(1, t=t0, name=name)
        event_te = Event(0, t=te, name=name)
        return (event_t0, event_te)

    @logged()
    def to_timeserie(self, fs=1, window=1):
        t0 = (self.t0 - (window / fs),)
        window_data = np.zeros(window)
        k = round((te - t0) * fs)
        data = np.ones(k)
        data = np.append(window_data, data, window_data)
        return TimeSerie(data, t0=t0, fs=fs)


class Event(object):
    def __init__(self, value, t=0, name="", validity=1):

        self.value = value
        self.t = t
        self.name = name
        self.validity = validity

    def state(self):
        if validity:
            return self.value
        else:
            return "?"

    def __repr__(self):
        return "Event({}, t={}, name={})".format(
            *map(repr, [self.value, self.t, self.name])
        )

    def __print__(self):
        return "<{}\t{}: {}>".format(self.t, self.name, self.state())
