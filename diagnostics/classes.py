import datetime
import pytz

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

import operator as op

from .log import logged
from .functions import plot

from .errors import DataLossError

# TODO: think about rounding (data) and possible rounding errors when rounding data
# TODO: implement interpolation to match channel properties (t0, fs, len(data))
# TODO: think about displaying timeseries using datetime (UTC or LOCAL)


OPS = {">": op.gt, ">=": op.ge, "<": op.lt, "<=": op.le}


class TimeSerie(object):

    # TODO: TimeSerie.to_reports
    # TODO: TimeSerie.from_statechangearray
    # TODO: TimeSerie.from_reports
    # TODO: TimeSerie.from_events

    def __init__(self, data, t0=0, name="", fs=1):
        """

        :param data:
        :param t0:
        :param name:
        :param fs:
        """

        if not isinstance(data, np.ndarray):
            data = np.array(data)
        self._data = data

        if isinstance(t0, datetime.datetime):
            if t0.tzinfo is None:
                t0 = t0.replace(tzinfo=pytz.UTC)
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
        self._reprconfig = {"threshold": 10, "separator": ", "}
        self._type = "TimeSerie"
        return

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
            np.array2string(self.data, **self._reprconfig),
            *map(repr, [self.t0, self.name, self.fs])
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return self.data.__getitem__(index)

    def _check_channel_other(self, other):
        if self.channel != other.channel:
            raise ValueError(
                "TimeSerie channels inconsistent! ({} & {})".format(
                    self.channel, other.channel
                )
            )

    def __eq__(self, other):
        if isinstance(other, TimeSerie):
            self._check_channel_other(other)
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
            self._check_channel_other(other)
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
            self._check_channel_other(other)
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
            self._check_channel_other(other)
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
            self._check_channel_other(other)
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
            self._check_channel_other(other)
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
            self._check_channel_other(other)
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
            self._check_channel_other(other)
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
        self._check_channel_other(other)
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
        self._check_channel_other(other)
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
        self._check_channel_other(other)
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
        """

        :param t0:
        :param te:
        :param fs:
        :param name:
        :param inclusive:
        :return:
        """

        if isinstance(t0, datetime.datetime):
            if t0.tzinfo is None:
                t0 = t0.replace(tzinfo=pytz.UTC)
            t0 = t0.timestamp()
        if isinstance(te, datetime.datetime):
            if te.tzinfo is None:
                te = te.replace(tzinfo=pytz.UTC)
            te = te.timestamp()
        k = int(np.ceil((te - t0) * fs))
        if inclusive:
            k += 1
        data = np.zeros(k)
        return cls(data, t0=t0, fs=fs, name=name)

    def at(self, t):
        """

        :param t:
        :return:
        """
        return self.data[np.where(self.t == t)]

    def where(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        :return:
        """
        return self.data[np.where(*args, **kwargs)]

    @logged()
    def reset_t0(self):
        """

        :return:
        """
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

    @property
    def dt(self):
        return np.array(
            [
                datetime.datetime.utcfromtimestamp(t_) for t_ in self._t()
            ],  # THINKOF: should this always be UTC?
            dtype="datetime64",
        )

    @property
    def dt0(self):
        return datetime.datetime.utcfromtimestamp(
            self.t0
        )  # THINKOF: should this always be UTC?

    @property
    def dte(self):
        return datetime.datetime.utcfromtimestamp(
            self.te
        )  # THINKOF: should this always be UTC?

    def iter(self):
        """

        :return:
        """

        for t, v in zip(self.t, self.data):
            yield t, v

    def line(self, **kwargs):
        defaults = dict(label=self.name, as_dt=False)
        defaults.update(kwargs)

        x = self._x(as_dt=defaults.pop("as_dt"))
        y = self._y()
        return Line2D(x, y, **defaults)

    def _x(self, as_dt=False):
        if as_dt:
            return self.dt
        else:
            return self.t

    def _y(self):
        return self.data

    def plot(self, *args, **kwargs):
        """

        :param kwargs:
        :return:
        """
        return plot(self, *args, **kwargs)

    def to_channel(self, c):
        """

        :param c:
        :return:
        """

        if self.fs != c.fs:
            raise ValueError(
                "can't modify channel to a different fs! please interpolate first"
            )
        t0_diff = self.t0 - c.t0
        te_diff = c.te - self.te

        if t0_diff < 0:
            raise DataLossError(
                "channel data window does not fully overlap! (c.t0 > self.t0)"
            )
        if te_diff < 0:
            raise DataLossError(
                "channel data window does not fully overlap! (c.te < self.te)"
            )

        if (t0_diff * c.fs) % 1 > 0:
            raise ValueError("Cant reach both t0 values using new fs")

        pre_data = np.zeros(int(t0_diff * c.fs))
        post_data = np.zeros(int(te_diff * c.fs))
        self.data = np.append(pre_data, np.append(self.data, post_data))
        self.t0 = c.t0

    @logged()
    def modify(self, method, inplace=False):
        """

        :param method:
        :param inplace:
        :return:
        """

        if inplace:
            self.data = self.mod_func[method](self.data)
        else:
            from copy import deepcopy

            b = deepcopy(self)
            b.data = self.mod_func[method](self.data)
            return b

    @logged()
    def interpolate(self, t_new, inplace=False):
        """

        :param t_new:
        :param inplace:
        :return:
        """

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
        """

        :return:
        """

        return self.data.dtype == np.bool

    @logged()
    def to_bool(self, inplace=False):
        """

        :param inplace:
        :return:
        """

        data = self.data.astype(np.bool)
        if inplace:
            self.data = data
        else:
            return BooleanTimeSerie(data, t0=self.t0, fs=self.fs, name=self.name)

    @logged()
    def to_events(self):
        """

        :return:
        """

        return list(self.events())

    @logged()
    def events(self):
        """

        :return:
        """

        state = None
        for t, v in self.iter():
            if v != state:
                state = v
                e = Event(value=v, t=t, name=self.name)
                yield e

    @logged()
    def to_statechangearray(self):
        """

        :return:
        """

        events = self.to_events()
        return StateChangeArray.from_events(events)

    def to_reports(self):
        """

        :return:
        """

        array = self.to_statechangearray()
        return array.to_reports()

    @logged()
    def from_events(self, events):
        """

        :param events:
        :return:
        """

        pass  # TODO: implement creation of TimeSeries from event (not boolean, since the validity of single events are not always known) (also return 'validity' BooleanTimeSerie!)


class BooleanTimeSerie(TimeSerie):
    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """

        super().__init__(*args, **kwargs)  # TODO: create check for boolean values data
        if self.data.dtype != np.bool:
            raise ValueError("data is not of dtype 'bool'")

        self._type = "BooleanTimeSerie"

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
            np.array2string(self.data, **self._reprconfig),
            *map(repr, [self.t0, self.name, self.fs])
        )

    @logged()
    def from_reports(self, reports):
        """

        :param reports:
        :return:
        """

        pass  # TODO: implement creation of BooleanTimeSerie from (multiple) reports. Might use report.to_timeserie() method and then combines multiple timeseries


class StateChangeArray(object):
    _reprconfig = {"threshold": 10, "separator": ", "}

    def __init__(self, data, t, name="", shrink=False):
        """

        :param data:
        :param t:
        :param name:
        :param shrink:
        """

        if len(data) != len(t):
            raise ValueError("data & t should be of the same length")

        if not isinstance(data, np.ndarray):
            data = np.array(data)

        # special case when no data is entered
        if len(t) == 0:
            t = np.array(t)
            self.data = data
            self.t = t
            self.name = name
            return

        if hasattr(t, "dtype"):
            # for ss unit, use:
            if t.dtype == np.dtype("<M8[ns]"):
                t = t.astype("int64") / 1e9
            # for us unit, use:
            elif t.dtype == np.dtype("<M8[us]"):
                t = t.astype("int64") / 1e6
                # for ms unit, use:
            elif t.dtype == np.dtype("<M8[ms]"):
                t = t.astype("int64") / 1e3
            else:
                t = t.astype("int64")

        if isinstance(t[0], datetime.datetime):
            if t[0].tzinfo is None:
                t = [ti.replace(tzinfo=pytz.UTC).timestamp() for ti in t]
            else:
                t = [ti.timestamp() for ti in t]

        if not isinstance(t, np.ndarray):  # TODO: implement t being a datetimearray
            t = np.array(t)

        if any(data[1:] == data[:-1]):
            if not shrink:
                raise ValueError("There was a jump where no state-change happend!")
            else:
                state = None
                new_t = []
                new_d = []
                for t, d in zip(t, data):
                    if state != d:
                        state = d
                        new_t.append(t)
                        new_d.append(d)
                t = np.array(new_t)
                data = np.array(new_d)

        if any(np.diff(t) < 0):
            raise ValueError("t should be chronilogical!")

        self.data = data
        self.t = t
        self.name = name

        self._type = "StateChangeArray"

        return

    def __repr__(self):
        return "StateChangeArray({}, t={}, name={})".format(
            np.array2string(self.data, **self._reprconfig),
            np.array2string(self.t, **self._reprconfig),
            repr(self.name),
        )

    @property
    def dt(self):
        return np.array(
            [datetime.datetime.utcfromtimestamp(t_) for t_ in self.t],
            dtype="datetime64",
        )

    def __len__(self):
        return self.data.__len__()

    def iter(self):
        """

        :return:
        """

        for t, v in zip(self.t, self.data):
            yield t, v

    @classmethod
    def from_events(cls, events):
        """

        :param events:
        :return:
        """

        data = []
        t = []
        for e in events:
            if t:
                if e.t < t[-1]:
                    raise ValueError("Events are not ordered chronically!")
            t.append(e.t)
            data.append(e.value)
        return cls(
            data=data, t=t, name=e.name
        )  # THINKOF: I do not check for consistency in e.name

    @classmethod
    def from_reports(cls, reports, on_error="fail"):
        """

        :param reports:
        :param on_error:
        :return:
        """

        data = []
        t = []
        for r in reports:
            t0 = r.t0
            te = r.te
            if te < t0:
                raise ValueError("te is before t0 for report!")
            if t:
                if not t0 > t[-1]:
                    if on_error == "fail":
                        raise ValueError("Reports are not ordered chronically!")
                    elif on_error == "ignore":
                        continue
                    elif on_error == "extend":
                        t.pop(-1)
                        data.pop(-1)
                        t.append(te)
                        data.append(False)
                        continue
            t.append(t0)
            data.append(True)
            t.append(te)
            data.append(False)

        return cls(
            data=data, t=t, name=r.name
        )  # THINKOF: I do not check for consistency in r.name

    def events(self):
        """

        :return:
        """

        for t, v in self.iter():
            e = Event(value=v, t=t, name=self.name)
            yield e

    def to_events(self):
        """

        :return:
        """

        return list(self.events())

    def duration(self):
        """

        :return:
        """

        return np.append(np.diff(self.t), 0)

    def timerule(self, duration, operator=">=", when=True, inplace=False):
        if not self.is_bool():
            raise ValueError("Can't perform timerule on non-boolean StateChangeArray!")
        data = self.data
        t = self.t
        diff = self.duration()
        mask = (data != when) | (
            (data == when) & (OPS[operator](diff, duration))
        )  # TODO: fix operator
        if inplace:
            data = data[mask]
            t = t[mask]
            state = None
            new_t = []
            new_d = []
            for t, d in zip(t, data):
                if state != d:
                    state = d
                    new_t.append(t)
                    new_d.append(d)
            self.t = np.array(new_t)
            self.data = np.array(new_d)
        else:
            return BooleanStateChangeArray(
                data[mask], t=t[mask], name=self.name, shrink=True
            )

    def to_timeseries(
        self, fs, method="default", tol=1e-4, tail=0
    ):  # TODO: fix ! end is not exlusive <-- am i sure of this?
        """

        :param fs:
        :param method:
        :param tol:
        :param tail:
        :return:
        """

        t0 = self.t[0]
        name = self.name
        if method == "default":  # TODO: do I want tail or window as a parameter?
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

    @classmethod
    def from_timeserie(cls, timeserie):
        """

        :param timeserie:
        :return:
        """

        return timeserie.to_statechangearray()

    def reports(self):
        """

        :return:
        """

        if not self.is_bool():
            raise ValueError("Can't create reports from non-boolean statechangearray!")
        gen = self.iter()
        if self.data[0] == False:
            _ = next(gen)
        while True:
            try:
                t0, _ = next(gen)
                te, _ = next(gen)
                yield Report(t0, te, name=self.name)
            except StopIteration:
                break

    def to_reports(self):
        """

        :return:
        """

        return list(self.reports())

    def is_bool(self):
        """

        :return:
        """

        return self.data.dtype == np.bool

    def to_bool(self, inplace=False):
        """

        :param inplace:
        :return:
        """

        data = self.data.astype(np.bool)
        if inplace:
            self.data = data
        else:
            return BooleanStateChangeArray(data, t=self.t, name=self.name)

    def at(self, t):
        """

        :param t:
        :return:
        """

        return self.data[np.where(self.t == t)]

    def where(self, statement):
        """

        :param statement:
        :return:
        """

        return self.data[np.where(statement)]

    def state(self, t=None):
        """

        :param t:
        :return:
        """

        if t:
            if isinstance(t, datetime.datetime):
                if t.tzinfo is None:
                    t = t.replace(tzinfo=pytz.UTC)
                t = t.timestamp()
            try:
                s = self.data[np.where(self.t <= t)[0][-1]]
            except IndexError:
                s = None
        else:
            s = self.data[-1]
        return s

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def _combine(self, other, null=None):
        left = self.to_events()
        right = other.to_events()
        left_t = [e.t for e in left]
        right_t = [e.t for e in right]
        times = sorted(left_t + right_t)
        state = (null, null)
        data = dict()
        for t in times:
            if left:
                if t == left[0].t:
                    e = left.pop(0)
                    state = (e.value, state[1])
            if right:
                if t == right[0].t:
                    e = right.pop(0)
                    state = (state[0], e.value)
            data[t] = state
        return data

    def __lt__(self, other):
        if isinstance(other, StateChangeArray):
            states = self._combine(other)
            data = []
            new_t = []
            for t, state in states.items():
                if None in state:
                    continue
                data.append(state[0] < state[1])
                new_t.append(t)
            return BooleanStateChangeArray(
                data, t=new_t, name="{} < {}".format(self.name, other.name), shrink=True
            )
        else:
            return BooleanStateChangeArray(
                self.data < other, t=self.t, name="", shrink=True
            )

    def __gt__(self, other):
        if isinstance(other, StateChangeArray):
            states = self._combine(other)
            data = []
            new_t = []
            for t, state in states.items():
                if None in state:
                    continue
                data.append(state[0] > state[1])
                new_t.append(t)
            return BooleanStateChangeArray(
                data, t=new_t, name="{} > {}".format(self.name, other.name), shrink=True
            )
        else:
            return BooleanStateChangeArray(
                self.data > other, t=self.t, name="", shrink=True
            )

    def __le__(self, other):
        if isinstance(other, StateChangeArray):
            states = self._combine(other)
            data = []
            new_t = []
            for t, state in states.items():
                if None in state:
                    continue
                data.append(state[0] <= state[1])
                new_t.append(t)
            return BooleanStateChangeArray(
                data,
                t=new_t,
                name="{} <= {}".format(self.name, other.name),
                shrink=True,
            )
        else:
            return BooleanStateChangeArray(
                self.data <= other, t=self.t, name="", shrink=True
            )

    def __ge__(self, other):
        if isinstance(other, StateChangeArray):
            states = self._combine(other)
            data = []
            new_t = []
            for t, state in states.items():
                if None in state:
                    continue
                data.append(state[0] >= state[1])
                new_t.append(t)
            return BooleanStateChangeArray(
                data,
                t=new_t,
                name="{} >= {}".format(self.name, other.name),
                shrink=True,
            )
        else:
            return BooleanStateChangeArray(
                self.data >= other, t=self.t, name="", shrink=True
            )

    def __add__(self, other):
        if isinstance(other, StateChangeArray):
            states = self._combine(other, null=0)
            data = []
            new_t = []
            for t, state in states.items():
                data.append(state[0] + state[1])
                new_t.append(t)
            return StateChangeArray(
                data, t=new_t, name="{} + {}".format(self.name, other.name), shrink=True
            )
        else:
            return StateChangeArray(self.data + other, t=self.t, name="", shrink=True)

    def __radd__(self, other):
        return StateChangeArray(other + self.data, t=self.t, name="")

    def __sub__(self, other):
        if isinstance(other, StateChangeArray):
            states = self._combine(other, null=0)
            data = []
            new_t = []
            for t, state in states.items():
                data.append(state[0] - state[1])
                new_t.append(t)
            return StateChangeArray(
                data, t=new_t, name="{} - {}".format(self.name, other.name), shrink=True
            )
        else:
            return StateChangeArray(self.data - other, t=self.t, name="", shrink=True)

    def __rsub__(self, other):
        return StateChangeArray(other - self.data, t=self.t, name="")

    def __and__(self, other):
        if not isinstance(other, StateChangeArray):
            raise ValueError("Expected StateChangeArray!")
        if not (self.is_bool() and other.is_bool()):
            raise ValueError(
                "Can't perform bitwise operation on non-boolean StateChangeArrays!"
            )
        states = self._combine(other)
        data = []
        new_t = []
        for t, state in states.items():
            if None in state:
                continue
            data.append(state[0] & state[1])
            new_t.append(t)
        return BooleanStateChangeArray(
            data, t=new_t, name="({} & {})".format(self.name, other.name), shrink=True
        )

    def __or__(self, other):
        if not isinstance(other, StateChangeArray):
            raise ValueError("Expected StateChangeArray!")
        if not (self.is_bool() and other.is_bool()):
            raise ValueError(
                "Can't perform bitwise operation on non-boolean StateChangeArrays!"
            )
        states = self._combine(other)
        data = []
        new_t = []
        for t, state in states.items():
            data.append((state[0] or False) | (state[1] or False))
            new_t.append(t)
        return BooleanStateChangeArray(
            data, t=new_t, name="({} | {})".format(self.name, other.name), shrink=True
        )

    def __xor__(self, other):
        if not isinstance(other, StateChangeArray):
            raise ValueError("Expected StateChangeArray!")
        if not (self.is_bool() and other.is_bool()):
            raise ValueError(
                "Can't perform bitwise operation on non-boolean StateChangeArrays!"
            )
        states = self._combine(other)
        data = []
        new_t = []
        for t, state in states.items():
            if None in state:
                continue
            data.append(state[0] ^ state[1])
            new_t.append(t)
        return BooleanStateChangeArray(
            data, t=new_t, name="({} ^ {})".format(self.name, other.name), shrink=True
        )

    def __invert__(self):  # THINKOF: should this return BooleanStateChangeArray?
        if not self.is_bool():
            raise ValueError(
                "Can't perform bitwise operation on non-boolean StateChangeArray!"
            )
        return StateChangeArray(~self.data, t=self.t, name="(~{})".format(self.name))

    def line(self, **kwargs):
        defaults = dict(label=self.name, drawstyle="steps-post", as_dt=False)
        defaults.update(kwargs)

        x = self._x(as_dt=defaults.pop("as_dt"))
        y = self._y()
        return Line2D(x, y, **defaults)

    def _x(self, as_dt=False):
        if as_dt:
            return self.dt
        else:
            return self.t

    def _y(self):
        return self.data

    def plot(self, *args, **kwargs):
        """

        :param kwargs:
        :return:
        """
        return plot(self, *args, **kwargs)


class BooleanStateChangeArray(StateChangeArray):
    def __init__(self, *args, **kwargs):
        """

        :param args:
        :param kwargs:
        """

        super().__init__(*args, **kwargs)
        if self.data.dtype != np.bool:
            raise ValueError("data is not of dtype 'bool'")

        self._type = "BooleanStateChangeArray"
        return

    def __repr__(self):
        return "BooleanStateChangeArray({}, t={}, name={})".format(
            np.array2string(self.data, **self._reprconfig),
            np.array2string(self.t, **self._reprconfig),
            repr(self.name),
        )


class Report(object):
    def __init__(self, t0, te, name=""):
        """

        :param t0:
        :param te:
        :param name:
        """

        if isinstance(t0, datetime.datetime):
            if t0.tzinfo is None:
                t0 = t0.replace(tzinfo=pytz.UTC)
            t0 = t0.timestamp()
        self.t0 = t0
        if isinstance(te, datetime.datetime):
            if te.tzinfo is None:
                te = te.replace(tzinfo=pytz.UTC)
            te = te.timestamp()
        if te < t0:
            raise ValueError("te can't be before t0!")
        self.te = te
        self.name = name
        self._type = "Report"

    def __repr__(self):
        return "Report(t0={}, te={}, name={})".format(
            *map(repr, [self.t0, self.te, self.name])
        )

    @property
    def duration(self):
        return self.te - self.t0

    @logged()
    def to_timeserie(self, fs=1, window=1):  # TODO: implement tolerance warning/error
        """

        :param fs:
        :param window:
        :return:
        """

        t0 = self.t0 - (window / fs)  # FINDOUT: Why do I do dis?
        window_data = np.zeros(window)
        k = round((self.te - self.t0) * fs)
        data = np.ones(k)
        data = np.append(window_data, np.append(data, window_data))
        return TimeSerie(data, t0=t0, fs=fs, name=self.name)

    @logged()
    def to_statechangearray(self):
        """

        :return:
        """

        t = [self.t0, self.te]
        data = [True, False]
        return StateChangeArray(data, t=t, name=self.name)

    @logged()
    def to_events(self):
        """

        :return:
        """

        event_t0 = Event(1, t=self.t0, name=self.name)
        event_te = Event(0, t=self.te, name=self.name)
        return (event_t0, event_te)


class Event(object):
    def __init__(self, value, t=0, name="", validity=1):
        """
        :param value:
        :param t:
        :param name:
        :param validity:
        """

        self.value = value
        if isinstance(t, datetime.datetime):
            if t.tzinfo is None:
                t = t.replace(tzinfo=pytz.UTC)
            t = t.timestamp()
        self.t = t
        self.name = name
        self.validity = validity
        self._type = "Report"

    @property
    def state(self):
        if self.validity:
            return self.value
        else:
            return "?"

    def __repr__(self):
        return "Event({}, t={}, name={})".format(
            *map(repr, [self.value, self.t, self.name])
        )
