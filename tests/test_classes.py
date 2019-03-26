import numpy as np
import matplotlib.pyplot as plt
import pytest
from diagnostics import (
    TimeSerie,
    BooleanTimeSerie,
    StateChangeArray,
    BooleanStateChangeArray,
    Report,
    Event,
)
import datetime as dt
import pytz


def compare_timeseries(a, b):
    if len(a) != len(b):
        return False
    data = all(a.data == b.data)
    channel = a.channel == b.channel
    name = a.name == b.name
    class_ = type(a) == type(b)
    return data & channel & name


def compare_statechangearrays(a, b):
    if len(a) != len(b):
        return False
    data = all(a.data == b.data)
    t = all(a.t == b.t)
    name = a.name == b.name
    class_ = type(a) == type(b)
    return data & t & name


def compare_events(a, b):
    value = a.value == b.value
    t = a.t == b.t
    name = a.name == b.name
    validity = a.validity == b.validity
    class_ = type(a) == type(b)
    return value & t & name & validity & class_


def compare_reports(a, b):
    t0 = a.t0 == b.t0
    te = a.te == b.te
    name = a.name == b.name
    return t0 & te & name


def test_timeserie_init():
    a = TimeSerie([1, 2, 3], t0=0, fs=1, name="a")
    assert all(a.data == [1, 2, 3])
    assert a.t0 == 0
    assert a.fs == 1
    assert a.name == "a"
    return True


def test_timeserie_datetime_t0():
    a = TimeSerie(
        [1, 2, 3], t0=pytz.utc.localize(dt.datetime(2000, 1, 1)), fs=1, name="a"
    )
    assert a.t0 == 946684800.0
    assert a.te == 946684802.0
    assert a.dt0 == dt.datetime(2000, 1, 1)
    assert a.dte == dt.datetime(2000, 1, 1, 0, 0, 2)
    assert all(a.dt == np.array(['2000-01-01T00:00:00', '2000-01-01T00:00:01', '2000-01-01T00:00:02'], dtype='datetime64'))
    return True


def test_timeserie_nparray():
    a = TimeSerie(np.array([1, 2, 3]), t0=0, fs=1, name="a")
    assert isinstance(a.data, np.ndarray)
    b = TimeSerie([1, 2, 3], t0=0, fs=1, name="b")
    assert isinstance(b.data, np.ndarray)
    return True


def test_timeserie_properties():
    a = TimeSerie([1, 2, 3], name="a", fs=2, t0=1)
    assert a.hz == a.fs
    a.hz = 1
    assert a.fs == 1
    b = TimeSerie([4, 5, 6], name="a", fs=1, t0=1)
    b.data = [1, 2, 3]
    assert isinstance(b.data, np.ndarray)
    assert all(b.data == [1, 2, 3])
    assert compare_timeseries(a, b)
    with pytest.raises(ValueError):
        a.channel = (1, 2, 3)
    return True


def test_timeserie_resett0():
    a = TimeSerie([1, 2, 3], name="a", fs=2, t0=1)
    assert a.t0 == 1
    a.reset_t0()
    assert a.t0 == 0
    return True


def test_timeserie_roundt0():
    a = TimeSerie(np.arange(10), name="a", fs=1, t0=1.1)
    assert a.t0 == 1.1
    a.round_t0()
    assert a.t0 == 1.0
    a = TimeSerie(np.arange(100), name="a", fs=10, t0=1.11)
    assert a.t0 == 1.11
    a.round_t0()
    assert a.t0 == 1.1


def test_timeserie_iter():
    a = TimeSerie([1, 2, 3], fs=2, t0=1)
    lst = list(a.iter())
    assert lst == [(1.0, 1), (1.5, 2), (2.0, 3)]
    return True


def test_timeserie_defaultsettings():
    a = TimeSerie([1, 2, 3])
    assert a.name == ""
    assert a.t0 == 0
    assert a.fs == 1
    return True


def test_timeserie_fsfloat():
    a = TimeSerie([1, 2, 3], fs=1.1)
    assert a.fs == 1
    return True


def test_timeserie_repr():
    a = TimeSerie([1, 2, 3], fs=2, t0=1, name="a")
    assert repr(a) == "TimeSerie([1 2 3], t0=1, name='a', fs=2)"
    b = TimeSerie([1, 2, 3, 4, 5, 6, 7], fs=3, t0=2, name="b")
    assert repr(b) == "TimeSerie([1 2 3 4 5 6 7], t0=2, name='b', fs=3)"
    return True


def test_timeserie_len():
    a = TimeSerie([1, 2, 3])
    assert len(a) == 3
    b = TimeSerie([1, 2, 3, 4, 5, 6, 7])
    assert len(b) == 7
    return True


def test_timeserie_getitem():
    a = TimeSerie([1, 2, 3])
    assert a[0] == 1
    assert a[-1] == 3
    b = TimeSerie([1, 2, 3, 4, 5, 6, 7])
    assert b[1] == 2
    return True


def test_timeserie_eq():
    a = TimeSerie([1, 2, 3], name="a", fs=1, t0=1)
    eq_1 = a == a
    assert compare_timeseries(
        eq_1, BooleanTimeSerie([True, True, True], t0=1, name="(a == a)", fs=1)
    )
    eq_2 = a == 1
    assert compare_timeseries(
        eq_2, BooleanTimeSerie([True, False, False], t0=1, fs=1, name="")
    )
    b = TimeSerie([1, 2, 3], fs=2, name="b", t0=1)
    with pytest.raises(ValueError):
        eq_4 = a == b
    return True


def test_timeserie_neq():
    a = TimeSerie([1, 2, 3], name="a", fs=1, t0=1)
    eq_1 = a != a
    assert compare_timeseries(
        eq_1, BooleanTimeSerie([False, False, False], t0=1, name="(a != a)", fs=1)
    )
    eq_2 = a != 1
    assert compare_timeseries(
        eq_2, BooleanTimeSerie([False, True, True], t0=1, fs=1, name="")
    )
    b = TimeSerie([1, 2, 3], fs=2, name="b", t0=1)
    with pytest.raises(ValueError):
        eq_4 = a != b
    return True


def test_timeserie_lessthen():
    a = TimeSerie([1, 2, 3], name="a", fs=1, t0=1)
    b = TimeSerie([3, 1, 5], name="b", fs=1, t0=1)
    lt_1 = b < a
    assert compare_timeseries(
        lt_1, BooleanTimeSerie([False, True, False], t0=1, fs=1, name="(b < a)")
    )
    lt_2 = a < 2
    assert compare_timeseries(
        lt_2, BooleanTimeSerie([True, False, False], t0=1, fs=1, name="")
    )
    c = TimeSerie([7, 8, 9, 0], name="c", fs=1, t0=0)
    with pytest.raises(ValueError):
        eq_3 = a < c
    return True


def test_timeserie_greaterthen():
    a = TimeSerie([1, 2, 3], name="a", fs=1, t0=1)
    b = TimeSerie([3, 1, 5], name="b", fs=1, t0=1)
    lt_1 = a > b
    assert compare_timeseries(
        lt_1, BooleanTimeSerie([False, True, False], t0=1, fs=1, name="(a > b)")
    )
    lt_2 = a > 2
    assert compare_timeseries(
        lt_2, BooleanTimeSerie([False, False, True], t0=1, fs=1, name="")
    )
    c = TimeSerie([7, 8, 9, 0], name="c", fs=1, t0=0)
    with pytest.raises(ValueError):
        eq_3 = a > c
    return True


def test_timeserie_lessequalthen():
    a = TimeSerie([1, 2, 3], name="a", fs=1, t0=1)
    b = TimeSerie([3, 1, 3], name="b", fs=1, t0=1)
    lt_1 = b <= a
    assert compare_timeseries(
        lt_1, BooleanTimeSerie([False, True, True], t0=1, fs=1, name="(b <= a)")
    )
    lt_2 = a <= 2
    assert compare_timeseries(
        lt_2, BooleanTimeSerie([True, True, False], t0=1, fs=1, name="")
    )
    c = TimeSerie([7, 8, 9, 0], name="c", fs=1, t0=0)
    with pytest.raises(ValueError):
        eq_3 = a <= c
    return True


def test_timeserie_greaterequalthen():
    a = TimeSerie([1, 2, 3], name="a", fs=1, t0=1)
    b = TimeSerie([3, 1, 3], name="b", fs=1, t0=1)
    lt_1 = a >= b
    assert compare_timeseries(
        lt_1, BooleanTimeSerie([False, True, True], t0=1, fs=1, name="(a >= b)")
    )
    lt_2 = a >= 2
    assert compare_timeseries(
        lt_2, BooleanTimeSerie([False, True, True], t0=1, fs=1, name="")
    )
    c = TimeSerie([7, 8, 9, 0], name="c", fs=1, t0=0)
    with pytest.raises(ValueError):
        eq_3 = a >= c
    return True


def test_timeserie_and():
    a = TimeSerie([True, True, False, False], name="a", fs=1, t0=1)
    b = BooleanTimeSerie([True, False, True, False], name="b", fs=1, t0=1)
    c = TimeSerie([1, 2, 3, 4], name="c", fs=1, t0=1)
    d = BooleanTimeSerie([True, True, True, True], name="d", fs=2, t0=3)
    and_1 = a & b
    assert compare_timeseries(
        and_1, BooleanTimeSerie([True, False, False, False], name="(a & b)", fs=1, t0=1)
    )
    with pytest.raises(ValueError):
        and_2 = a & c
    with pytest.raises(ValueError):
        and_3 = b & d
    return True


def test_timeserie_or():
    a = TimeSerie([True, True, False, False], name="a", fs=1, t0=1)
    b = BooleanTimeSerie([True, False, True, False], name="b", fs=1, t0=1)
    c = TimeSerie([1, 2, 3, 4], name="c", fs=1, t0=1)
    d = BooleanTimeSerie([True, True, True, True], name="d", fs=2, t0=3)
    and_1 = a | b
    assert compare_timeseries(
        and_1, BooleanTimeSerie([True, True, True, False], name="(a | b)", fs=1, t0=1)
    )
    with pytest.raises(ValueError):
        and_2 = a | c
    with pytest.raises(ValueError):
        and_3 = b | d
    return True


def test_timeserie_xor():
    a = TimeSerie([True, True, False, False], name="a", fs=1, t0=1)
    b = BooleanTimeSerie([True, False, True, False], name="b", fs=1, t0=1)
    c = TimeSerie([1, 2, 3, 4], name="c", fs=1, t0=1)
    d = BooleanTimeSerie([True, True, True, True], name="d", fs=2, t0=3)
    and_1 = a ^ b
    assert compare_timeseries(
        and_1, BooleanTimeSerie([False, True, True, False], name="(a ^ b)", fs=1, t0=1)
    )
    with pytest.raises(ValueError):
        and_2 = a ^ c
    with pytest.raises(ValueError):
        and_3 = b ^ d
    return True


def test_timeserie_invert():
    a = TimeSerie([True, False], name="a", fs=1, t0=1)
    b = TimeSerie([1, 2, 3, 4], name="b", fs=1, t0=1)
    invert_1 = ~a
    assert compare_timeseries(
        invert_1, BooleanTimeSerie([False, True], name="(~a)", fs=1, t0=1)
    )
    with pytest.raises(ValueError):
        invert_2 = ~b
    return True


def test_timeserie_add():
    a = TimeSerie([1, 2, 3], name="a", fs=1, t0=1)
    b = TimeSerie([3, 1, 3], name="b", fs=1, t0=1)
    add_1 = a + b
    assert compare_timeseries(add_1, TimeSerie([4, 3, 6], name="(a + b)", fs=1, t0=1))
    add_2 = a + 1
    assert compare_timeseries(add_2, TimeSerie([2, 3, 4], name="", fs=1, t0=1))
    c = TimeSerie([7, 8, 9, 0], name="c", fs=1, t0=0)
    with pytest.raises(ValueError):
        add_3 = a + c
    return True


def test_timeserie_radd():
    a = TimeSerie([1, 2, 3], name="a", fs=1, t0=1)
    add_1 = 1 + a
    assert compare_timeseries(add_1, TimeSerie([2, 3, 4], name="", fs=1, t0=1))
    return True


def test_timeserie_sub():
    a = TimeSerie([1, 2, 3], name="a", fs=1, t0=1)
    b = TimeSerie([3, 1, 3], name="b", fs=1, t0=1)
    sub_1 = a - b
    assert compare_timeseries(sub_1, TimeSerie([-2, 1, 0], name="(a - b)", fs=1, t0=1))
    sub_2 = a - 1
    assert compare_timeseries(sub_2, TimeSerie([0, 1, 2], name="", fs=1, t0=1))
    c = TimeSerie([7, 8, 9, 0], name="c", fs=1, t0=0)
    with pytest.raises(ValueError):
        sub_3 = a - c
    return True


def test_timeserie_rsub():
    a = TimeSerie([1, 2, 3], name="a", fs=1, t0=1)
    sub_1 = 1 - a
    assert compare_timeseries(sub_1, TimeSerie([0, -1, -2], name="", fs=1, t0=1))
    return True


def test_timeserie_at():
    a = TimeSerie([1, 2, 3, 4, 5, 6], t0=1, fs=1, name="a")
    assert a.at(2) == 2
    return True


def test_timeserie_where():
    a = TimeSerie([1, 2, 3, 4, 5, 6], t0=1, fs=1, name="a")
    d = a.where(a.t >= 3)
    d_test = np.array([3, 4, 5, 6])
    assert len(d) == len(d_test)
    assert all(d == d_test)
    return True


def test_timeserie_empty():
    a = TimeSerie.empty(1, 10, fs=10, name="a")
    assert len(a) == 90
    assert a.te == 9.9
    b = TimeSerie.empty(2, 5, 4, name="b", inclusive=True)
    assert len(b) == 13
    assert b.te == 5.0
    c = TimeSerie.empty(
        dt.datetime(2018, 1, 1, 12),
        dt.datetime(2018, 1, 1, 13),
        fs=1,
        name="c",
        inclusive=True,
    )
    assert c.te - c.t0 == 3600.0
    return True


def test_timeserie_plot():
    plt.ioff()
    a = TimeSerie([-2, -1, 0, 1, 2, 3, 4, 5], name="a", fs=1, t0=1)
    f, ax, lines = a.plot(show=False)

    b = TimeSerie([-2, -1, 0, 1, 2, 3, 4, 5], name="b", fs=1, t0=dt.datetime(2019,1,1))
    f, ax, lines = b.plot(as_dt=True, show=False)
    return True


def test_timserie_mod():
    a = TimeSerie([-2, -1, 0, 1, 2, 3, 4, 5], name="a", fs=1, t0=1)
    b = a.modify("default", inplace=False)
    assert compare_timeseries(b, a)
    a.modify("default", inplace=True)
    assert compare_timeseries(a, b)

    a = TimeSerie([-2, -1, 0, 1, 2, 3, 4, 5], name="a", fs=1, t0=1)
    b = a.modify("zero_negatives", inplace=False)
    assert compare_timeseries(
        b, TimeSerie([0, 0, 0, 1, 2, 3, 4, 5], name="a", fs=1, t0=1)
    )
    a.modify("zero_negatives", inplace=True)
    assert compare_timeseries(a, b)

    a = TimeSerie([-2, -1, 0, 1, 2, 3, 4, 5], name="a", fs=1, t0=1)
    b = a.modify("correct_negatives", inplace=False)
    assert compare_timeseries(
        b, TimeSerie([0, 1, 2, 3, 4, 5, 6, 7], name="a", fs=1, t0=1)
    )
    a.modify("correct_negatives", inplace=True)
    assert compare_timeseries(a, b)
    return True


def test_timeserie_toevents():
    a = TimeSerie([1, 2, 3], t0=1, fs=2, name="a")
    e1, e2, e3 = a.to_events()
    assert compare_events(e1, Event(1, t=1, name="a"))
    assert compare_events(e2, Event(2, 1.5, name="a"))
    assert compare_events(e3, Event(3, 2.0, name="a"))
    return True


def test_timeserie_tobool():
    a = TimeSerie([0, 0, 0, 1, 2, 3], t0=1, fs=2, name="a")
    b = a.to_bool(inplace=False)
    assert compare_timeseries(
        b,
        BooleanTimeSerie([False, False, False, True, True, True], t0=1, fs=2, name="a"),
    )
    a.to_bool(inplace=True)
    assert compare_timeseries(a, b)
    return True


def test_timeserie_tostatechangearray():
    a = TimeSerie([1, 1, 1, 2, 2, 3, 4, 4, 4], t0=1, fs=2, name="a")
    sta_a = a.to_statechangearray()
    assert compare_statechangearrays(
        sta_a, StateChangeArray([1, 2, 3, 4], t=[1, 2.5, 3.5, 4], name="a")
    )
    return True


def test_timeserie_interpolate():
    a = TimeSerie([1, 2, 3, 4, 5, 6, 7, 8, 9, 10], t0=0, fs=1, name="a")
    t_new = np.arange(0, 9 + 0.5, 0.5)
    b = a.interpolate(t_new, inplace=False)
    assert compare_timeseries(
        b, TimeSerie(np.arange(1, 10.5, 0.5), t0=0, fs=2, name="a")
    )
    a.interpolate(t_new, inplace=True)
    assert compare_timeseries(a, b)
    c = BooleanTimeSerie(
        [False, False, True, True, True, False, False], t0=1, fs=1, name="c"
    )
    t_new = np.arange(1, 7 + 0.5, 0.5)
    d = c.interpolate(t_new, inplace=False)
    assert compare_timeseries(
        d,
        BooleanTimeSerie(3 * [False] + 7 * [True] + 3 * [False], t0=1, fs=2, name="c"),
    )
    c.interpolate(t_new, inplace=True)
    assert compare_timeseries(c, d)
    return True


def test_booleantimeserie_init():
    with pytest.raises(ValueError):
        a = BooleanTimeSerie([1, 2, 3], fs=1, t0=1, name="a")
    return True


def test_booleantimeserie_properties():
    a = BooleanTimeSerie([False, True, False], name="a", fs=2, t0=1)
    assert a.hz == a.fs
    a.hz = 1
    assert a.fs == 1
    b = BooleanTimeSerie([True, True, True], name="a", fs=1, t0=1)
    b.data = [False, True, False]
    assert isinstance(b.data, np.ndarray)
    assert all(b.data == [False, True, False])
    assert compare_timeseries(a, b)
    with pytest.raises(ValueError):
        a.data = [1, 2, 3]
    return True


def test_statechangearray_init():
    a = StateChangeArray([1, 3, 5, 7], t=[1, 2, 4, 8])
    assert all(a.data == [1, 3, 5, 7])
    assert all(a.t == [1, 2, 4, 8])
    assert a.name == ""
    b = StateChangeArray([2, 4, 6, 8], t=[1, 2, 4, 8], name="b")
    with pytest.raises(ValueError):
        c = StateChangeArray([1, 2, 3, 4], t=[1, 2, 4], name="c")
    d = StateChangeArray(np.array([1, 3, 5, 7]), t=np.array([1, 2, 4, 8]))
    assert compare_statechangearrays(a, d)
    with pytest.raises(ValueError):
        e = StateChangeArray([1, 2, 3, 4], [1, 5, 3, 8], name="e")
    with pytest.raises(ValueError):
        f = StateChangeArray([1, 2, 2, 3], [1, 2, 4, 7], name="f")
    g = StateChangeArray([1, 2, 2, 3], [1, 2, 4, 7], name="g", shrink=True)
    assert compare_statechangearrays(
        g, StateChangeArray([1, 2, 3], [1, 2, 7], name="g")
    )
    h = StateChangeArray([1, 3], t=[pytz.utc.localize(dt.datetime(2019, 1, 1, 8)),
                                    pytz.utc.localize(dt.datetime(2019, 1, 1, 9))])
    assert all(h.t == [1546329600.0, 1546333200.0])
    return True


def test_statechangearray_len():
    a = StateChangeArray([1, 3, 5, 7], t=[1, 2, 4, 8])
    assert len(a) == 4
    return True


def test_statechangearray_iter():
    a = StateChangeArray([1, 3, 5, 7], t=[1, 2, 4, 8])
    gen = a.iter()
    assert next(gen) == (1, 1)
    assert next(gen) == (2, 3)
    assert next(gen) == (4, 5)
    assert next(gen) == (8, 7)
    with pytest.raises(StopIteration):
        next(gen)
    return True


def test_statechangearray_dt():
    a = StateChangeArray([1, 3, 5, 7, 9],
                         t=[pytz.utc.localize(dt.datetime(2019,1,1,0)),
                            pytz.utc.localize(dt.datetime(2019,1,1,1)),
                            pytz.utc.localize(dt.datetime(2019,1,1,3)),
                            pytz.utc.localize(dt.datetime(2019,1,1,6)),
                            pytz.utc.localize(dt.datetime(2019,1,1,12))],
                         name='a')
    assert all(a.dt == np.array(['2019-01-01T00:00:00.000000',
                                 '2019-01-01T01:00:00.000000',
                                 '2019-01-01T03:00:00.000000',
                                 '2019-01-01T06:00:00.000000',
                                 '2019-01-01T12:00:00.000000'], dtype='datetime64'))
    return True


def test_statechangearray_plot():
    plt.ioff()
    a = StateChangeArray([-2, -1, 0, 1, 2, 3, 4, 5], t=[0, 1, 2, 3, 4, 5, 8, 10], name="a")
    f, ax, lines = a.plot(show=False)

    f, ax, lines = a.plot(show=False, style="nostyle")

    b = StateChangeArray([-2, -1, 0, 1, 2, 3, 4, 5],
                         t=[dt.datetime(2019, 1, 1, 0),
                            dt.datetime(2019, 1, 1, 1),
                            dt.datetime(2019, 1, 1, 3),
                            dt.datetime(2019, 1, 1, 6),
                            dt.datetime(2019, 1, 1, 7),
                            dt.datetime(2019, 1, 1, 8),
                            dt.datetime(2019, 1, 1, 10),
                            dt.datetime(2019, 1, 1, 11)], name="b")
    f, ax, lines = b.plot(as_dt=True, show=False)
    return True


def test_statechangearray_events():
    a = StateChangeArray([1, 3, 5, 7], t=[1, 2, 4, 8], name="a")
    e1, e2, e3, e4 = a.to_events()
    assert compare_events(e1, Event(1, t=1, name="a"))
    assert compare_events(e2, Event(3, t=2, name="a"))
    assert compare_events(e3, Event(5, t=4, name="a"))
    assert compare_events(e4, Event(7, t=8, name="a"))
    return True


def test_statechangearray_fromevents():
    a1 = Event(1, t=1, name='a')
    a2 = Event(3, t=2, name='a')
    a3 = Event(5, t=4, name='a')
    a4 = Event(7, t=8, name='a')
    a = StateChangeArray.from_events([a1, a2, a3, a4])
    assert compare_statechangearrays(a, StateChangeArray([1,3,5,7],t=[1,2,4,8],name='a'))
    a5 = Event(9, t=6, name='a')
    with pytest.raises(ValueError):
        b = StateChangeArray.from_events([a1,a2,a3,a4,a5])
    return True


def test_statechangearray_fromreports():
    a1 = Report(t0=2, te=4, name='a')
    a2 = Report(t0=6, te=8, name='a')
    a = StateChangeArray.from_reports([a1, a2])
    assert compare_statechangearrays(a, StateChangeArray([True,False,True,False], t=[2,4,6,8],name='a'))
    a3 = Report(t0=12, te=13, name='a')
    a3.te = 10  # this is a hacky way to test errors in the from_reports() method
    with pytest.raises(ValueError):
        b = StateChangeArray.from_reports([a1, a2, a3])
    a4 = Report(t0=5, te=10, name='a')
    with pytest.raises(ValueError):
        c = StateChangeArray.from_reports([a1,a2,a4])
    return True


def test_statechangearray_at():
    a = StateChangeArray([1, 3, 5, 7], t=[1, 2, 4, 7], name="a")
    a_at = a.at(2)
    assert a_at == 3
    return True


def test_statechangearray_where():
    a = StateChangeArray([1, 3, 5, 7], t=[1, 2, 4, 7], name="a")
    a_where = a.where(a.t >= 4)
    assert all(a_where == [5, 7])
    return True


def test_statechangearray_state():
    a = StateChangeArray([True,False,True,False], t=[2,4,6,8], name="a")
    assert a.state() == False
    assert a.state(2) == True
    assert a.state(3) == True
    assert a.state(3.9) == True
    assert a.state(4) == False
    with pytest.raises(IndexError):
        a.state(1)
    b = StateChangeArray([1,3,7,9], t=[2,4,6,8], name="b")
    assert b.state() == 9
    assert b.state(2) == 1
    assert b.state(3) == 1
    assert b.state(3.9) == 1
    assert b.state(4) == 3
    return True


def test_statechangearray_getitem():
    a = StateChangeArray([1, 3, 5, 7], t=[1, 2, 4, 7], name="a")
    assert a[1] == 3
    return True


def test_statechangearray_and():
    a = StateChangeArray([True, False, True, False], t=[2, 4, 6, 8], name="a")
    b = StateChangeArray([True, False, True, False], t=[3, 5, 7, 9], name="b")
    with pytest.raises(ValueError):
        a_and_1 = a & 1
    c = StateChangeArray([2, 4, 6, 8], t=[1, 3, 4, 6], name="c")
    with pytest.raises(ValueError):
        a_and_c = a & c
    a_and_b = a & b
    assert compare_statechangearrays(
        a_and_b,
        BooleanStateChangeArray(
            [True, False, True, False], t=[3, 4, 7, 8], name="(a & b)"
        ),
    )
    a_and_a = a & a
    a_ = a
    a_.name = "(a & a)"
    assert compare_statechangearrays(a_and_a, a_)
    return True


def test_statechangearray_or():
    a = StateChangeArray([True, False, True, False], t=[2, 4, 6, 8], name="a")
    b = StateChangeArray([True, False, True, False], t=[3, 5, 7, 9], name="b")
    with pytest.raises(ValueError):
        a_and_1 = a | 1
    c = StateChangeArray([2, 4, 6, 8], t=[1, 3, 4, 6], name="c")
    with pytest.raises(ValueError):
        a_and_c = a | c
    a_and_b = a | b
    assert compare_statechangearrays(
        a_and_b,
        BooleanStateChangeArray(
            [True, False, True, False], t=[2, 5, 6, 9], name="(a | b)"
        ),
    )
    a_and_a = a | a
    a_ = a
    a_.name = "(a | a)"
    assert compare_statechangearrays(a_and_a, a_)
    return True


def test_statechangearray_exor():
    a = StateChangeArray([True, False, True, False], t=[2, 4, 6, 8], name="a")
    b = StateChangeArray([True, False, True, False], t=[3, 5, 7, 9], name="b")
    with pytest.raises(ValueError):
        a_and_1 = a ^ 1
    c = StateChangeArray([2, 4, 6, 8], t=[1, 3, 4, 6], name="c")
    with pytest.raises(ValueError):
        a_and_c = a ^ c
    a_and_b = a ^ b
    assert compare_statechangearrays(
        a_and_b,
        BooleanStateChangeArray(
            [False, True, False, True, False, True, False],
            t=[3, 4, 5, 6, 7, 8, 9],
            name="(a ^ b)",
        ),
    )
    a_and_a = a ^ a
    a_ = a
    a_.name = "(a ^ a)"
    assert compare_statechangearrays(
        a_and_a, BooleanStateChangeArray([False], t=[2], name="(a ^ a)")
    )
    return True


def test_statechangearray_invert():
    a = StateChangeArray([True, False, True, False], t=[2, 4, 6, 8], name="a")
    not_a = ~a
    assert compare_statechangearrays(
        not_a,
        BooleanStateChangeArray(
            [False, True, False, True], t=[2, 4, 6, 8], name="(~a)"
        ),
    )
    with pytest.raises(ValueError):
        b = StateChangeArray([1,2,3,4], t=[2, 4, 6, 8], name="b")
        not_b = ~b
    return True


def test_statechangearray_duration():
    a = StateChangeArray([1, 3, 5, 7], t=[1, 2, 4, 7], name="a")
    assert all(a.duration() == [1, 2, 3])
    return True


def test_statechangearray_totimeseries():
    a = StateChangeArray([1, 3, 5, 7], t=[1, 2, 4, 7], name="a")
    ts_a = a.to_timeseries(fs=2)
    assert compare_timeseries(
        ts_a, TimeSerie([1, 1, 3, 3, 3, 3, 5, 5, 5, 5, 5, 5, 7], t0=1, fs=2, name="a")
    )
    b = StateChangeArray([1, 3, 5, 7], t=[1, 2, 2.25, 4])
    with pytest.raises(ValueError):
        ts_b = b.to_timeseries(fs=2)
    with pytest.raises(NameError):
        b.to_timeseries(fs=2, method="nonExistingMethod")
    c = StateChangeArray([1, 3, 5, 7], t=[1, 2, 4, 8], name="c")
    ts_c = c.to_timeseries(fs=2, method="interpolate")
    assert compare_timeseries(
        ts_c,
        TimeSerie(
            [1, 2, 3, 3.5, 4, 4.5, 5, 5.25, 5.5, 5.75, 6, 6.25, 6.5, 6.75, 7],
            t0=1,
            fs=2,
            name="c",
        ),
    )
    return True


def test_statechangearray_isbool():
    a = StateChangeArray([1, 3, 5, 7], t=[1, 2, 4, 7], name="a")
    assert a.is_bool() is False
    a = StateChangeArray([True, False, True, False], t=[1, 2, 4, 7], name="a")
    assert a.is_bool() is True


def test_statechangearray_tobool():
    a = StateChangeArray([0, 1, 0, 2, 0], t=[1, 2, 4, 5, 7], name="a")
    b = a.to_bool(inplace=False)
    assert compare_statechangearrays(
        b,
        BooleanStateChangeArray(
            [False, True, False, True, False], t=[1, 2, 4, 5, 7], name="a"
        ),
    )
    a.to_bool(inplace=True)
    assert a.is_bool()
    assert compare_statechangearrays(a, b)
    return True


def test_booleanstatechangearray_init():
    a = BooleanStateChangeArray(
        [False, True, False, True, False], [1, 3, 5, 6, 9], name="a"
    )
    with pytest.raises(ValueError):
        b = BooleanStateChangeArray([1, 2, 3, 5, 8], t=[1, 2, 4, 8, 16], name="b")
    return True


def test_booleanstatechangearray_repr():
    a = BooleanStateChangeArray(
        [False, True, False, True, False], [1, 3, 5, 6, 9], name="a"
    )
    assert (
        repr(a)
        == "BooleanStateChangeArray([False  True False  True False], t=[1 3 5 6 9], name='a')"
    )
    return True


def test_report_init():
    a = Report(1, 3, name="a")
    assert a.t0 == 1
    assert a.te == 3
    assert a.name == "a"
    start = pytz.utc.localize(dt.datetime(2000, 1, 1))
    end = start + dt.timedelta(seconds=60)
    b = Report(start, end, name="b")
    assert b.t0 == 946684800.0
    assert b.te == 946684860.0
    with pytest.raises(ValueError):
        c = Report(t0=5, te=1, name="c")
    return True


def test_report_toevent():
    a = Report(1, 3, name="a")
    a_start, a_end = a.to_events()
    assert compare_events(a_start, Event(1, t=1, name="a"))
    assert compare_events(a_end, Event(0, t=3, name="a"))
    return True


def test_report_totimeserie():
    a = Report(2, 4, name="a")
    ts_a = a.to_timeserie()
    assert compare_timeseries(
        ts_a, TimeSerie([0.0, 1.0, 1.0, 0.0], t0=1.0, fs=1, name="a")
    )
    b = Report(3, 8, name="b")
    ts_b = b.to_timeserie(fs=4, window=2)
    cmp_b = TimeSerie([0.0, 0.0] + (20 * [1.0]) + [0.0, 0.0], t0=2.5, fs=4, name="b")
    ts_b._reprconfig["threshold"] = 100
    cmp_b._reprconfig["threshold"] = 100
    assert compare_timeseries(ts_b, cmp_b)
    return True


def test_event_init():
    a = Event(1)
    assert a.value == 1
    assert a.t == 0
    assert a.name == ""
    assert a.validity == 1
    start = pytz.utc.localize(dt.datetime(2000, 1, 1))
    b = Event(2, t=start)
    assert b.value == 2
    assert b.t == 946684800.0
    c = Event(3, t=3, name="c")
    assert c.value == 3
    assert c.t == 3
    assert c.name == "c"
    d = Event(4, t=4, name="d", validity=0)
    assert d.value == 4
    assert d.t == 4
    assert d.name == "d"
    assert d.validity == 0
    return True


def test_event_state():
    a = Event(4, t=4, name="a", validity=1)
    assert a.state == a.value
    a.validity = 0
    assert a.state == "?"
    return True
