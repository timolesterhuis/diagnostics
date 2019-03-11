import numpy as np
import pytest
from diagnostics import TimeSerie, BooleanTimeSerie, StateChangeArray, Report, Event
import datetime as dt
import pytz 


def compare_timeseries(a, b):
    data = all(a.data == b.data)
    channel = (a.channel == b.channel)
    name = (a.name == b.name)
    class_ = type(a) == type(b)
    return (data & channel & name)


def test_timeserie_init():
    a = TimeSerie([1,2,3], t0=0, fs=1, name='a')
    assert all(a.data == [1,2,3])
    assert a.t0 == 0
    assert a.fs == 1
    assert a.name == 'a'
    return True


def test_timeserie_datetime_t0():
    a = TimeSerie([1,2,3], t0=pytz.localize(dt.datetime(2000,1,1)), fs=1, name='a')
    assert a.t0 == 946684800.0
    return True


def test_timeserie_nparray():
    a = TimeSerie(np.array([1,2,3]), t0=0, fs=1, name='a')
    assert isinstance(a.data, np.ndarray)
    b = TimeSerie([1,2,3], t0=0, fs=1, name='b')
    assert isinstance(b.data, np.ndarray)
    return True


def test_timeserie_properties():
    a = TimeSerie([1,2,3], name='a', fs=2, t0=1)
    assert a.hz == a.fs
    a.hz = 1
    assert a.fs == 1
    b = TimeSerie([4,5,6], name='a', fs=1, t0=1)
    b.data = [1,2,3]
    assert isinstance(b.data, np.ndarray)
    assert all(b.data == [1,2,3])
    assert compare_timeseries(a,b)
    with pytest.raises(ValueError):
        a.channel = (1,2,3)
    return True


def test_timeserie_resett0():
    a = TimeSerie([1,2,3], name='a', fs=2, t0=1)
    assert a.t0 == 1
    a.reset_t0()
    assert a.t0 == 0
    return True


def test_timeserie_roundt0():
    a = TimeSerie(np.arange(10), name='a', fs=1, t0=1.1)
    assert a.t0 == 1.1
    a.round_t0()
    assert a.t0 == 1.0
    a = TimeSerie(np.arange(100), name='a', fs=10, t0=1.11)
    assert a.t0 == 1.11
    a.round_t0()
    assert a.t0 == 1.1


def test_timeserie_iter():
    a = TimeSerie([1,2,3], fs=2, t0=1)
    lst = list(a.iter())
    assert lst == [(1.0,1), (1.5,2), (2.0,3)]
    return True


def test_timeserie_defaultsettings():
    a = TimeSerie([1,2,3])
    assert a.name == ''
    assert a.t0 == 0
    assert a.fs == 1
    return True


def test_timeserie_fsfloat():
    a = TimeSerie([1,2,3], fs=1.1)
    assert a.fs == 1
    return True


def test_timeserie_repr():
    a = TimeSerie([1,2,3], fs=2, t0=1, name='a')
    assert repr(a) == "TimeSerie([1 2 3], t0=1, name='a', fs=2)"
    b = TimeSerie([1,2,3,4,5,6,7], fs=3, t0=2, name='b')
    assert repr(b) ==  "TimeSerie([1 2 3 ... 5 6 7], t0=2, name='b', fs=3)"
    return True


def test_timeserie_len():
    a = TimeSerie([1,2,3])
    assert len(a) == 3
    b = TimeSerie([1,2,3,4,5,6,7])
    assert len(b) == 7
    return True


def test_timeserie_getitem():
    a = TimeSerie([1,2,3])
    assert a[0] == 1
    assert a[-1] == 3
    b = TimeSerie([1,2,3,4,5,6,7])
    assert b[1] == 2
    return True


def test_timeserie_eq():
    a = TimeSerie([1,2,3], name='a', fs=1, t0=1)    
    eq_1 = (a == a)
    assert compare_timeseries(eq_1, BooleanTimeSerie([True, True, True], t0=1, name='(a == a)', fs=1))
    eq_2 = (a == 1)
    assert compare_timeseries(eq_2, BooleanTimeSerie([True, False, False], t0=1, fs=1, name=''))
    b = TimeSerie([1,2,3], fs=2, name='b', t0=1)
    with pytest.raises(ValueError):
        eq_4 = (a == b)
    return True


def test_timeserie_neq():
    a = TimeSerie([1,2,3], name='a', fs=1, t0=1)    
    eq_1 = (a != a)
    assert compare_timeseries(eq_1, BooleanTimeSerie([False, False, False], t0=1, name='(a != a)', fs=1))
    eq_2 = (a != 1)
    assert compare_timeseries(eq_2, BooleanTimeSerie([False, True, True], t0=1, fs=1, name=''))
    b = TimeSerie([1,2,3], fs=2, name='b', t0=1)
    with pytest.raises(ValueError):
        eq_4 = (a != b)    
    return True


def test_timeserie_lessthen():
    a = TimeSerie([1,2,3], name='a', fs=1, t0=1)
    b = TimeSerie([3,1,5], name='b', fs=1, t0=1)
    lt_1 = (b < a)
    assert compare_timeseries(lt_1, BooleanTimeSerie([False, True, False], t0=1, fs=1, name='(b < a)'))
    lt_2 = (a < 2)
    assert compare_timeseries(lt_2, BooleanTimeSerie([True, False, False], t0=1, fs=1, name=''))
    c = TimeSerie([7,8,9,0], name='c', fs=1, t0=0)
    with pytest.raises(ValueError):
        eq_3 = (a < c)
    return True


def test_timeserie_greaterthen():
    a = TimeSerie([1,2,3], name='a', fs=1, t0=1)
    b = TimeSerie([3,1,5], name='b', fs=1, t0=1)
    lt_1 = (a > b)
    assert compare_timeseries(lt_1, BooleanTimeSerie([False, True, False], t0=1, fs=1, name='(a > b)'))
    lt_2 = (a > 2)
    assert compare_timeseries(lt_2, BooleanTimeSerie([False, False, True], t0=1, fs=1, name=''))
    c = TimeSerie([7,8,9,0], name='c', fs=1, t0=0)
    with pytest.raises(ValueError):
        eq_3 = (a > c)
    return True


def test_timeserie_lessequalthen():
    a = TimeSerie([1,2,3], name='a', fs=1, t0=1)
    b = TimeSerie([3,1,3], name='b', fs=1, t0=1)
    lt_1 = (b <= a)
    assert compare_timeseries(lt_1, BooleanTimeSerie([False, True, True], t0=1, fs=1, name='(b <= a)'))
    lt_2 = (a <= 2)
    assert compare_timeseries(lt_2, BooleanTimeSerie([True, True, False], t0=1, fs=1, name=''))
    c = TimeSerie([7,8,9,0], name='c', fs=1, t0=0)
    with pytest.raises(ValueError):
        eq_3 = (a <= c)
    return True


def test_timeserie_greaterequalthen():
    a = TimeSerie([1,2,3], name='a', fs=1, t0=1)
    b = TimeSerie([3,1,3], name='b', fs=1, t0=1)
    lt_1 = (a >= b)
    assert compare_timeseries(lt_1, BooleanTimeSerie([False, True, True], t0=1, fs=1, name='(a >= b)'))
    lt_2 = (a >= 2)
    assert compare_timeseries(lt_2, BooleanTimeSerie([False, True, True], t0=1, fs=1, name=''))
    c = TimeSerie([7,8,9,0], name='c', fs=1, t0=0)
    with pytest.raises(ValueError):
        eq_3 = (a >= c)
    return True


def test_timeserie_and():
    a = TimeSerie([True,True,False,False], name='a', fs=1, t0=1)
    b = BooleanTimeSerie([True,False,True,False], name='b', fs=1, t0=1)
    c = TimeSerie([1,2,3,4], name='c', fs=1, t0=1)
    d = BooleanTimeSerie([True, True, True, True], name='d', fs=2, t0=3)
    and_1 = (a & b)
    assert compare_timeseries(and_1, BooleanTimeSerie([True, False, False, False], name="(a & b)", fs=1, t0=1))
    with pytest.raises(ValueError):
        and_2 = (a & c)
    with pytest.raises(ValueError):
        and_3 = (b & d)
    return True


def test_timeserie_or():
    a = TimeSerie([True,True,False,False], name='a', fs=1, t0=1)
    b = BooleanTimeSerie([True,False,True,False], name='b', fs=1, t0=1)
    c = TimeSerie([1,2,3,4], name='c', fs=1, t0=1)
    d = BooleanTimeSerie([True, True, True, True], name='d', fs=2, t0=3)
    and_1 = (a | b)
    assert compare_timeseries(and_1, BooleanTimeSerie([True, True, True, False], name="(a | b)", fs=1, t0=1))
    with pytest.raises(ValueError):
        and_2 = (a | c)
    with pytest.raises(ValueError):
        and_3 = (b | d)
    return True


def test_timeserie_xor():
    a = TimeSerie([True,True,False,False], name='a', fs=1, t0=1)
    b = BooleanTimeSerie([True,False,True,False], name='b', fs=1, t0=1)
    c = TimeSerie([1,2,3,4], name='c', fs=1, t0=1)
    d = BooleanTimeSerie([True, True, True, True], name='d', fs=2, t0=3)
    and_1 = (a ^ b)
    assert compare_timeseries(and_1, BooleanTimeSerie([False, True, True, False], name="(a ^ b)", fs=1, t0=1))
    with pytest.raises(ValueError):
        and_2 = (a ^ c)
    with pytest.raises(ValueError):
        and_3 = (b ^ d)
    return True


def test_timeserie_invert():
    a = TimeSerie([True,False], name='a', fs=1, t0=1)
    b = TimeSerie([1,2,3,4], name='b', fs=1, t0=1)
    invert_1 = ~a
    assert compare_timeseries(invert_1, BooleanTimeSerie([False,True], name='(~a)', fs=1, t0=1))
    with pytest.raises(ValueError):
        invert_2 = ~b
    return True


def test_timeserie_add():
    a = TimeSerie([1,2,3], name='a', fs=1, t0=1)
    b = TimeSerie([3,1,3], name='b', fs=1, t0=1)
    add_1 = a + b
    assert compare_timeseries(add_1, TimeSerie([4,3,6], name='(a + b)', fs=1, t0=1))
    add_2 = a + 1
    assert compare_timeseries(add_2, TimeSerie([2,3,4], name='', fs=1, t0=1))
    c = TimeSerie([7,8,9,0], name='c', fs=1, t0=0)
    with pytest.raises(ValueError):
        add_3 = (a + c)    
    return True


def test_timeserie_radd():
    a = TimeSerie([1,2,3], name='a', fs=1, t0=1)
    add_1 = 1 + a
    assert compare_timeseries(add_1, TimeSerie([2,3,4], name='', fs=1, t0=1))
    return True


def test_timeserie_sub():
    a = TimeSerie([1,2,3], name='a', fs=1, t0=1)
    b = TimeSerie([3,1,3], name='b', fs=1, t0=1)
    sub_1 = a - b
    assert compare_timeseries(sub_1, TimeSerie([-2,1,0], name='(a - b)', fs=1, t0=1))
    sub_2 = a - 1
    assert compare_timeseries(sub_2, TimeSerie([0,1,2], name='', fs=1, t0=1))
    c = TimeSerie([7,8,9,0], name='c', fs=1, t0=0)
    with pytest.raises(ValueError):
        sub_3 = (a - c)    
    return True


def test_timeserie_rsub():
    a = TimeSerie([1,2,3], name='a', fs=1, t0=1)
    sub_1 = 1 - a
    assert compare_timeseries(sub_1, TimeSerie([0, -1, -2], name='', fs=1, t0=1))
    return True


def test_timeserie_empty():
    a = TimeSerie.empty(1, 10, fs=10, name='a')
    assert len(a) == 90
    assert a.te == 9.9
    b = TimeSerie.empty(2, 5, 4, name='b', inclusive=True)
    assert len(b) == 13
    assert b.te == 5.0
    c = TimeSerie.empty(dt.datetime(2018,1,1,12), dt.datetime(2018,1,1,13), fs=1, name='c', inclusive=True)
    assert c.te - c.t0 == 3600.0
    return True


def test_timserie_mod():
    a = TimeSerie([-2, -1, 0, 1, 2, 3, 4, 5], name='a', fs=1, t0=1)
    b = a.modify("default", inplace=False)
    assert compare_timeseries(b, a)
    a.modify('default', inplace=True)
    assert compare_timeseries(a, b)

    a = TimeSerie([-2, -1, 0, 1, 2, 3, 4, 5], name='a', fs=1, t0=1)
    b = a.modify("zero_negatives", inplace=False)
    assert compare_timeseries(b, TimeSerie([0,0,0,1,2,3,4,5], name='a', fs=1, t0=1))
    a.modify('zero_negatives', inplace=True)
    assert compare_timeseries(a, b)

    a = TimeSerie([-2, -1, 0, 1, 2, 3, 4, 5], name='a', fs=1, t0=1)
    b = a.modify("correct_negatives", inplace=False)
    assert compare_timeseries(b, TimeSerie([0,1,2,3,4,5,6,7], name='a', fs=1, t0=1))
    a.modify('correct_negatives', inplace=True)
    assert compare_timeseries(a, b)
    return True


def test_timeserie_toevents():
    a = TimeSerie([1,2,3], t0=1, fs=2, name='a')
    events = a.to_events()
    # TODO: create assertions that check for correct events
    return True


def test_timeserie_tobool():
    a = TimeSerie([0,0,0,1,2,3], t0=1, fs=2, name='a')
    b = a.to_bool(inplace=False)
    assert compare_timeseries(b, BooleanTimeSerie([False, False, False, True, True, True], t0=1, fs=2, name='a'))
    a.to_bool(inplace=True)
    assert compare_timeseries(a, b)
    return True


def test__timeserie_tostatechangearray():
    a = TimeSerie([1,1,1,2,2,3,4,4,4], t0=1, fs=2, name='a')
    statechangearray = a.to_statechangearray()
    # TODO: create assertions that check for correct statechangearray
    return True


def test_timeserie_interpolate():
    a = TimeSerie([1,2,3,4,5,6,7,8,9,10], t0=0, fs=1, name='a')
    t_new = np.arange(0,9+0.5,0.5)
    b = a.interpolate(t_new, inplace=False)
    assert compare_timeseries(b, TimeSerie(np.arange(1,10.5, 0.5),t0=0, fs=2, name='a'))
    a.interpolate(t_new, inplace=True)
    assert compare_timeseries(a, b)
    c = BooleanTimeSerie([False,False,True,True,True,False,False], t0=1, fs=1, name='c')
    t_new = np.arange(1,7+0.5,0.5)
    d = c.interpolate(t_new, inplace=False)
    assert compare_timeseries(d, BooleanTimeSerie(3*[False] + 7*[True] + 3*[False],t0=1, fs=2, name='c'))
    c.interpolate(t_new, inplace=True)
    assert compare_timeseries(c, d)
    return True


def test_booleantimeserie_init():
    with pytest.raises(ValueError):
        a = BooleanTimeSerie([1,2,3], fs=1, t0=1, name='a')
    return True


def test_booleantimeserie_properties():
    a = BooleanTimeSerie([False,True,False], name='a', fs=2, t0=1)
    assert a.hz == a.fs
    a.hz = 1
    assert a.fs == 1
    b = BooleanTimeSerie([True,True,True], name='a', fs=1, t0=1)
    b.data = [False,True,False]
    assert isinstance(b.data, np.ndarray)
    assert all(b.data == [False,True,False])
    assert compare_timeseries(a,b)
    with pytest.raises(ValueError):
        a.data = [1,2,3]
    return True