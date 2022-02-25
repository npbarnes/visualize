#!/usr/bin/env python
import os
import sys
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from HybridReader2 import HybridReader2 as hr
from HybridReader2 import step_iter
import argparse

# Load data at a few gridpoints as a timeseries
def getbobs(data, zindex):
    s = data.shape
    nx,ny,nz = s[0],s[1],s[2]
    cx = nx//2
    cy = ny//2
    cz = nz//2
    return data[(cx-2,cx-1,cx,cx+1,cx+2),cy,cz+zindex]

def point_selection(data, mode, zindex):
    b = getbobs(data, zindex)

    if mode is None:
        return b.copy()
    elif mode == 'mag':
        return np.linalg.norm(b, axis=-1)
    elif mode == 'x':
        return b[:,0].copy()
    elif mode == 'y':
        return b[:,1].copy()
    elif mode == 'z':
        return b[:,2].copy()
    else:
        raise ValueError("mode must be None, 'mag', 'x', 'y', or 'z'.")

def loadarray_fromhybrid(var, mode, zindex):
    h = hr('data', var)
    if h.isScalar:
        assert mode is None
    else:
        assert mode in ('mag', 'x', 'y', 'z')
    ret_lst = []
    ts = []
    for m,data in step_iter(h):
        ts.append(h.para['dt']*m)
        ret_lst.append(point_selection(data, mode, zindex))
    return np.asarray(ts), np.stack(ret_lst, axis=-1)

def unique(ar):
    """Similar to numpy.unique with return_index=True.
    This function returns the unique values and last indices where a value occurs (rather than 
    the first indices). There's no option to not return indices, just use np.unique in that case.
    There's no option for inverse or counts, sorry.

    Needed because whenever a restart happens in the simulation there are some timesteps that
    get calculated and saved twice, but because of the randomless built into the simulation
    they don't come out exactly the same. I chose to always use the lastest version so that the
    line is continuous.
    """
    ar = np.asanyarray(ar)
    perm = ar.argsort(kind='stable')
    aux = ar[perm]
    mask = np.empty(aux.shape, dtype=bool)
    mask[-1] = True
    mask[:-1] = aux[:-1] != aux[1:]
    return aux[mask], perm[mask]

# Utils for working with multiple timeseries
class TimeSeries:
    def __init__(self, *args):
        if len(args) == 1:
            self.raw_ts = args[0].pop("ts").flatten()
            self.raw_data = args[0]
        elif len(args) == 2:
            self.raw_ts = args[0]
            self.raw_data = args[1]
        else:
            raise TypeError("__init__() requires either 1 or 2 positional aruments.")
        self.ts, idxs = unique(self.raw_ts)
        self.data = {k:self.raw_data[k][:, idxs] for k in self.raw_data}

    def call_func(self, f):
        result = {}
        for k in self.data:
            result[k] = f(self.data[k])
        return result

    def update_withfunc(self, f):
        for k in self.data:
            self.data[k] = f(self.data[k])

    def __getitem__(self, var_str):
        return self.data[var_str]

    def __setitem__(self, key, val):
        self.data[key] = val

    def save(self, filename="timeseries.npz"):
        np.savez(filename, ts=self.ts, **self.data)

    def __iter__(self):
        return iter(self.data.values())

def load_fromfile(filename="timeseries.npz"):
    data = dict(np.load(filename))
    return TimeSeries(data)

def convert_units(series, specification):
    q = 1.602e-19 # C
    m = 16*1.6726e-27 # kg
    q_over_m = q/m # C/kg
    mpkm = 1000

    Bs = [name for name, var, _ in specification if var.startswith('bt')]
    for B_name in Bs:
        series[B_name] *= 1e9/q_over_m # nT

    Es = [name for name, var, _ in specification if var == 'E']
    for E_name in Es:
        series[E_name] *= 1e6/q_over_m #micro V/m

    ns = [name for name, var, _ in specification if var.startswith('np')]
    for n_name in ns:
        series[n_name] /= mpkm**3 # m^-3

    us = [name for name, var, _ in specification if var.startswith('up')]
    for u_name in us:
        series[u_name] *= mpkm # m/s

    return series

def loadtimeseries_fromhybrid(specification, zindex):
    name1, var1, mode1 = specification.pop()
    ts1, arr1 = loadarray_fromhybrid(var1, mode1, zindex)
    data_dict = {name1: arr1}
    for name, var, mode in specification:
        ts, arr = loadarray_fromhybrid(var, mode, zindex)
        assert np.array_equal(ts1, ts)
        data_dict[name] = arr

    return convert_units(TimeSeries(ts1, data_dict), specification)

# Plotting utils
def smootherator(arrs, n=3):
    ret = np.empty_like(arrs)
    for i, arr in enumerate(arrs):
        ret[i,0] = (arr[0]+arr[1])/2
        ret[i,1:-1] = np.convolve(arr, np.ones((n,))/n, mode='valid')
        ret[i,-1] = (arr[-2] + arr[-1])/2
    arrs[:,:] = ret
    return arrs

#rc('text', usetex=True)

def plot_each_line(ax, ts, arrs):
    for a, label in zip(arrs, ['-1000 m','-500 m','0 m','500 m','1000 m']):
        ax.plot(ts, a, label=label)

def fig_axs_setup_simple(spec):
    fig, axs = plt.subplots(nrows=len(spec), ncols=1, sharex=True, figsize=(8.5,11))
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05, top=0.96, wspace=0, hspace=0.25)

    axs[0].set_title('Plasma properties timeseries')

    for ax, (name, _, _) in zip(axs, spec):
        ax.set_ylabel(name)

    axs[-1].set_xlabel('seconds since release')

    return fig, axs

def fig_axs_setup_fancy1():
    fig, axs = plt.subplots(nrows=8, ncols=1, sharex=True, figsize=(8.5,11))
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05, top=0.96, wspace=0, hspace=0.25)

    axs[0].set_title('Plasma properties timeseries')

    axs[0].set_ylabel('$n$\n$(\mathrm{m}^{-3})$', rotation=0, labelpad=20, multialignment='center')
    axs[1].set_ylabel('$n_O$\n$(\mathrm{m}^{-3})$', rotation=0, labelpad=20, multialignment='center')
    axs[2].set_ylabel('$n_{Ba}$\n$(\mathrm{m}^{-3})$', rotation=0, labelpad=20, multialignment='center')
    axs[3].set_ylabel('$|B|$\n$(\mathrm{nT})$', rotation=0, labelpad=20, multialignment='center')
    axs[4].set_ylabel('$B_x$\n$(\mathrm{nT})$', rotation=0, labelpad=20, multialignment='center')
    axs[5].set_ylabel('$|u|$\n$(\mathrm{m}/\mathrm{s})$', rotation=0, labelpad=20, multialignment='center')
    axs[6].set_ylabel('$|E|$\n$(\mathrm{\mu V}/\mathrm{m})$', rotation=0, labelpad=20, multialignment='center')
    axs[7].set_ylabel('$\mathrm{T}_i$\n(eV)', rotation=0, labelpad=20, multialignment='center')

    axs[-1].set_xlabel('seconds since release')

    return fig, axs

def plot_each_variable(fig, axs, spec, series):
    for ax, (name, _, _) in zip(axs, spec):
        plot_each_line(ax, series.ts, series[name])

    l = axs[2].legend(title='Approx Offset\n(+ upstream)', bbox_to_anchor=(1.001,1), loc='upper left', fontsize=10)
    plt.setp(l.get_title(), multialignment='center')

parser = argparse.ArgumentParser()
parser.add_argument('--load', choices=('hybrid', 'npz'), default='npz')
parser.add_argument('--path', default='timeseries.npz')
parser.add_argument('--no-show', dest='show', action='store_false')
parser.add_argument('--zindex', type=int, default=5)

if __name__ == "__main__":
    spec = [
        ('total', 'np_tot', None),
        ('oxygen', 'np_H', None),
        ('barium', 'np_CH4', None),
        ('B', 'bt', 'mag'),
        ('Bx', 'bt', 'x'),
        ('u', 'up_tot', 'mag'),
        ('E', 'E', 'mag'),
        ('T_i', 'temp_p', None)
    ]
    spec = [
        ('total', 'np_tot', None),
        ('oxygen', 'np_H', None),
        ('barium', 'np_CH4', None),
        ('B', 'bt', 'mag'),
        ('Bx', 'bt', 'x'),
        ('u', 'up_tot', 'mag'),
        ('Ex', 'E', 'x'),
        ('Ey', 'E', 'y'),
        ('Ez', 'E', 'z'),
        ('T_i', 'temp_p', None)
    ]
    args = parser.parse_args()
    if args.load == 'hybrid':
        series = loadtimeseries_fromhybrid(spec, args.zindex)
        series.save(args.path)
    elif args.load == 'npz':
        series = load_fromfile(args.path)

    if args.show:
        fig, axs = fig_axs_setup_simple(spec)
        plot_each_variable(fig, axs, spec, series)
        plt.show()

