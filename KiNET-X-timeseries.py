#!/usr/bin/env python
import sys
import numpy as np
from matplotlib import rc
import matplotlib.pyplot as plt
from HybridReader2 import HybridReader2 as hr
from HybridReader2 import step_iter

def getbobs(data):
    s = data.shape
    nx,ny,nz = s[0],s[1],s[2]
    cx = nx//2
    cy = ny//2
    cz = nz//2
    return data[(cx-2,cx-1,cx,cx+1,cx+2),cy,cz+5]

def point_selection(data, mode):
    if mode is None:
        return getbobs(data).copy()
    elif mode == 'mag':
        return np.linalg.norm(getbobs(data), axis=-1)
    elif mode == 'x':
        return getbobs(data)[:,0].copy()
    elif mode == 'y':
        return getbobs(data)[:,1].copy()
    elif mode == 'z':
        return getbobs(data)[:,2].copy()
    else:
        raise ValueError("mode must be None, 'mag', 'x', 'y', or 'z'.")

def timeseries_lowmemory(var, mode=None):
    h = hr('data', var)
    if h.isScalar:
        assert mode is None
    else:
        assert mode in ('mag', 'x', 'y', 'z')
    ret_lst = []
    ts = []
    for m,data in step_iter(h):
        ts.append(h.para['dt']*m)
        ret_lst.append(point_selection(data, mode))
    return ts, np.stack(ret_lst, axis=-1)

def smootherator(arrs, n=3):
    ret = np.empty_like(arrs)
    for i, arr in enumerate(arrs):
        ret[i,0] = (arr[0]+arr[1])/2
        ret[i,1:-1] = np.convolve(arr, np.ones((n,))/n, mode='valid')
        ret[i,-1] = (arr[-2] + arr[-1])/2
    return ret
def smootherator_inplace(arrs, n=3):
    ret = np.empty_like(arrs)
    for i, arr in enumerate(arrs):
        ret[i,0] = (arr[0]+arr[1])/2
        ret[i,1:-1] = np.convolve(arr, np.ones((n,))/n, mode='valid')
        ret[i,-1] = (arr[-2] + arr[-1])/2
    arrs[:,:] = ret
    return arrs

def plot_each(ax, ts, arrs):
    for a, label in zip(arrs, ['-1000 m','-500 m','0 m','500 m','1000 m']):
        ax.plot(ts, a, label=label)

#######################################################################################################
rc('text', usetex=True)
q = 1.602e-19 # C
m = 16*1.6726e-27 # kg
q_over_m = q/m # C/kg

def fig_axs_setup():
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

def plot_all(fig, axs, ts, timeseries_lst):
    for ax, timeseries in zip(axs, timeseries_lst):
        plot_each(ax, ts, timeseries)

    l = axs[2].legend(title='Approx Offset\n(+ upstream)', bbox_to_anchor=(1.001,1), loc='upper left', fontsize=10)
    plt.setp(l.get_title(), multialignment='center')

    plt.show()

def unique(ar):
    """Similar to numpy.unique with return_index=True.
    This function returns the unique values and last indices where a value occurs (rather than 
    the first indices). There's no option to not return indices, just use np.unique in that case.
    There's no option for inverse or counts, sorry.
    """
    perm = ar.argsort(kind='stable')
    aux = ar[perm]
    mask = np.empty(aux.shape, dtype=bool)
    mask[-1] = True
    mask[:-1] = aux[:-1] != aux[1:]
    return aux[mask], perm[mask]

def load_and_prep():
    data = np.load("timeseries.npz")
    ts = data["ts"]
    total = data["total"]
    oxygen = data["oxygen"]
    barium = data["barium"]
    B = data["B"]
    Bx = data["Bx"]
    u = data["u"]
    E = data["E"]
    T_i = data["T_i"]

    ts = ts.reshape(len(ts))
    ts,idxs = unique(ts)
    total  = total[:, idxs]
    oxygen = oxygen[:, idxs]
    barium = barium[:, idxs]
    B      = B[:, idxs]
    Bx     = Bx[:, idxs]
    u      = u[:, idxs]
    E      = E[:, idxs]
    T_i    = T_i[:, idxs]

    smootherator_inplace(total)
    smootherator_inplace(oxygen)
    smootherator_inplace(barium)
    smootherator_inplace(B)
    smootherator_inplace(Bx)
    smootherator_inplace(u)
    smootherator_inplace(E)
    smootherator_inplace(T_i)

    # Units
    B  *= 1e9/q_over_m # nT
    Bx *= 1e9/q_over_m # nT
    E  *= 1e6/q_over_m # micro V/m

    mpkm = 1000
    total  /= mpkm**3 # m^-3
    oxygen /= mpkm**3 # m^-3
    barium /= mpkm**3 # m^-3
    u      *= mpkm    # m/s

    return ts, total, oxygen, barium, B, Bx, u, E, T_i

if __name__ == "__main__":
    fig, axs = fig_axs_setup()
    ts, *timeseries_lst = load_and_prep()
    plot_all(fig, axs, ts, timeseries_lst)
