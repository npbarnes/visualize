#!/home/nathan/anaconda2/envs/jupyterlab/bin/python
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
    return data[(cx-2,cx-1,cx,cx+1,cx+2),cy-2,cz+5]

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

def plot_each(ax, ts, arrs):
    for a, label in zip(arrs, ['-1000 m','-500 m','0 m','500 m','1000 m']):
        ax.plot(ts, a, label=label)

#######################################################################################################
rc('text', usetex=True)
q = 1.602e-19 # C
m = 16*1.6726e-27 # kg
q_over_m = q/m # C/kg

if __name__ == "__main__":
    ts1, total = timeseries_lowmemory('np_tot')
    ts2, oxygen = timeseries_lowmemory('np_H')
    ts3, barium = timeseries_lowmemory('np_CH4')
    ts4, B = timeseries_lowmemory('bt', mode='mag')
    ts5, Bx = timeseries_lowmemory('bt', mode='x')
    ts6, u = timeseries_lowmemory('up', mode='mag')
    ts7, E = timeseries_lowmemory('E', mode='mag')
    ts8, T_i = timeseries_lowmemory('temp_p')

    assert np.array_equal(ts1,ts2)
    assert np.array_equal(ts1,ts3)
    assert np.array_equal(ts1,ts4)
    assert np.array_equal(ts1,ts5)
    assert np.array_equal(ts1,ts6)
    assert np.array_equal(ts1,ts7)
    assert np.array_equal(ts1,ts8)
    ts = ts1

    # some profiles benefit from smoothing
    oxygen = smootherator(oxygen)
    T_i = smootherator(T_i)

    # B needs to be converted to nT, and E to micro V/m
    B  *= 1e9/q_over_m # nT
    Bx *= 1e9/q_over_m # nT
    E  *= 1e6/q_over_m # micro V/m

    mpkm = 1000

    #######################################################################################################
    fig, axs = plt.subplots(nrows=8, ncols=1, sharex=True, figsize=(8,8/7*10))
    fig.subplots_adjust(left=0.15, right=0.85, bottom=0.05, top=0.95, wspace=0, hspace=0)

    plot_each(axs[0], ts, total/mpkm**3)
    axs[0].set_ylabel('$n$\n$(\mathrm{m}^{-3})$', rotation=0, labelpad=20, multialignment='center')
    locs = axs[0].get_yticks()
    axs[0].set_yticklabels((f"${x:.1f} \\times 10^{{12}}$" for x in locs*1e-12))

    plot_each(axs[1], ts, oxygen/mpkm**3)
    axs[1].set_ylabel('$n_O$\n$(\mathrm{m}^{-3})$', rotation=0, labelpad=20, multialignment='center')
    locs = axs[1].get_yticks()
    axs[1].set_yticklabels((f"${x:.1f} \\times 10^{{12}}$" for x in locs*1e-12))

    plot_each(axs[2], ts, barium/mpkm**3)
    axs[2].set_ylabel('$n_{Ba}$\n$(\mathrm{m}^{-3})$', rotation=0, labelpad=20, multialignment='center')
    locs = axs[2].get_yticks()
    axs[2].set_yticklabels((f"${x:.1f} \\times 10^{{12}}$" for x in locs*1e-12))

    plot_each(axs[3], ts, B)
    axs[3].set_ylabel('$|B|$\n$(\mathrm{nT})$', rotation=0, labelpad=20, multialignment='center')

    plot_each(axs[4], ts, Bx)
    axs[4].set_ylabel('$B_x$\n$(\mathrm{nT})$', rotation=0, labelpad=20, multialignment='center')

    plot_each(axs[5], ts, u*mpkm)
    axs[5].set_ylabel('$|u|$\n$(\mathrm{m}/\mathrm{s})$', rotation=0, labelpad=20, multialignment='center')

    plot_each(axs[6], ts, E)
    axs[6].set_ylabel('$|E|$\n$(\mathrm{\mu V}/\mathrm{m})$', rotation=0, labelpad=20, multialignment='center')

    plot_each(axs[7], ts, T_i)
    axs[7].set_ylabel('$\mathrm{T}_i$\n(eV)', rotation=0, labelpad=20, multialignment='center')

    axs[0].set_title('Plasma properties timeseries')
    l = axs[2].legend(title='Offset\n(+ upstream)', bbox_to_anchor=(1.001,1), loc='upper left', fontsize=10)
    plt.setp(l.get_title(), multialignment='center')
    axs[-1].set_xlabel('seconds since release')
    plt.show()
