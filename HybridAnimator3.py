#!/usr/bin/env python3
import numpy as np
from HybridReader2 import HybridReader2 as hr, step_iter, monotonic_step_iter, equal_spacing_step_iter
from FortranFile import NoMoreRecords
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import HybridHelper
from HybridHelper import parser, parse_cmd_line, data_slice, direct_plot

# Animation specific arguments
parser.add_argument('--framerate', type=int, default=20)
parser.add_argument('--xy-scale', type=float, default=HybridHelper.Rp)
args = parse_cmd_line()
HybridHelper.Rp = args.xy_scale

if args.single_fig:
    print("Combined plots no longer supported")
    print("Use --separate-figures, --xy, --xz, or --yz")
    exit(1)

def make_figures(args):
    ret = []
    for _ in args.directions:
        ret.append(
            plt.subplots(subplot_kw={'aspect':'equal'})
        )
    return ret
        
# Setup simulation reader
h = hr(args.prefix,args.variable.name)

# Preload all the slices
print(f"Loading {args.variable.name} data into memory")
all_data = []
all_steps = []
for m, step_data in step_iter(h):
    if not h.isScalar:
        step_data = step_data[:,:,:,args.variable.coordinate]
    if args.variable.name == 'bt':
        step_data = 1e9 * h.para['ion_amu'] * 1.6726219e-27/1.60217662e-19 * step_data # ion gyrofrequency -> nT
    elif args.variable.name.startswith('np'):
        step_data *= (1e-3)**3 # km^-3 -> m^-3
    elif args.variable.name.startswith('up'):
        step_data *= 1e3 # km -> m
    all_data.append(step_data)
    all_steps.append(m)

dms = [mm - m for mm,m in zip(all_steps[1:], all_steps[:-1])]
n_frames = len(all_steps)
#n_frames = []
#for i, dm in enumerate(dms):
    #n_frames.extend([i] * dm)

# Make the animations
for (fig, ax), d in zip(make_figures(args), args.directions):
    print('Animating {} plot'.format(d))
    ax.set_xlabel('X (km)')
    ax.set_ylabel('Y (km)')
    #ax.set_ylim([-50,50])
    
    if args.vmax is not None:
        vmax = args.vmax
    else:
        vmax = max((np.max(d) for d in all_data))
    if args.vmin is not None:
        vmin = args.vmin
    else:
        vmin = min((np.min(d[d!=0], initial=vmax/2) for d in all_data))
    # Make the inital plot
    artist = direct_plot(fig, ax, all_data[0], h.para, d, cmap=args.colormap, norm=args.norm, vmin=vmin, vmax=vmax, mccomas=args.mccomas, skip_labeling=True)[0]
    annotation = ax.annotate("...Placeholder String...",
            xy=(0.975, 0.975), xycoords='figure fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=8, fontname='monospace')

    ax.set_xlim([-20,20])
    ax.set_ylim([-20,20])

    qx,qy,qz = h.para['grid_points']
    cx = int(len(qx)/2)
    cz = int(len(qz)/2)
    dots, = ax.plot([qx[cx-2],qx[cx-1],qx[cx],qx[cx+1],qx[cx+2]], 5*[qz[cz+5]], color='black', marker='o', markersize=0.7, linestyle='None')
    
    def update_animation(frame):
        if frame == update_animation.prev_frame:
            update_animation.prev_frame = frame
            return ()
        prev_frame = frame
        s = data_slice(h.para, all_data[frame], d)

        # When changing the data for pcolormesh we need to remove the last element
        # in each direction for some reason.
        s = s[:-1, :-1]

        artist.set_array(s.T.ravel())
        annotation.set_text(r"Simulated time $\approx$ {:>4.1f} s".format(all_steps[frame]*h.para['dt']))
        return artist,
    update_animation.prev_frame = -1
    
    ani = animation.FuncAnimation(fig, frames=n_frames,
        func=update_animation, interval=100, blit=True)
    if h.isScalar:
        namestring = '{var}_{d}.gif'.format(var=args.variable.name, d=d)
    else:
        namestring = '{var}_{coor}_{d}.gif'.format(var=args.variable.name, coor=args.variable.coordinate, d=d)
    writer = animation.ImageMagickWriter(fps=12, bitrate=10000)
    ani.save(namestring, writer=writer)

