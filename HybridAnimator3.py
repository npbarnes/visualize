#!/usr/bin/env python
import numpy as np
from HybridReader2 import HybridReader2 as hr, monotonic_step_iter, equal_spacing_step_iter
from FortranFile import NoMoreRecords
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from HybridHelper import parser, parse_cmd_line, data_slice, direct_plot

# Animation specific arguments
parser.add_argument('--framerate', type=int, default=20)
args = parse_cmd_line()

if args.single_fig:
    print "Combined plots no longer supported"
    print "Use --separate-figures, --xy, --xz, or --yz"
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
print "Loading data into memory"
all_data = []
#for m, step_data in monotonic_step_iter(h):
for m, step_data in equal_spacing_step_iter(h):
    if not h.isScalar:
        step_data = step_data[:,:,:,args.variable.coordinate]
        if args.variable.name == 'bt':
            step_data = h.para['ion_amu'] * 1.6726219e-27/1.60217662e-19 * step_data # ion gyrofrequency -> nT
    all_data.append(step_data)
n_frames = len(all_data)

# Make the animations
for (fig, ax), d in zip(make_figures(args), args.directions):
    print 'Animating {} plot'.format(d)
    ax.set_xlabel(d[0])
    ax.set_ylabel(d[1])
    #ax.set_ylim([-50,50])
    
    if args.vmin is not None:
        vmin = args.vmin
    else:
        vmin = min((np.min(d) for d in all_data))
    if args.vmax is not None:
        vmax = args.vmax
    else:
        vmax = max((np.max(d) for d in all_data))
    # Make the inital plot
    artist = direct_plot(fig, ax, all_data[0], h.para, d, cmap=args.colormap, norm=args.norm, vmin=vmin, vmax=vmax, mccomas=args.mccomas)[0]
    annotation = ax.annotate(str(1),
            xy=(0.975, 0.975), xycoords='figure fraction',
            horizontalalignment='right', verticalalignment='top',
            fontsize=8)
    
    def update_animation(frame):
        s = data_slice(h.para, all_data[frame], d)

        # When changing the data for pcolormesh we need to remove the last element
        # in each direction for some reason.
        s = s[:-1, :-1]

        artist.set_array(s.T.ravel())
        annotation.set_text(str(frame))
        return artist,
    
    ani = animation.FuncAnimation(fig, frames=n_frames,
        func=update_animation, interval=100, blit=True)
    if h.isScalar:
        namestring = '{var}_{d}.mp4'.format(var=args.variable.name, d=d)
    else:
        namestring = '{var}_{coor}_{d}.mp4'.format(var=args.variable.name, coor=args.variable.coordinate, d=d)
    writer = animation.ImageMagickWriter(fps=12, bitrate=10000)
    ani.save(namestring, writer=writer)

