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
    all_data.append(step_data)
n_frames = len(all_data)

# Make the animations
for (fig, ax), d in zip(make_figures(args), args.directions):
    print 'Animating {} plot'.format(d)
    ax.set_xlabel(d[0])
    ax.set_ylabel(d[1])
    ax.set_ylim([-50,50])
    
    # Make the inital plot
    artist = direct_plot(fig, ax, all_data[0], h.para, d, cmap=args.colormap, norm=args.norm, vmin=args.vmin, vmax=args.vmax, mccomas=args.mccomas)[0]
    
    def update_animation(frame):
        s = data_slice(h.para, all_data[frame], d)

        # When changing the data for pcolormesh we need to remove the last element
        # in each direction for some reason.
        s = s[:-1, :-1]

        artist.set_array(s.T.ravel())
        return artist,
    
    ani = animation.FuncAnimation(fig, frames=n_frames,
        func=update_animation, interval=100, blit=True)
    if h.isScalar:
        namestring = '{var}_{d}.gif'.format(var=args.variable, d=d)
    else:
        namestring = '{var}_{coor}_{d}.gif'.format(var=args.variable, coor=args.coordinate, d=d)
    ani.save(namestring, writer='imagemagick')

