import numpy as np
from HybridReader2 import HybridReader2 as hr, monotonic_step_iter
from FortranFile import NoMoreRecords
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from HybridHelper import parser, parse_cmd_line, data_slice, direct_plot

# Animation specific arguments
parser.add_argument('--framerate', type=int, default=20)
args = parse_cmd_line()

if args.single_fig:
    print "Combined plots no longer supported"
    print "Use --separate-figures, --xy, or --xz"
    exit(1)

def make_figures(args):
    ret = []
    if args.xy:
        ret.append(
            plt.subplots(subplot_kw={'aspect':'equal'})
        )
    if args.xz:
        ret.append(
            plt.subplots(subplot_kw={'aspect':'equal'})
        )
    return ret
        
# Setup simulation reader
h = hr(args.prefix,args.variable.name)

# Preload all the slices
print "Loading data into memory"
init_data = h.get_next_timestep()[-1]
if not h.isScalar:
    init_data = init_data[:,:,:,args.variable.coordinate]

data_slices = {d:[
    data_slice(h.para, init_data, d, coordinate=args.variable.coordinate)
] for d in args.directions}
while True:
    try:
        data = h.get_next_timestep()[-1]
    except NoMoreRecords:
        break
    #data = h.get_next_timestep()[-1]
    if not h.isScalar:
        data = data[:,:,:,args.variable.coordinate]

    for d in args.directions:
        s = data_slice(h.para, data, d, coordinate=args.variable.coordinate)
        data_slices[d].append(s)
n_frames = len(data_slices[args.directions[0]])

# Make the animations
for (fig, ax), d in zip(make_figures(args), args.directions):
    print 'Animating {} plot'.format(d)
    ax.set_xlabel(d[0])
    ax.set_ylabel(d[1])
    ax.set_ylim([-50,50])
    
    # Make the inital plot
    artist = direct_plot(fig, ax, init_data, h.para, d, cmap=args.colormap, norm=args.norm, vmin=args.vmin, vmax=args.vmax, mccomas=args.mccomas)[0]
    
    def update_animation(frame):
        s = data_slices[d][frame]

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

