#!/usr/bin/python
import FortranFile as ff
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
from HybridReader2 import HybridReader2 as hr
from matplotlib.colors import Normalize, LogNorm, SymLogNorm
import matplotlib.animation as animation
from HybridHelper import parser, parse_cmd_line, get_next_slice, get_next_beta_slice, init_figures, direct_plot, beta_plot

import logging
logger = logging.getLogger('matplotlib')
logger.setLevel(logging.INFO)

#matplotlib.verbose.set_level('helpful')
#plt.rcParams['animation.convert_path'] = '/usr/local/pkg/vis/ImageMagick/7.0.8-11-pic-intel-2016b/bin/convert'

# Tools for later
class StoppableFrames:
    """If something external sets the done_playing atribute to True, then it will raise a StopIteration on the next call to next"""
    def __init__(self):
        self.done_playing = False

    def __iter__(self):
        return self

    def next(self):
        if self.done_playing:
            raise StopIteration
        else:
            return self

# Initial setup
parser.add_argument('--framerate', type=int, default=20)
args = parse_cmd_line()
fig1, fig2, ax1, ax2 = init_figures(args)
if args.separate:
    raise NotImplementedError("Separate figures are not implemented for animations")

# Set title, labels and limits
if args.fontsize is not None:
    fig1.suptitle(args.variable, fontsize=1.5*args.fontsize)
    if args.separate:
        fig2.suptitle(args.variable, fontsize=1.5*args.fontsize)
else:
    fig1.suptitle(args.variable, fontsize=14)
    if args.separate:
        fig2.suptitle(args.variable, fontsize=14)
ax1.set_xlabel('X')
ax1.set_ylabel('Y')
ax2.set_xlabel('X')
ax2.set_ylabel('Z')

ax1.set_ylim([-50,50])
ax2.set_ylim([-50,50])

if args.variable.name == 'beta':
    hn = hr(args.prefix, 'np')
    hT = hr(args.prefix, 'temp_p')
    hB = hr(args.prefix, 'bt')

    para = hn.para

    n = hn.get_next_timestep()[-1]
    T = hT.get_next_timestep()[-1]
    B = hB.get_next_timestep()[-1]

    # Convert units
    n = n/(1000.0**3)                    # 1/km^3 -> 1/m^3
    T = 1.60218e-19 * T                  # eV -> J
    B = 1.6726219e-27/1.60217662e-19 * B # proton gyrofrequency -> T

    # Compute B \cdot B
    B2 = np.sum(B**2, axis=-1)

    # Compute plasma beta
    data = n*T/(B2/(2*1.257e-6))

    artist_xy = beta_plot(fig1, ax1, data, para, 'xy', fontsize=args.fontsize, mccomas=args.mccomas, refinement=args.refinement)
    artist_xz = beta_plot(fig2, ax2, data, para, 'xz', fontsize=args.fontsize, mccomas=args.mccomas, refinement=args.refinement)

    def update_animation(frame):
        try:
            n = hn.get_next_timestep()[-1]
        except ff.NoMoreRecords:
            frame.done_playing = True
            return []
        T = hT.get_next_timestep()[-1]
        B = hB.get_next_timestep()[-1]

        # Convert units
        n = n/(1000.0**3)                    # 1/km^3 -> 1/m^3
        T = 1.60218e-19 * T                  # eV -> J
        B = 1.6726219e-27/1.60217662e-19 * B # proton gyrofrequency -> T

        # Compute B \cdot B
        B2 = np.sum(B**2, axis=-1)

        # Compute plasma beta
        data = n*T/(B2/(2*1.257e-6))

        artist_xy = beta_plot(fig1, ax1, data, para, 'xy', fontsize=args.fontsize, mccomas=args.mccomas, refinement=args.refinement, cax='None')
        artist_xz = beta_plot(fig2, ax2, data, para, 'xz', fontsize=args.fontsize, mccomas=args.mccomas, refinement=args.refinement, cax='None')

        # The object retuned by contourf is not actually an artist
        return artist_xy.collections + artist_xz.collections
else:
    # Get starting data
    h = hr(args.prefix,args.variable.name)
#    for i in range(193):
#        h.skip_next_timestep()
    data = h.get_next_timestep()[-1]
    if not h.isScalar:
        data = data[:,:,:,args.variable.coordinate]
    para = h.para

    # Make initial plots
    artist_xy = direct_plot(fig1, ax1, data, para, 'xy', cmap=args.colormap, norm=args.norm, vmin=args.vmin, vmax=args.vmax, mccomas=args.mccomas)[0]
    artist_xz = direct_plot(fig2, ax2, data, para, 'xz', cmap=args.colormap, norm=args.norm, vmin=args.vmin, vmax=args.vmax, mccomas=args.mccomas)[0]

    def update_animation(frame):
        try:
            xy_slice = get_next_slice(h, 'xy', coordinate=args.variable.coordinate)
            xz_slice = get_next_slice(h, 'xz', coordinate=args.variable.coordinate)
        except ff.NoMoreRecords:
            frame.done_playing = True
            return artist_xy, artist_xz
        # When changing the data for pcolormesh we need to remove the last element
        # in each direction for some reason.
        xy_slice = xy_slice[:-1, :-1]
        xz_slice = xz_slice[:-1, :-1]

        artist_xy.set_array(xy_slice.T.ravel())
        artist_xz.set_array(xz_slice.T.ravel())
        return artist_xy, artist_xz

if args.save:
    ani = animation.FuncAnimation(fig1, frames=StoppableFrames(),
                    func=update_animation, interval=1, blit=True, repeat=False)
    plt.show()
    #ani.save(args.save + '.mp4', fps=args.framerate, bitrate=5000, writer='ffmpeg')
    print(args.save)
    ani.save(args.save + '.gif', writer='imagemagick')
    #writer = animation.ImageMagickFileWriter(fps=args.framerate, bitrate=5000)
    ani.save(args.save + '.gif', writer=writer)
else:
    raise RuntimeError("You must save the animation with the --save flag and view separately. Directly viewing is not supported.")
