import numpy as np
import FortranFile3 as ff
from sys import argv
from os import listdir
from os.path import isfile, join
import re
from HybridParams import HybridParams as hp



def tryint(s):
    try:
        return int(s)
    except ValueError:
        return s

def alphanum_key(s):
    """ Turn a string into a list of string and number chunks.
        "z23a" -> ["z", 23, "a"]
    """
    return [ tryint(c) for c in re.split('([0-9]+)', s) ]

def sort_nicely(l):
    """ Sort the given list in the way that humans expect.
    >>> sort_nicely(['b2', 'b10', 'a2', 'a10'])
    ['a2','a10','b2','b10']
    """
    l.sort(key=alphanum_key)
    return l


class outflows:
    def __init__(self, prefix):
        self.beta = hp(prefix).para['beta']
        prefix = join(prefix, 'particle')
        self.files = [ ff(f) for f in sort_nicely(listdir(prefix)) if 'outflowing' in f ]


    def __iter__(self):
        return self

    def __next__(self):
        mrat = []
        beta_p = []
        tags = []
        for f in self.files:
            mrat.extend(   list(f.read_reals()) )
            beta_p.extend( list(f.read_reals()) )
            tags.extend(   list(f.read_ints())  )

        for m,b,t in zip(mrat,beta_p,tags):
            if t == 1:
                mass += m*self.munit/(b*self.beta)

        return mass


o = outflows(argv[1])
