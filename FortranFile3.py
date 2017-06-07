"""
Module to read / write Fortran unformatted sequential files.
This is in the spirit of code written by Neil Martinsen-Burrell and Joe Zuntz.
Modified from Scipy.
"""
import os
import numpy as np

class FortranFile:
    """
    A file object for unformatted sequential files from Fortran code.
    Parameters
    ----------
    filename : file or str
        Open file object or filename.
    mode : {'r', 'w'}, optional
        Read-write mode, default is 'r'.
    header_dtype : dtype, optional
        Data type of the header. Size and endiness must match the input/output file.
    Notes
    -----
    These files are broken up into records of unspecified types. The size of
    each record is given at the start (although the size of this header is not
    standard) and the data is written onto disk without any formatting. Fortran
    compilers supporting the BACKSPACE statement will write a second copy of
    the size to facilitate backwards seeking.
    This class only supports files written with both sizes for the record.
    It also does not support the subrecords used in Intel and gfortran compilers
    for records which are greater than 2GB with a 4-byte header.
    An example of an unformatted sequential file in Fortran would be written as::
        OPEN(1, FILE=myfilename, FORM='unformatted')
        WRITE(1) myvariable
    Since this is a non-standard file format, whose contents depend on the
    compiler and the endianness of the machine, caution is advised. Files from
    gfortran 4.8.0 and gfortran 4.1.2 on x86_64 are known to work.
    Consider using Fortran direct-access files or files from the newer Stream
    I/O, which can be easily read by `numpy.fromfile`.
    Examples
    --------
    To create an unformatted sequential Fortran file:
    >>> from scipy.io import FortranFile
    >>> f = FortranFile('test.unf', 'w')
    >>> f.write_record(np.array([1,2,3,4,5],dtype=np.int32))
    >>> f.write_record(np.linspace(0,1,20).reshape((5,-1)))
    >>> f.close()
    To read this file:
    >>> from scipy.io import FortranFile
    >>> f = FortranFile('test.unf', 'r')
    >>> print(f.read_ints(dtype=np.int32))
    [1 2 3 4 5]
    >>> print(f.read_reals(dtype=np.float).reshape((5,-1)))
    [[ 0.          0.05263158  0.10526316  0.15789474]
     [ 0.21052632  0.26315789  0.31578947  0.36842105]
     [ 0.42105263  0.47368421  0.52631579  0.57894737]
     [ 0.63157895  0.68421053  0.73684211  0.78947368]
     [ 0.84210526  0.89473684  0.94736842  1.        ]]
    >>> f.close()
    """
    def __init__(self, filename, mode='r', header_dtype=np.uint32):
        if header_dtype is None:
            raise ValueError('Must specify dtype')

        header_dtype = np.dtype(header_dtype)
        if header_dtype.kind != 'u':
            warnings.warn("Given a dtype which is not unsigned.")

        if mode not in 'rw' or len(mode) != 1:
            raise ValueError('mode must be either r or w')

        if hasattr(filename, 'seek'):
            self._fp = filename
        else:
            self._fp = open(filename, '%sb' % mode)

        self._header_dtype = header_dtype

    def _read_size(self):
        return int(np.fromfile(self._fp, dtype=self._header_dtype, count=1))

    def write_record(self, s):
        """
        Write a record (including sizes) to the file.
        Parameters
        ----------
        s : array_like
           The data to write.
        """
        s = np.array(s, order='F')
        np.array([s.nbytes],dtype=self._header_dtype).tofile(self._fp)
        s.tofile(self._fp)
        np.array([s.nbytes],dtype=self._header_dtype).tofile(self._fp)

    def read_record(self, dtype=None):
        """
        Reads a record of a given type from the file.
        Parameters
        ----------
        dtype : dtype, optional
            Data type specifying the size and endiness of the data.
        Returns
        -------
        data : ndarray
            A one-dimensional array object.
        Notes
        -----
        If the record contains a multi-dimensional array, calling reshape or
        resize will restructure the array to the correct size.
        Since Fortran multidimensional arrays are stored in column-major format,
        this may have some non-intuitive consequences. If the variable was
        declared as 'INTEGER var(5,4)', for example, var could be read with
        'read_record(dtype=np.integer).reshape( (4,5) )' since Python uses
        row-major ordering of indices.
        One can transpose to obtain the indices in the same order as in Fortran.
        For records that contain several variables or mixed types (as opposed
        to single scalar or array types), it is possible to specify a dtype
        with mixed types:
        >>> record = f.read_record([('a', '<f4'), ('b', '<i4')])
        >>> record['a']  # access the variable 'a'
        5.6
        and if any of the variables are arrays, the shape can be specified as
        the third item in the relevant tuple:
        >>> record = f.read_record([('a', '<f4'), ('b', '<i4', (3,3))])
        Numpy also supports a short syntax for this kind of type:
        >>> record = f.read_record('<f4,(3,3)<i4')
        >>> record['f0']  # variables are called f0, f1, ...
        5.6
        See Also
        --------
        read_reals
        read_ints
        """
        if dtype is None:
            raise ValueError('Must specify dtype')
        dtype = np.dtype(dtype)

        firstSize = self._read_size()
        if firstSize % dtype.itemsize != 0:
            raise ValueError('Size obtained ({0}) is not a multiple of the '
                             'dtype given ({1}).'.format(firstSize, dtype.itemsize))

        data = np.fromfile(self._fp, dtype=dtype, count=firstSize//dtype.itemsize)
        secondSize = self._read_size()
        if firstSize != secondSize:
            raise IOError('Sizes do not agree in the header and footer for '
                          'this record - check header dtype')
        return data

    def read_ints(self, dtype='i4'):
        """
        Reads a record of a given type from the file, defaulting to an integer
        type (INTEGER*4 in Fortran)
        Parameters
        ----------
        dtype : dtype, optional
            Data type specifying the size and endiness of the data.
        Returns
        -------
        data : ndarray
            A one-dimensional array object.
        See Also
        --------
        read_reals
        read_record
        """
        return self.read_record(dtype=dtype)

    def read_reals(self, dtype='f8'):
        """
        Reads a record of a given type from the file, defaulting to a floating
        point number (real*8 in Fortran)
        Parameters
        ----------
        dtype : dtype, optional
            Data type specifying the size and endiness of the data.
        Returns
        -------
        data : ndarray
            A one-dimensional array object.
        See Also
        --------
        read_ints
        read_record
        """
        return self.read_record(dtype=dtype)

    def close(self):
        """
        Closes the file. It is unsupported to call any other methods off this
        object after closing it. Note that this class supports the 'with'
        statement in modern versions of Python, to call this automatically
        """
        self._fp.close()

    def __enter__(self):
        return self

    def __exit__(self, type, value, tb):
        self.close()
        
    def readBackRecord(self, dtype=None):
        """Read the fortran record just before the current file position"""
        if self._fh.tell() < self._header_dtype.itemsize:
            raise IOError('There is no space for a record header before this position')

        self._fh.seek(-self._header_dtype.itemsize,os.SEEK_CUR)
        secondSize = self._read_size()
        if self._fh.tell() < -(2*self._header_dtype.itemsize + 1):
            raise IOError('There is no space for a record before this position')
        self._fh.seek(-(2*self._header_dtype.itemsize + l),os.SEEK_CUR)
        pos = self._fh.tell()

        data = self.read_record(dtype = dtype)

        self.seek(pos,os.SEEK_SET)
        return data

    def skipRecord(self):
        """Skip over a single fortran record"""
        firstSize = self._read_size()
        # seek has undefined behavior if seeking beyond the end of the file
        # we first check if there's enough room to seek, then seek.
        pos = self._fh.tell()
        self._fh.seek(0,os.SEEK_END)
        endpos = self._fh.tell()
        if pos+l > endpos:
            raise IOError('Not enough data to read')
        self._fh.seek(pos+l,os.SEEK_SET)
        secondSize = self._read_size()
        if firstSize != secondSize:
            raise IOError('Sizes do not agree in the header and footer for '
                          'this record - check header dtype')

######### TODO: Fix compatibility with all the crap below this line ##############
    def skipBackRecord(self):
        if self._fh.tell() < self._header_dtype.itemsize:
            raise IOError('Error reading record from data file')
        self._fh.seek(-self._header_dtype.itemsize,os.SEEK_CUR)
        secondSize = self._read_size()
        self.seek(-(2*self._header_dtype.itemsize + secondSize),os.SEEK_CUR)
        pos = self.tell()
        check_size = self._read_check()
        if check_size != l:
            raise IOError('Error reading record from data file')
        self.seek(pos,os.SEEK_SET)

    def readBackReals(self, prec='f'):
        """Read in an array of real numbers from before the current file position.

        Parameters
        ----------
        prec : character, optional
            Specify the precision of the array using character codes from
            Python's struct module.  Possible values are 'd' and 'f'.

        """

        _numpy_precisions = {'d': numpy.float64,
                             'f': numpy.float32
                            }

        if prec not in self._real_precisions:
            raise ValueError('Not an appropriate precision')

        data_str = self.readBackRecord()
        return numpy.fromstring(data_str, dtype=_numpy_precisions[prec])

    def readBackInts(self, prec='i'):
        """Read an array of integers.

        Parameters
        ----------
        prec : character, optional
            Specify the precision of the data to be read using
            character codes from Python's struct module.  Possible
            values are 'h', 'i', 'l' and 'q'

        """
        if prec not in self._int_precisions:
            raise ValueError('Not an appropriate precision')

        data_str = self.readBackRecord()
        return numpy.fromstring(data_str, dtype=self.ENDIAN+prec)
