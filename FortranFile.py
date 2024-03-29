# Copyright 2008-2010 Neil Martinsen-Burrell
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in
# all copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
# THE SOFTWARE.

# Modified 2016-2021 by Nathan Barnes, University of Alaska Fairbanks

"""Defines a file-derived class to read/write Fortran unformatted files.

The assumption is that a Fortran unformatted file is being written by
the Fortran runtime as a sequence of records.  Each record consists of
an integer (of the default size [usually 32 or 64 bits]) giving the
length of the following data in bytes, then the data itself, then the
same integer as before.

Examples
--------

To use the default precision settings, one can just do::
    >>> f = FortranFile('filename')
    >>> x = f.readReals()

One can read arrays with varying precisions::
    >>> f = FortranFile('filename')
    >>> x = f.readInts('h')
    >>> y = f.readInts('q')
    >>> z = f.readReals('f')
Where the format codes are those used by Python's struct module.

One can change the default header precision::
    >>> f = FortranFile('filename', header_prec='l')
for a file whose record headers are long
integers.
"""

__docformat__ = "restructuredtext en"

import numpy
import os
import io

class IntegrityError(Exception):
    pass

class NoMoreRecords(Exception):
    pass

class FortranFile(io.FileIO):

    """File with methods for dealing with fortran unformatted data files"""

    def _get_header_length(self):
        return numpy.dtype(self._header_prec).itemsize
    _header_length = property(fget=_get_header_length)

    def _set_header_prec(self, prec):
        if prec in 'hilq':
            self._header_prec = prec
        else:
            raise ValueError('Cannot set header precision')

    def _get_header_prec(self):
        return self._header_prec
    HEADER_PREC = property(fset=_set_header_prec,
                           fget=_get_header_prec,
                           doc="Possible header precisions are 'h', 'i', 'l', 'q'"
                          )

    def __init__(self, fname, header_prec='i', *args, **kwargs):
        """Open a Fortran unformatted file for writing.

        Parameters
        ----------
        header_prec : character, optional
            Specify the precision used for the record headers.  Possible
            values are 'h', 'i', 'l' and 'q' with their meanings from
            Python's struct module.  The default is 'i' (the system's
            default integer).

        """
        super(FortranFile, self).__init__(fname,  *args, **kwargs)
        self.HEADER_PREC = header_prec

    def _read_data(self, num_bytes):
        """Read in exactly num_bytes, raising an error if it can't be done."""
        data = self.read(num_bytes)
        l = len(data)
        if l < num_bytes:
            raise IntegrityError('Could not read enough data. Wanted %d bytes, got %d.' % (num_bytes, l))

        return data

    def _read_leading_indicator(self):
        indicator_str = self.read(self._header_length)
        if len(indicator_str) == 0:
            raise NoMoreRecords(f"Tried to read a record past the end of the file ({self.name}).")
        if len(indicator_str) < self._header_length:
            raise IntegrityError('Could not read the leading size indicator. Not enough bytes.')
        indicator = numpy.fromstring(indicator_str,
                                dtype=self.HEADER_PREC
                               )[0]
        if indicator < 0:
            raise IntegrityError('Invalid leading size indicator. Sizes must be non-negative.')

        return indicator

    def _read_trailing_indicator(self):
        indicator_str = self.read(self._header_length)
        if len(indicator_str) < self._header_length:
            raise IntegrityError('Could not read the trailing size indicator. Not enough bytes.')
        indicator = numpy.fromstring(indicator_str,
                                dtype=self.HEADER_PREC
                               )[0]

        if indicator < 0:
            raise IntegrityError('Invalid trailing size indicator. Sizes must be non-negative.')

        return indicator

    def _write_check(self, number_of_bytes):
        """Write the header for the given number of bytes"""
        self.write(numpy.array(number_of_bytes,
                               dtype=self.HEADER_PREC,).tostring()
                  )

    def readRecord(self):
        """Read a single fortran record"""
        l = self._read_leading_indicator()
        data_str = self._read_data(l)
        check_size = self._read_trailing_indicator()
        if check_size != l:
            raise IntegrityError('Leading size indicator (%d bytes) does not match trailing size'
                                 ' indicator (%d bytes).' % (l, check_size))
        return data_str

    def readBackRecord(self):
        """Read the fortran record just before the current file position"""
        if self.tell() == 0:
            raise NoMoreRecords(f"Tried to read a record before the beginning of the file ({self.name}).")
        if self.tell() < 2*self._header_length:
            raise IntegrityError('Not enough space for any record before this position.')
        self.seek(-self._header_length,os.SEEK_CUR)
        l = self._read_trailing_indicator()
        if self.tell() < 2*self._header_length + l:
            raise IntegrityError('Not enough space for a record of the indicated size before this position.')
        self.seek(-(2*self._header_length + l),os.SEEK_CUR)
        pos = self.tell()
        check_size = self._read_leading_indicator()
        if check_size != l:
            raise IntegrityError('Leading size indicator (%d bytes) does not match trailing size'
                                 ' indicator (%d bytes).' % (check_size, l))
        data_str = self._read_data(l)
        self.seek(pos,os.SEEK_SET)
        return data_str

    def skipRecord(self):
        """Skip over a single fortran record"""
        l = self._read_leading_indicator()
        pos = self.tell()
        self.seek(0,os.SEEK_END)
        endpos = self.tell()
        if pos+l > endpos:
            raise IntegrityError('Not enough data left in the file for another record.'
                                 ' Need %d bytes, only %d available.' % (2*self._header_length + l,
                                                                        endpos-pos+self._header_length))
        self.seek(pos+l,os.SEEK_SET)
        check_size = self._read_trailing_indicator()

        if check_size != l:
            raise IntegrityError('Leading size indicator (%d bytes) does not match trailing size'
                                 ' indicator (%d bytes).' % (l, check_size))

    def skipBackRecord(self):
        if self.tell() == 0:
            raise NoMoreRecords(f"Tried to skip a record before the start of the file ({self.name}).")
        if self.tell() < 2*self._header_length:
            raise IntegrityError('Not enough space for any record before this position.')
        self.seek(-self._header_length,os.SEEK_CUR)
        l = self._read_trailing_indicator()
        if self.tell() < 2*self._header_length + l:
            raise IntegrityError('Not enough space for a record of the indicated size before this position.')
        self.seek(-(2*self._header_length + l),os.SEEK_CUR)
        pos = self.tell()
        check_size = self._read_leading_indicator()
        if check_size != l:
            raise IntegrityError('Leading size indicator (%d bytes) does not match trailing size'
                                 ' indicator (%d bytes).' % (check_size, l))

        self.seek(pos,os.SEEK_SET)

    def writeRecord(self,s):
        """Write a record with the given bytes.

        Parameters
        ----------
        s : the string to write

        """
        length_bytes = len(s)
        self._write_check(length_bytes)
        self.write(s)
        self._write_check(length_bytes)

    def readString(self):
        """Read a string."""
        return self.readRecord()

    def writeString(self,s):
        """Write a string

        Parameters
        ----------
        s : the string to write

        """
        self.writeRecord(s)

    _real_precisions = 'df'

    def readReals(self, prec='f'):
        """Read in an array of real numbers.

        Parameters
        ----------
        prec : character, optional
            Specify the precision of the array using character codes from
            Python's struct module.  Possible values are 'd' and 'f'.

        """

        if prec not in self._real_precisions:
            raise ValueError('Not an appropriate precision')

        data_str = self.readRecord()
        return numpy.fromstring(data_str, dtype=prec)

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

    def writeReals(self, reals, prec='f'):
        """Write an array of floats in given precision

        Parameters
        ----------
        reals : array
            Data to write
        prec` : string
            Character code for the precision to use in writing
        """
        if prec not in self._real_precisions:
            raise ValueError('Not an appropriate precision')

        nums = numpy.array(reals, dtype=prec)
        self.writeRecord(nums.tostring())

    _int_precisions = 'hilq'

    def readInts(self, prec='i'):
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

        data_str = self.readRecord()
        return numpy.fromstring(data_str, dtype=prec)

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
        return numpy.fromstring(data_str, dtype=prec)

    def writeInts(self, ints, prec='i'):
        """Write an array of integers in given precision

        Parameters
        ----------
        reals : array
            Data to write
        prec : string
            Character code for the precision to use in writing
        """
        if prec not in self._int_precisions:
            raise ValueError('Not an appropriate precision')

        nums = numpy.array(ints, dtype=prec)
        self.writeRecord(nums.tostring())

    def readOther(self, dtype):
        data_str = self.readRecord()
        dtype = numpy.dtype(dtype)
        #if len(data_str) % dtype.itemsize != 0:
            #raise IntegrityError
        return numpy.fromstring(data_str, dtype=dtype)

    def _index(self, repair=False):
        self.seek(0)
        index = [self.tell()]
        truncated = False
        while(True):
            try:
                self.skipRecord()
            except NoMoreRecords:
                return index, truncated
            except IntegrityError:
                if repair:
                    self.seek(index[-1])
                    self.truncate()
                    truncated = True
                    # This way, the next iteration will raise NoMoreRecords
                else:
                    raise
            else:
                index.append(self.tell())

    def repair(self):
        """Read records until one has an IntegrityError then truncate the file at the end of the last good record.
        Leave the file position at the end of the file. Return a boolean indicating if the file was truncated.
        """
        return self._index(repair=True)[1]

    def index(self, repair=False):
        """Return the self.tell() value for each record in a list
        If successful,
        The first entry will be 0, the begining of the file, and
        the last entry will seek to the end of the file.
        Intermediate values, index[n], seek to the begining of the (n+1)th record.
        """
        if repair:
            return self._index(repair)
        else:
            return self._index(repair)[0]

    def as_gen(self, step=1, dtype_prec=None):
        while True:
            try:
                r = self.readRecord()
                for i in range(step-1):
                    self.skipRecord()
            except NoMoreRecords:
                break
            if dtype_prec is None:
                yield r
            else:
                yield numpy.fromstring(r, dtype=dtype_prec)

