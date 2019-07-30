try:
    import awkward.numba as awkward
except ImportError:
    import awkward

from awkward.util import numpy
import numba

akd = awkward
np = numpy
nb = numba

import lz4.frame
import cloudpickle


def load(filename):
    '''
    Load a coffea file from disk
    '''
    with lz4.frame.open(filename) as fin:
        output = cloudpickle.load(fin)
    return output


def save(output, filename):
    '''
    Save a coffea object or collection thereof to disk
    Suggested suffix: .coffea
    '''
    with lz4.frame.open(filename, 'wb') as fout:
        cloudpickle.dump(output, fout)

import os
USE_CUPY = int(os.environ.get("HEPACCELERATE_CUDA", 0)) == 1
numpy_backend = numpy

if USE_CUPY:
    import cupy
    numpy_backend = cupy
    from hepaccelerate.backend_cuda import searchsorted
else:
    from hepaccelerate.backend_cpu import searchsorted

def searchsorted_wrapped(arr, vals, side="left", asnumpy=True):
    if USE_CUPY:
        arr = cupy.array(arr, dtype=cupy.float32)
        vals = cupy.array(vals, dtype=cupy.float32)

    idx = searchsorted(arr, vals, side)

    if USE_CUPY and asnumpy:
        idx = cupy.asnumpy(idx)
    assert(np.all(idx < len(arr))) 
    assert(np.all(idx >= 0)) 
    return idx

class interp1d:
    def __init__(self, x, y):
        self.x = numpy_backend.array(x)
        self.y = numpy_backend.array(y)
        self.ydiffs = numpy_backend.diff(self.y)
        self.xdiffs = numpy_backend.diff(self.x)
        self.delta = self.ydiffs / self.xdiffs

        #self.scipy_interp1d = scipy.interpolate.interp1d(x, y)

    def __call__(self, xvals):
        idx = searchsorted_wrapped(self.x, xvals, asnumpy=False)
       
        xshifts =  self.x[idx] - xvals 
        yvals = self.y[idx] 
        yvals_new = yvals - self.delta[idx-1]*xshifts
        
        #yvals_scipy = self.scipy_interp1d(xvals)
        #assert(np.sum(np.power(yvals_new - yvals_scipy,2)) < 1e-1)
        return yvals_new
