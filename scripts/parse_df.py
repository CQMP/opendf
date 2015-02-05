# An example of parsing df data
import h5py
import numpy as np
from numpy import pi



def main(fname = "output.h5", verbosity = 2):
    h5file = h5py.File(fname, "r")
    data = h5file["df"]
    # import lattice gf
    (grids, glat) = read_hdf5(data["glat"])
    # extract grid of Matsubara values (it is stored as a complex number, so h5py imports into 
    wgrid = grids[0].copy()[:,1]
    # extract grid of k-points (in one of the dimensions)
    kgrid = grids[1].copy()
    print "Matsubara grid =", wgrid if verbosity > 1 else None 
    print "BZ mesh =", kgrid if verbosity > 1 else None 

    # extract beta from Matsubara spacing
    beta = 2.0*pi/(wgrid[1]-wgrid[0])
    print "beta =", beta if verbosity > 0 else None
    
    # find the Matsubara frequency, closest to zero
    w0 = find_nearest_index(wgrid, pi/beta)

    # read self-energy
    (grids, sigma_lat) = read_hdf5(data["sigma_lat"])

    potential_energy = (sigma_lat*glat).sum()/pow(len(kgrid),2)
    print "Tr[\Sigma * G] = ", potential_energy

    
def read_hdf5(group):
    ''' read gridobject from hdf5 '''
    data = group["data"]
    ngrids = len(group["grids"].keys())
    grids = [np.array(group["grids"][str(i)]["values"][()]) for i in range(ngrids)]
    return (grids, np.array(data))
def find_nearest_index(array,value):
    idx = (abs(array-value)).argmin()
    return idx 
def find_nearest(array,value):
    return array[find_nearest_index(array,value)]




if __name__ == "__main__":
    main()
