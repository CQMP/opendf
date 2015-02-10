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

    potential_energy = (sigma_lat*glat).sum()/pow(len(kgrid),2)/beta
    print "Tr[\Sigma * G] = ", potential_energy

    sigma_g = wgrid.copy() * 0.0j
    for w in range(len(wgrid)):
        sigma_g[w] = (sigma_lat[w,:]*glat[w,:]).sum()/pow(len(kgrid),2)/beta
    np.savetxt("sigma_g.dat", np.vstack([wgrid,np.real(sigma_g)]).transpose())
    
def read_hdf5(group):
    ''' read gridobject from hdf5 '''
    data = np.array(group["data"])
    ngrids = len(group["grids"].keys())
    # check if we have a compelx number dataset 
    if len(data.shape) > ngrids and data.shape[ngrids] == 2:
        data2 = data.view(complex)
        data = data2.copy()
    grids = [np.array(group["grids"][str(i)]["values"][()]) for i in range(ngrids)]
    return (grids, data)
def find_nearest_index(array,value):
    idx = (abs(array-value)).argmin()
    return idx 
def find_nearest(array,value):
    return array[find_nearest_index(array,value)]




if __name__ == "__main__":
    main()
