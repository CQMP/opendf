#pragma once 

#include <alps/params.hpp>
#include <alps/hdf5/archive.hpp>

namespace open_df { 

template <typename SCType>
void save_data(SCType const& sc, alps::params p)
{
    typedef typename SCType::gk_type gk_type;
    typedef typename SCType::gw_type gw_type;
    typedef typename SCType::disp_type disp_type;

    static constexpr int D = SCType::NDim;
    int plaintext = p["plaintext"] | 1;
    double mu = p["mu"] | 0.0;
    fmatsubara_grid const& fgrid = sc.gd().template grid<0>(); 
    double beta = fgrid.beta();
    kmesh const& kgrid = sc.gd().template grid<1>();
    int kpts = kgrid.size();
    double knorm = pow<D>(kpts);

    auto w0 = fgrid.find_nearest(I*PI / beta);

    std::string output_file = p["output"] | "output.h5";
    std::string top = "/df";

    std::cout << "Saving data to " << output_file << top << std::endl; 
    alps::hdf5::archive ar(output_file, "w");

    // save dual quantities
    save_grid_object(ar, top + "/gd", sc.gd(), plaintext > 1);
    save_grid_object(ar, top + "/gd0", sc.gd(), plaintext > 1);
    save_grid_object(ar, top + "/sigma_d", sc.sigma_d(), plaintext > 1);

    // save lattice gf
    save_grid_object(ar, top + "/glat", sc.glat(), plaintext > 1);
    save_grid_object(ar, top + "/gloc", sc.glat_loc(), plaintext > 0);

    // save lattice self-energy
    gk_type sigma_lat = sc.sigma_lat();
    save_grid_object(ar, top + "/sigma_lat", sigma_lat, plaintext > 1);

    // save local part of lattice self-energy
    gw_type sigma_local(fgrid);
    for (auto iw : fgrid.points()) { sigma_local[iw] = sigma_lat[iw].sum() / knorm; } 
    save_grid_object(ar, top + "/sigma_local", sigma_local, plaintext > 0);

    // save cut of lattice self-energy at the first matsubara
    disp_type sigma_w0(tuple_tools::repeater<kmesh,D>::get_tuple(kgrid));
    sigma_w0.data() = sigma_lat[w0];
    save_grid_object(ar, top + "/sigma_w0", sigma_w0, plaintext > 0);
    sigma_w0.data() = sc.sigma_d()[w0];
    save_grid_object(ar, top + "/sigma_d_w0", sigma_w0, plaintext > 0);

    // save dmft self-energy for a consistency check
    save_grid_object(ar, top + "/sigma_dmft", sc.sigma_dmft(mu), plaintext > 0);


}

} // end of namespace open_df
