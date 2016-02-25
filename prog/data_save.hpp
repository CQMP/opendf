#pragma once 

#include <alps/params.hpp>
#include <alps/hdf5/archive.hpp>

namespace open_df { 

alps::params& save_define_parameters(alps::params& p)
{
     p.define<int>("plaintext",    0,      "save additionally to plaintext files (2 = verbose, 1 = save essential, 0 = no plaintext)");
     p.define<bool>("save_susc",    1,      "save susceptibilities");
    return p;
}

template <typename SCType>
void save_data(SCType const& sc, typename SCType::gw_type new_delta, alps::params p)
{
    typedef typename SCType::gk_type gk_type;
    typedef typename SCType::gw_type gw_type;
    typedef typename SCType::disp_type disp_type;

    static constexpr int D = SCType::NDim;
    int plaintext = p["plaintext"];
    double mu = p["mu"];
    fmatsubara_grid const& fgrid = sc.gd().template grid<0>(); 
    bmatsubara_grid const& bgrid = sc.bgrid();
    double beta = fgrid.beta();
    kmesh const& kgrid = sc.gd().template grid<1>();
    int kpts = kgrid.size();
    double knorm = pow<D>(kpts);

    fmatsubara_grid::point w0 = fgrid.find_nearest(I*PI / beta);
    bmatsubara_grid::point W0 = bgrid.find_nearest(0.0);

    std::string output_file = p["output"];
    std::string top = "/df";

    std::cout << "Saving data to " << output_file << top << std::endl; 
    alps::hdf5::archive ar(output_file, "w");

    // save parameters
    ar[top + "/parameters"] << p;

    // save updated hybridization function
    save_grid_object(ar, top + "/delta_update", new_delta, plaintext > 0);

    // save dual quantities
    save_grid_object(ar, top + "/gd", sc.gd(), plaintext > 1);
    save_grid_object(ar, top + "/gd0", sc.gd0(), plaintext > 1);
    save_grid_object(ar, top + "/sigma_d", sc.sigma_d(), plaintext > 1);

    // save dual bubbles
    gk_type dual_bubbles0_W0 = SCType::diagram_traits::calc_bubbles(sc.gd0(), W0);
    gk_type dual_bubbles_W0 = SCType::diagram_traits::calc_bubbles(sc.gd(), W0);
    save_grid_object(ar, top + "/dual_bubble0_W0", dual_bubbles0_W0, plaintext > 1);
    save_grid_object(ar, top + "/dual_bubble_W0", dual_bubbles_W0, plaintext > 1);

    disp_type db0_W0_w0(tuple_tools::repeater<kmesh,D>::get_tuple(kgrid));
    disp_type db_W0_w0(db0_W0_w0.grids());
    db0_W0_w0.data() = dual_bubbles0_W0[w0];
    db_W0_w0.data() = dual_bubbles_W0[w0];
    save_grid_object(ar, top + "/dual_bubble0_W0_w0", db0_W0_w0, plaintext > 1);
    save_grid_object(ar, top + "/dual_bubble_W0_w0", db_W0_w0, plaintext > 1);

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

    // susceptibilitiles
    bool save_susc = p["save_susc"];

    enum_grid rgrid(0, kgrid.size(), false); // a grid in real space
    // typedef for susceptibility in real space
    // same as grid_object<std::complex<double>, enum_grid, enum_grid> with enum_grid repeated D times.
    typedef typename gftools::tools::ArgBackGenerator<D,enum_grid,grid_object,std::complex<double>>::type susc_r_type; 
    if (save_susc) { 
        auto spin_susc = sc.spin_susc(W0);
        save_grid_object(ar, top + "/spin_susc_W" + std::to_string(W0.value().imag())+"_k", spin_susc, plaintext > 0); 
        auto charge_susc = sc.charge_susc(W0);
        save_grid_object(ar, top + "/charge_susc_W" + std::to_string(W0.value().imag())+"_k", charge_susc, plaintext > 0); 

        susc_r_type susc_r(gftools::tuple_tools::repeater<enum_grid, SCType::NDim>::get_tuple(rgrid));
        susc_r.data() = run_fft(spin_susc.data(), FFTW_BACKWARD);
        save_grid_object(ar, top + "/spin_susc_W" + std::to_string(W0.value().imag())+"_r", susc_r, plaintext > 0); 
        susc_r.data() = run_fft(charge_susc.data(), FFTW_BACKWARD);
        save_grid_object(ar, top + "/charge_susc_W" + std::to_string(W0.value().imag())+"_r", susc_r, plaintext > 0); 
    }

/*
    if (plaintext > 0) { 
    sigma_w0 = 0;
    disp_type glat0_w0 = 1./(w0 - sc.dispersion()); 
    disp_type glat_w0(glat0_w0); 
    glat_w0.data() = sc.glat()[w0];
    sigma_w0 = 1./glat0_w0 - 1./glat_w0;
    save_grid_object(ar, top + "/sigma_w0_dyson", sigma_w0, plaintext > 0);
    }
*/
}

} // end of namespace open_df
