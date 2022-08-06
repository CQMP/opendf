#pragma once 

#include <alps/params.hpp>
#include <alps/hdf5/archive.hpp>

namespace open_df { 

alps::params& save_define_parameters(alps::params& p)
{
     p.define<int>("plaintext",    0,      "save additionally to plaintext files (2 = verbose, 1 = save essential, 0 = no plaintext)");
     p.define<bool>("save_susc",    1,      "save susceptibilities");
     p.define<bool>("fluct_diag",    0,      "perform fluctuation diagnostics");
     p.define<bool>("add_lattice_bubble", 0, "add lattice bubble to the susceptibility. WARNING : lattice bubble is evaluated naively and produces log error, consider adding it yourself");
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
        for (typename bmatsubara_grid::point W : bgrid.points()) {  
            auto spin_susc = sc.spin_susc(W, p["add_lattice_bubble"] );
            if (is_float_equal(spin_susc.diff(spin_susc*0), 0, 1e-12)) continue;
            save_grid_object(ar, top + "/spin_susc_W" + std::to_string(W.value().imag())+"_k", spin_susc, plaintext > 0); 
            auto charge_susc = sc.charge_susc(W, p["add_lattice_bubble"]);
            save_grid_object(ar, top + "/charge_susc_W" + std::to_string(W.value().imag())+"_k", charge_susc, plaintext > 0); 

            susc_r_type susc_r(gftools::tuple_tools::repeater<enum_grid, SCType::NDim>::get_tuple(rgrid));
            susc_r.data() = run_fft(spin_susc.data(), FFTW_BACKWARD);
            save_grid_object(ar, top + "/spin_susc_W" + std::to_string(W.value().imag())+"_r", susc_r, plaintext > 0); 
            susc_r.data() = run_fft(charge_susc.data(), FFTW_BACKWARD);
            save_grid_object(ar, top + "/charge_susc_W" + std::to_string(W.value().imag())+"_r", susc_r, plaintext > 0); 
        }
    }
    
    // fluctuation diagnostics
    if (p["fluct_diag"].as<bool>()) { 
        typedef typename SCType::bz_point bz_point;
        typedef std::array<double, D> bz_point_val;
        // define mu and tp
        double mu=p["mu"];
        double tp=p["tp"];
        // do pi,0 and pi/2 pi/2
        kmesh::point k_pi = kgrid.find_nearest(M_PI);
        kmesh::point k_pi_half = kgrid.find_nearest(M_PI/2);
        kmesh::point k_zero = kgrid.find_nearest(0);
        // do kx = pi cut zeroes
        kmesh::point kx_pi_cut = kgrid.find_nearest(std::acos((mu-2)/(4*tp-2)));
        // do kx = ky cut zeroes. Values for both zero and nonzero tp
        kmesh::point kx_ky_cut_zero = kgrid.find_nearest(std::acos((mu)/(4)));
        kmesh::point kx_ky_cut_nonzero = kgrid.find_nearest(std::acos(-(1-std::sqrt(1-tp*mu))/(2*tp)));

        std::vector<bz_point> fluct_pts;

        // add here points for fluctuation diagnostics
        fluct_pts.reserve(4);
        // add pi/2 pi/2Raman demo session
        bz_point p1 = gftools::tuple_tools::repeater<kmesh::point, D>::get_array(k_pi_half);
        fluct_pts.push_back(p1); 
        // add pi, 0
        p1.fill(k_zero);
        p1[0] = k_pi;
        fluct_pts.push_back(p1); 
        // add kx = pi cut zeroes
        p1.fill(kx_pi_cut);
        p1[0] = k_pi;
        fluct_pts.push_back(p1);
        //add kx = ky cut zeroes
        (tp==0) ? p1.fill(kx_ky_cut_zero) : p1.fill(kx_ky_cut_nonzero);
        fluct_pts.push_back(p1);

        //std::cout << "Fluctuation diagnostics points: " << std::endl;
        //for (bz_point k : fluct_pts) std::cout << "--> " <<  gftools::tuple_tools::print_array(k) << std::endl;

        save_grid_object(ar, top + "/fluct_diag/full_diag_dual_vertex", sc.full_diag_vertex(), plaintext > 3); 
        auto fluct_data = sc.fluctuation_diagnostics(fluct_pts, true);

        auto& sigma_d_diagnostics = std::get<0>(fluct_data);
        auto& sigma_lat_diagnostics = std::get<1>(fluct_data);

        for (int k_ = 0; k_ < fluct_pts.size(); ++k_) { 
            bz_point k = fluct_pts[k_];
            typename SCType::disp_type::arg_tuple k1 = std::tuple_cat(k);
            std::string postfix = gftools::tuple_tools::print_tuple(k1);
            std::replace(postfix.begin(),postfix.end(), ' ', '_');
            std::cout << "--> " << postfix << std::endl;
            save_grid_object(ar, top + "/fluct_diag/sigma_d_" + postfix, sigma_d_diagnostics[k_], plaintext > 3);
            save_grid_object(ar, top + "/fluct_diag/sigma_lat_" + postfix, sigma_lat_diagnostics[k_], plaintext > 3);
        }
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
