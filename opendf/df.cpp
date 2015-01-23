#include "df.hpp"
#include "diagrams.hpp"
#include "lattice_traits.hpp"

namespace open_df { 

template <typename LatticeT>
df_base<LatticeT>::df_base(gw_type gw, gw_type Delta, lattice_t lattice, kmesh kgrid):
    lattice_(lattice),
    fgrid_(gw.grid()),
    kgrid_(kgrid),
    gw_(gw),
    delta_(Delta),
    disp_(gftools::tuple_tools::repeater<kmesh,NDim>::get_array(kgrid_)), 
    gd0_(std::tuple_cat(std::forward_as_tuple(fgrid_), gftools::tuple_tools::repeater<kmesh,NDim>::get_array(kgrid_))), 
    gd_(gd0_.grids()),
    sigma_d_(gd0_.grids()),
    glat_(gd0_.grids())
{
    disp_.fill(lattice_.get_dispersion());
    glat_ = this->glat_dmft();
    for (auto w : fgrid_.points()) { gd0_[w] = glat_[w] - gw_[w]; }
    gd_ = gd0_; 
    sigma_d_ = 0.0;
}
    
template <typename LatticeT>
typename df_base<LatticeT>::gk_type df_base<LatticeT>::glat_dmft() const
{
    gk_type glat_dmft(gd0_.grids());
    for (auto w : fgrid_.points()) { 
        glat_dmft[w] = 1.0 / ( 1.0 / gw_[w] + delta_[w] - disp_.data());  
        //if (!is_float_equal(glat_dmft[w].sum()/pow<D>(kpts), gw[w], 1e-3)) ERROR(glat_dmft[w].sum()/pow<D>(kpts) << " != " << gw[w]);
        assert(is_float_equal(glat_dmft[w].sum()/pow<NDim>(kgrid_.size()), gw_[w], 1e-3)); // check consistency of gw, delta and lattice gk
        }
    return glat_dmft;
}

template <typename LatticeT>
typename df_base<LatticeT>::gw_type df_base<LatticeT>::sigma_dmft(double mu) const
{
    gw_type sigma_out(fgrid_);
    for (auto iw : fgrid_.points()) { 
        sigma_out[iw] =  iw.value() + mu - delta_[iw] - 1./gw_[iw];
        };
    return sigma_out;
}

template <typename LatticeT>
typename df_base<LatticeT>::gk_type df_base<LatticeT>::sigma_lat(double mu) const
{
    gk_type sigma_lat(this->gd0_.grids());
    sigma_lat = 0;
    double non_zero = !is_float_equal(sigma_d_.diff(sigma_lat), 0, 1e-12);
    gw_type sigma_dmft1 = this->sigma_dmft(mu);
    for (auto w : fgrid_.points()) { 
        sigma_lat[w] = sigma_dmft1[w] + non_zero / ( 1.0 + sigma_d_[w] * gw_[w] ) * sigma_d_[w];
        }
    return sigma_lat;
}

template <typename LatticeT>
typename df_base<LatticeT>::gw_type df_base<LatticeT>::glat_loc() const
{
    gw_type glatloc(fgrid_);
    double knorm = boost::math::pow<NDim>(kgrid_.size());
    for (auto w : fgrid_.points()) { 
        glatloc[w] = glat_[w].sum()/knorm; 
        }
    return std::move(glatloc);
}

template <typename LatticeT>
typename df_hubbard<LatticeT>::gw_type df_hubbard<LatticeT>::operator()(alps::params p)
{
    std::cout << "Starting ladder dual fermion calculations" << std::endl;

    int n_bs_iter = p["n_bs_iter"] | 100;
    double bs_mix = p["bs_mix"] | 1.0;
    double df_sc_cutoff = p["df_sc_cutoff"] | 1e-7;
    double df_sc_mix = p["df_sc_mix"] | 1.0;
    bool update_df_sc_mixing = p["update_df_mixing"] | true;
    int df_sc_iter = p["df_sc_iter"] | 1000;
    int nbosonic_ = std::min(int(p["nbosonic"] | 1), magnetic_vertex_.grid().max_n() + 1);
    #ifndef NDEBUG
    int verbosity = p["verbosity"] | 0; // relevant only in debug mode
    #endif

    int kpts = kgrid_.size();
    int totalkpts = boost::math::pow<NDim>(kpts);
    double knorm = double(totalkpts);
    double beta = fgrid_.beta();
    double T = 1.0/beta;
    bmatsubara_grid const& bgrid = magnetic_vertex_.grid();
    std::cout << "Vertices are defined on bosonic grid : " << bgrid << std::endl; 
    const auto unique_kpts = lattice_t::getUniqueBZPoints(kgrid_);

    gk_type gd_initial(gd_);

    gw_type gd_sum(fgrid_);
    for (auto iw : fgrid_.points()) { gd_sum[iw] = std::abs(gd_[iw].sum())/totalkpts; }; 
    std::cout << "Beginning with GD sum = " << std::abs(gd_sum.sum())/double(fgrid_.size()) << std::endl;
    
    // stream convergence
    std::ofstream diffDF_stream("diffDF.dat",std::ios::out);
    diffDF_stream.close();

    // prepare caches        
    fvertex_type m_v(fgrid_, fgrid_), d_v(m_v);
    gw_type dual_bubble(fgrid_);
    gk_type full_vertex(gd0_.grids());
    matrix_type full_m(fgrid_.size(), fgrid_.size()), full_d(full_m), full_m2(full_m), full_d2(full_d);

    
    double diff_gd = 1.0, diff_gd_min = diff_gd;
    int diff_gd_min_count = 0;

    for (size_t nd_iter=0; nd_iter<df_sc_iter && diff_gd > df_sc_cutoff; ++nd_iter) { 
        sigma_d_ = 0.0;
        std::cout << std::endl << "DF iteration " << nd_iter << std::endl;

        for (int Windex = -nbosonic_ + 1; Windex < nbosonic_; Windex++) { 
            std::complex<double> W_val = BMatsubara(Windex, beta);
            typename bmatsubara_grid::point W = bgrid.find_nearest(W_val); 
            assert(is_float_equal(W.value(), W_val, 1e-4));
            std::cout << "W (bosonic) = " << W << std::endl;
            std::cout << "Calculating bubbles" << std::endl;
            gk_type dual_bubbles = diagrams::calc_bubbles(gd_, W); 

            d_v.data() = density_vertex_[W];
            m_v.data() = magnetic_vertex_[W];

            const matrix_type density_v_matrix = d_v.data().as_matrix(); 
            const matrix_type magnetic_v_matrix = m_v.data().as_matrix(); 

            // loop through bz
            size_t nq = 1;
            for (auto pts_it = unique_kpts.begin(); pts_it != unique_kpts.end(); pts_it++) { 
                std::array<kmesh::point, NDim> q = pts_it->first; // point
                real_type q_weight = real_type(pts_it->second.size()); // it's weight
                auto other_pts = pts_it -> second; // other points, equivalent by symmetry
                std::cout << nq++ << "/" << unique_kpts.size() << ": [" << std::flush;
                //for (size_t i=0; i<D; ++i) std::cout << real_type(q[i]) << " " << std::flush; std::cout << "]. Weight : " << q_weight << ". " << std::endl;

                // define arguments conserved on the ladder
                typename gk_type::arg_tuple Wq_args = std::tuple_cat(std::make_tuple(W),q);
                std::cout << tuple_tools::print_tuple(Wq_args) << "]. Weight : " << q_weight << ". " << std::endl;
                // get dual bubble
                dual_bubble.fill([&](typename fmatsubara_grid::point w){return dual_bubbles(std::tuple_cat(std::make_tuple(w), q)); });

                matrix_type dual_bubble_matrix = dual_bubble.data().as_diagonal_matrix(); 

                std::cout << "\tMagnetic channel : " << std::flush;
                full_m = diagrams::BS(dual_bubble_matrix, magnetic_v_matrix, true, false, n_bs_iter, bs_mix);
                std::cout << "\tDensity channel  : " << std::flush;
                full_d = diagrams::BS(dual_bubble_matrix, density_v_matrix, true, false, n_bs_iter, bs_mix);

                // optimize me!
                full_m2 = diagrams::BS(dual_bubble_matrix, magnetic_v_matrix, true, true, 1, 1.0); // second order correction
                full_d2 = diagrams::BS(dual_bubble_matrix, density_v_matrix, true, true, 1, 1.0); // second order correction

                for (typename fmatsubara_grid::point iw1 : fgrid_.points())  {
                    int iwn = iw1.index();
                    std::complex<double> magnetic_val = full_m(iwn, iwn) - 0.5*full_m2(iwn, iwn);
                    std::complex<double> density_val  = full_d(iwn, iwn) - 0.5*full_d2(iwn, iwn); 
                    DEBUG("magnetic : " <<  full_m(iwn, iwn) << " " << 0.5*full_m2(iwn, iwn) << " --> " << magnetic_val, verbosity, 2);
                    DEBUG("density  : " <<  full_d(iwn, iwn) << " " << 0.5*full_d2(iwn, iwn) << " --> " << density_val, verbosity, 2);
                    for (auto q_pt : other_pts) { 
                        full_vertex.get(std::tuple_cat(std::make_tuple(iw1),q_pt)) = 0.5*(3.0*(magnetic_val)+density_val);
                        };
                    };
                } // end bz loop

            std::cout << "Updating sigma" << std::endl;
            for (auto iw1 : fgrid_.points()) {
                auto v4r = run_fft(full_vertex[iw1], FFTW_FORWARD)/knorm;
                auto gdr = run_fft(gd_[iw1], FFTW_BACKWARD);
                // in chosen notation - a.k.a horizontal ladder with (-0.25 \gamma^4 f^+ f f^+ f ) the sign in +
                sigma_d_[iw1]+= (1.0*T)*run_fft(v4r*gdr, FFTW_FORWARD); 
                };
            std::cout << "After W = " << W << " sigma diff = " << sigma_d_.diff(sigma_d_ * 0) << std::endl;

            } // end bgrid loop

        std::cout << "Total sigma diff = " << sigma_d_.diff(sigma_d_*0) << std::endl;

        // check convergence
        auto gd_new = df_sc_mix/(1.0/gd0_ - sigma_d_) + gd_*(1.0-df_sc_mix); // Dyson eq;
        diff_gd = gd_new.diff(gd_);
        if (diff_gd<diff_gd_min-df_sc_cutoff/10.) { diff_gd_min = diff_gd; diff_gd_min_count = 0; }
        else diff_gd_min_count++;
        std::cout << "Dual gd diff = " << diff_gd << std::endl;

        if (diff_gd_min_count > 12 && std::abs(df_sc_mix-0.05)>1e-3 && update_df_sc_mixing) {
            std::cerr << "\n\tCaught loop cycle. Reducing DF mixing to " << df_sc_mix/2 << " .\n" << std::endl;
            df_sc_mix=std::max(df_sc_mix/1.5, 0.05);
            gd_new = gd_initial;
            sigma_d_ = 0.0;
            diff_gd_min = diff_gd;
            diff_gd_min_count = 0;
            }

        diffDF_stream.open("diffDF.dat",std::ios::app);
        diffDF_stream << diff_gd << "  " << df_sc_mix << std::endl;
        diffDF_stream.close();

        gd_=gd_new;
        gd_.tail() = gd0_.tail(); // assume DMFT asymptotics are good 
        sigma_d_ = 0.0;

        for (auto iw : fgrid_.points()) { gd_sum[iw] = std::abs(gd_[iw].sum())/knorm; }; 
        std::cout << "GD sum = " << std::abs(gd_sum.sum())/double(fgrid_.size()) << std::endl; 
        }
    std::cout << "Finished DF iterations" << std::endl;
        
    sigma_d_ = 1.0/gd0_ - 1.0/gd_; 

    // Finish - prepare all lattice quantities
    gw_type gdloc(fgrid_), glatloc(fgrid_);
    disp_type denom(disp_.grids());
    for (auto iw : fgrid_.points()) {
        size_t iwn = size_t(iw);
        denom.data() = 1.0 / (delta_(iw) - disp_.data());
        glat_[iw] = denom.data() + denom.data() / gw_(iw) * gd_[iw] / gw_(iw) * denom.data(); 
        gdloc[iwn] = gd_[iwn].sum()/knorm; 
        glatloc[iwn] = glat_[iwn].sum()/knorm; 
        };

    gw_type delta_out(delta_); 
    delta_out = delta_ + 1.0/gw_ * gdloc / glatloc;
    // Assume DMFT asymptotics
    delta_out.tail() = delta_.tail();
    //DEBUG("GD0 = " << GD0);
    //DEBUG("GD  = " << GD);
    //DEBUG("SigmaD = " << SigmaD);
    return delta_out;

}

template class df_base<cubic_traits<2>>;
template class df_hubbard<cubic_traits<2>>;

} // end of namespace open_df
