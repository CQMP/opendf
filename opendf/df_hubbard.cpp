#include "df_hubbard.hpp"
#include "diagrams.hpp"
#include "lattice_traits.hpp"

namespace open_df { 

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

    // fill in initial eigenvalues of vertices
    
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
            gk_type dual_bubbles = diagram_traits::calc_bubbles(gd_, W); 

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
                full_m2 = diagrams::BS(dual_bubble_matrix, magnetic_v_matrix, true, true, 1, 1.0, true); // second order correction
                full_d2 = diagrams::BS(dual_bubble_matrix, density_v_matrix, true, true, 1, 1.0, true); // second order correction

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
        gd_.set_tail(gd0_.tail()); // assume DMFT asymptotics are good 
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
    delta_out.set_tail(delta_.tail());
    //DEBUG("GD0 = " << GD0);
    //DEBUG("GD  = " << GD);
    //DEBUG("SigmaD = " << SigmaD);
    return delta_out;

}

template <typename LatticeT>
typename df_hubbard<LatticeT>::disp_type df_hubbard<LatticeT>::spin_susc(bmatsubara_grid::point W)
{
    disp_type susc_q_data(disp_.grids()); 
/*
    auto grids = std::tuple_cat(std::make_tuple(gridF),__repeater<KMesh,D>::get_tuple(_kGrid));
    // Prepate interpolated Green's functions
    GKType GD0_interp (grids), GD_interp(grids), GLat_interp(grids);
    if (gridF._w_max != _fGrid._w_max || gridF._w_min != _fGrid._w_min) {
        GD0_interp.copyInterpolate(GD0);
        GD_interp.copyInterpolate(GD);
        GLat_interp.copyInterpolate(GLat);
        }
    else { 
        GD0_interp = GD0;
        GD_interp = GD;
        GLat_interp=GLat;
        };
        
    auto mult = _S.beta*_S.U*_S.U*_S.w_0*_S.w_1;
    GLocalType Lambda(gridF);
    Lambda.copyInterpolate(_S.getLambda());

        GridObject<ComplexType,FMatsubaraGrid,FMatsubaraGrid> magnetic_vertex(std::forward_as_tuple(gridF,gridF)); 
        decltype(magnetic_vertex)::PointFunctionType VertexF2 = [&](FMatsubaraGrid::point w1, FMatsubaraGrid::point w2){return _S.getVertex4(0.0, w1,w2);};
        auto U = _S.U;
        typename GLocalType::FunctionType lambdaf = [mult,U](ComplexType w){return 1. - U*U/4./w/w;};
        Lambda.fill(lambdaf);
        VertexF2 = [&](FMatsubaraGrid::point w1, FMatsubaraGrid::point w2)->ComplexType{
            return  mult*Lambda(w1)*Lambda(w2)*(2. + RealType(w1.index_ == w2.index_));
        };
        magnetic_vertex.fill(VertexF2);
        auto StaticV4 = magnetic_vertex.getData().getAsMatrix();
    */

    gk_type Lwk = this->glat_dmft()/this->gd0()*(-1.0);
    Lwk.set_tail(gftools::tools::fun_traits<typename gk_type::function_type>::constant(-1.0));
    auto GDL = this->gd()*Lwk;

/*
    // Prepare output
    size_t nqpts = qpts.size();
    std::vector<ComplexType> out;
    out.reserve(nqpts);

    GKType dual_bubbles = Diagrams::getStaticBubbles(GD_interp); 
    GKType gdl_bubbles = Diagrams::getStaticBubbles(GDL); 
    GKType lattice_bubbles = Diagrams::getStaticBubbles(GLat_interp); 
    
    GLocalType dual_bubble(gridF), GDL_bubble(gridF), LatticeBubble(gridF);
    for (auto q : qpts) {

        ComplexType susc=0.0;
        INFO_NONEWLINE("Evaluation of static susceptibility for q=["); for (int i=0; i<D; ++i) INFO_NONEWLINE(RealType(q[i])<<" "); INFO("]");

        GDL_bubble.fill([&](typename FMatsubaraGrid::point w){return gdl_bubbles(std::tuple_cat(std::make_tuple(w), q)); });
        dual_bubble.fill([&](typename FMatsubaraGrid::point w){return dual_bubbles(std::tuple_cat(std::make_tuple(w), q)); });
        LatticeBubble.fill([&](typename FMatsubaraGrid::point w){return lattice_bubbles(std::tuple_cat(std::make_tuple(w), q)); });

        auto GDL_bubble_vector = GDL_bubble.getData().getAsVector();

        auto dual_bubble_matrix = dual_bubble.getData().getAsDiagonalMatrix();

        #ifdef bs_matrix
        auto size = StaticV4.rows();
        auto V4Chi = MatrixType<ComplexType>::Identity(size,size) - StaticV4*dual_bubble_matrix;
        auto D1 = V4Chi.determinant();
        if (std::imag(D1)<1e-7 && std::real(D1)>0) { 
            auto full_magnetic_v = Diagrams::BS(dual_bubble_matrix, StaticV4, true, false);
            susc = (GDL_bubble_vector.transpose()*full_magnetic_v*GDL_bubble_vector)(0,0)*0.5;
            }
        else susc = -1.;

        #else
        auto m1 = mult*dual_bubble*Lambda*Lambda;
        ComplexType B_=2.0*(m1/(1.0-m1)).getData().sum();
        if (std::imag(B_)>1e-5) throw (exRuntimeError("B is imaginary."));
        RealType B = std::real(B_);
        INFO("\t\tB = "<<B);
        GLocalType C=m1*Lambda/(1.0-m1);
        GLocalType ksi = (B*Lambda + C)/(1.-B);
    
        for (auto w1 : gridF.getPoints()) {
            auto F = mult/(1.0-m1(w1))*Lambda(w1)/(1.0-B);
            for (auto w2 : gridF.getPoints()) {
                RealType kronecker = RealType(w1.index_ == w2.index_);
                susc+=GDL_bubble(w1)*F*(2.*Lambda(w2)+Lambda(w1)*kronecker*(1.-B)+2.*C(w2))*GDL_bubble(w2)*0.5;
            }
        }
        #endif

        susc+=LatticeBubble.sum()*0.5;
        INFO("Static susceptibility at q=" << q[0] << " = " << susc);
        out.push_back(susc);
        };
*/

    return susc_q_data;
} 

template class df_hubbard<cubic_traits<2>>;
template class df_hubbard<cubic_traits<3>>;

} // end of namespace open_df
