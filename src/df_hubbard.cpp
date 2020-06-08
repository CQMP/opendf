#ifdef OPENDF_ENABLE_MPI
#include <alps/utilities/boost_mpi.hpp>
#define mpi_df_cout if(mpi_rank==0) std::cout
#else
#define mpi_df_cout std::cout
#endif

#include "opendf/df_hubbard.hpp"
#include "opendf/diagrams.hpp"
#include "opendf/lattice_traits.hpp"


namespace open_df { 

template <typename LatticeT>
alps::params& df_hubbard<LatticeT>::define_parameters(alps::params& p)
{
    p.define<double>("df_sc_mix",    1.0,   "mixing between df iterations")
     .define<double>("df_sc_cutoff", 1e-7,  "cutoff for dual iterations")
     .define<double>("bs_mix",       1.0,   "mixing between df iterations")

     .define<int>("df_sc_iter",     1000,   "maximum df iterations")
     .define<int>("nbosonic",       1,     "amount of bosonic freqs to use (reduced also by the amount of freqs in the vertex")
     .define<int>("n_bs_iter",      100,      "amount of self-consistent iterations in BS (with eval_bs_sc = 1)")
     #ifndef NDEBUG 
     .define<int>("verbosity",      0,      "debugging verbosity (only in Debug builds)")
     #endif
        
     .define<bool>("update_df_mixing", 1, "update mixing of dual gf for better accuracy")
     .define<bool>("eval_bs_sc", 0, "evaluate Bethe-Salpeter equation self-consistently");
    
    p["store_full_diag_vertex"] = false;
    p["diff_stream"] = "diffDF.dat";

    return p;
}

template <typename LatticeT>
typename df_hubbard<LatticeT>::gw_type df_hubbard<LatticeT>::operator()(alps::params p)
{
    int mpi_rank, mpi_size;
    mpi_rank=0; mpi_size=1;
    #ifdef OPENDF_ENABLE_MPI
     alps::mpi::communicator comm_df;
     mpi_rank=comm_df.rank();
     mpi_size=comm_df.size();     
     if(mpi_rank==0) std::cout<<"myrank = "<< mpi_rank<<std::endl;
    #endif 
 
    if(mpi_rank==0) std::cout << "Starting ladder dual fermion calculations" << std::endl;

    int n_bs_iter = p["n_bs_iter"];   // number of Bethe Salpeter iterations, if iterative evaluation is requested (also used, when the ladder can't be converged)
    double bs_mix = p["bs_mix"];      // mixing between subsequent Bethe-Salpeter iterations 
    double df_sc_cutoff = p["df_sc_cutoff"]; // cutoff for determining convergence of dual Green's function
    double df_sc_mix = p["df_sc_mix"];    // mixing between DF iterations
    bool update_df_sc_mixing = p["update_df_mixing"]; // reduce mixing of DF iterations, if convergence is not achieved 
    int df_sc_iter = p["df_sc_iter"]; // maximum number of DF iterations
    int nbosonic_ = std::min(int(p["nbosonic"]), magnetic_vertex_.grid().max_n() + 1); // amount of bosonic frequencies to use
    bool eval_bs_sc = p["eval_bs_sc"]; 
    bool store_full_diag_vertex = p["store_full_diag_vertex"];
    #ifndef NDEBUG 
    int verbosity = p["verbosity"]; // degugging verbosity - relevant only in debug build mode
    #else
    int verbosity = 0;
    #endif

    int kpts = kgrid_.size();
    int totalkpts = boost::math::pow<NDim>(kpts);
    double knorm = double(totalkpts);
    double beta = fgrid_.beta();
    double T = 1.0/beta;
    bmatsubara_grid const& bgrid = magnetic_vertex_.grid();
    if(mpi_rank==0) std::cout << "Vertices are defined on bosonic grid : " << bgrid << std::endl; 
    const auto unique_kpts = lattice_t::getUniqueBZPoints(kgrid_);
    const auto all_kpts = lattice_t::getAllBZPoints(kgrid_);

    gk_type gd_initial(gd_);

    gw_type gd_sum(fgrid_);
    for (auto iw : fgrid_.points()) { gd_sum[iw] = std::abs(gd_[iw].sum())/totalkpts; }; 
    if(mpi_rank==0) std::cout << "Beginning with GD sum = " << std::abs(gd_sum.sum())/double(fgrid_.size()) << std::endl;
    
    // prepare caches        
    fvertex_type m_v(fgrid_, fgrid_), d_v(m_v);
    gw_type dual_bubble(fgrid_);
    gk_type full_vertex(gd0_.grids());
    matrix_type full_m(fgrid_.size(), fgrid_.size()), full_d(full_m), full_m2(full_m), full_d2(full_d);
    double min_det = 1;
    container<std::complex<double>, NDim> v4r(disp_.data()), gdr(disp_.data());

    // full vertex cache
    typedef typename diagram_traits::full_diag_vertex_type full_diag_vertex_t; 
    if (store_full_diag_vertex) { 
        full_diag_vertex_ptr_.reset(new full_diag_vertex_t(std::tuple_cat(std::make_tuple(bgrid), gd0_.grids()))); 
        *full_diag_vertex_ptr_ = 0;
    }

    // stream convergence
    std::ofstream diffDF_stream(p["diff_stream"].as<std::string>(),std::ios::out);
    diffDF_stream.close();
    double diff_gd = 1.0, diff_gd_min = diff_gd;
    int diff_gd_min_count = 0;

    int av_Windex= (2*nbosonic_-1)/mpi_size+1;
    int start_Windex = mpi_rank*av_Windex - nbosonic_ +1;
    int end_Windex = (mpi_rank+1)*av_Windex - nbosonic_;
    if(end_Windex>=nbosonic_) end_Windex=nbosonic_-1;
    //std::cout<<nbosonic_<<"  "<<mpi_size<<"   "<<av_Windex<<std::endl;
   
    std::complex<double> reduced_sigma_d[sigma_d_.size()];
    std::complex<double> vector_sigma_d[sigma_d_.size()];
    

    for (size_t nd_iter=0; nd_iter<df_sc_iter && diff_gd > df_sc_cutoff; ++nd_iter) { 
        sigma_d_ = 0.0;
        if(mpi_rank==0) std::cout << std::endl << "DF iteration " << nd_iter << std::endl;
       
        // loop through conserved bosonic frequencies
        
        for (int Windex = start_Windex; Windex <= end_Windex; Windex++) { 
            std::complex<double> W_val = BMatsubara(Windex, beta);
            typename bmatsubara_grid::point W = bgrid.find_nearest(W_val); 
            assert(is_float_equal(W.value(), W_val, 1e-4));
            if(mpi_rank==0) std::cout << "W (bosonic) = " << W << std::endl;
            if(mpi_rank==0) std::cout << "Calculating bubbles" << std::endl;
            gk_type dual_bubbles = diagram_traits::calc_bubbles(gd_, W); 

                
                
            std::array<double, NDim> q1; q1.fill(0.0);
            typename gk_type::arg_tuple W_shift = std::tuple_cat(std::make_tuple(W_val),q1);
            gk_type gd_shift = gd_.shift(W_shift);

            d_v.data() = density_vertex_[W];
            m_v.data() = magnetic_vertex_[W];

            const matrix_type density_v_matrix = d_v.data().as_matrix(); 
            const matrix_type magnetic_v_matrix = m_v.data().as_matrix(); 

            min_det = 1;
            // loop through the Brilloin zone. We do it by looping through the irreducible wedge and assigning the correspondent weight to each point
            size_t nq = 1;
            for (auto pts_it = unique_kpts.begin(); pts_it != unique_kpts.end(); pts_it++) { 
                std::array<kmesh::point, NDim> q = pts_it->first; // point
                real_type q_weight = real_type(pts_it->second.size()); // it's weight
                auto other_qpts = pts_it -> second; // other points, equivalent by symmetry
                if(mpi_rank==0) std::cout << nq++ << "/" << unique_kpts.size() << ": [" << std::flush;
                //for (size_t i=0; i<D; ++i) std::cout << real_type(q[i]) << " " << std::flush; std::cout << "]. Weight : " << q_weight << ". " << std::endl;

                // define arguments conserved on the ladder
                typename gk_type::arg_tuple Wq_args = std::tuple_cat(std::make_tuple(W),q);
                if(mpi_rank==0) std::cout << tuple_tools::print_tuple(Wq_args) << "]. Weight : " << q_weight << ". " << std::flush;
                // get dual bubble
                dual_bubble.fill([&](typename fmatsubara_grid::point w){return dual_bubbles(std::tuple_cat(std::make_tuple(w), q)); });

                matrix_type dual_bubble_matrix = dual_bubble.data().as_diagonal_matrix(); 

                // Calculate ladders in different channels and get a determinant of 1 - vertex * bubble. 
                // If it's negative - one eigenvalue is negative, i.e. the ladder can't be evaluated.
                // magnetic channel
                if(mpi_rank==0) std::cout << "\tMagnetic " << std::flush;
                forward_bs magnetic_bs(dual_bubble_matrix, magnetic_v_matrix, 0);
                full_m = magnetic_bs.solve(eval_bs_sc, n_bs_iter, bs_mix);
                double m_det = magnetic_bs.determinant().real();
                if(mpi_rank==0) std::cout << "det = " << m_det;
                min_det = std::min(min_det, m_det); 
                // density channel
                if(mpi_rank==0) std::cout << "\tDensity " << std::flush;
                forward_bs density_bs(dual_bubble_matrix, density_v_matrix, 0);
                full_d = density_bs.solve(eval_bs_sc, n_bs_iter, bs_mix);
                double d_det = density_bs.determinant().real();
                if(mpi_rank==0) std::cout<< "det = " << d_det;
                min_det = std::min(min_det, d_det); 

                // second order correction
                full_m2 = magnetic_bs.solve_iterations(1, 1.0, true);
                full_d2 = density_bs.solve_iterations(1, 1.0, true); 

                for (typename fmatsubara_grid::point iw1 : fgrid_.points())  {
                    int iwn = iw1.index();
                    std::complex<double> magnetic_val = full_m(iwn, iwn) - 0.5*full_m2(iwn, iwn);
                    std::complex<double> density_val  = full_d(iwn, iwn) - 0.5*full_d2(iwn, iwn); 
                    DEBUG("magnetic : " <<  full_m(iwn, iwn) << " " << 0.5*full_m2(iwn, iwn) << " --> " << magnetic_val, verbosity, 2);
                    DEBUG("density  : " <<  full_d(iwn, iwn) << " " << 0.5*full_d2(iwn, iwn) << " --> " << density_val, verbosity, 2);
                    for (auto q_pt : other_qpts) { 
                        full_vertex.get(std::tuple_cat(std::make_tuple(iw1),q_pt)) = 0.5*(3.0*(magnetic_val)+density_val);
                        };
                    };
                if(mpi_rank==0) std::cout << std::endl;
                } // end bz loop

            if (store_full_diag_vertex) { 
                (*full_diag_vertex_ptr_)[W] = full_vertex.data();
                }
            if(mpi_rank==0) std::cout << "Updating sigma" << std::endl;
            for (auto iw1 : fgrid_.points()) {
                v4r = run_fft(full_vertex[iw1], FFTW_FORWARD)/knorm;
                gdr = run_fft(gd_shift[iw1], FFTW_BACKWARD);
                // in chosen notation - a.k.a horizontal ladder with (-0.25 \gamma^4 f^+ f f^+ f ) the sign in +
                sigma_d_[iw1]+= (1.0*T)*run_fft(v4r*gdr, FFTW_FORWARD); 
                };
            //std::cout << "After W = " << W << " sigma diff = " << sigma_d_.diff(sigma_d_ * 0) << std::endl;
            } // end bgrid loop

            #ifdef OPENDF_ENABLE_MPI
            MPI_Barrier(MPI_COMM_WORLD);
            int jj=0;
            for(auto iw1 : fgrid_.points()){
              for (auto pts_it = all_kpts.begin(); pts_it != all_kpts.end(); pts_it++) { 
                std::array<kmesh::point, NDim> q = *pts_it;    
                auto wk = std::tuple_cat(std::make_tuple(iw1),q); 
                vector_sigma_d[jj]=sigma_d_(wk);
                reduced_sigma_d[jj]=0.0;
                jj++;
             }
            }

            int size_sigma=sigma_d_.size();
            MPI_Barrier(MPI_COMM_WORLD);                     
            MPI_Allreduce(&vector_sigma_d[0],&reduced_sigma_d[0],size_sigma,MPI_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_WORLD);
            MPI_Barrier(MPI_COMM_WORLD);    

            jj=0;
            for(auto iw1 : fgrid_.points()){
              for (auto pts_it = all_kpts.begin(); pts_it != all_kpts.end(); pts_it++) { 
                std::array<kmesh::point, NDim> q = *pts_it;    
                auto wk = std::tuple_cat(std::make_tuple(iw1),q); 
                sigma_d_(wk)=reduced_sigma_d[jj];
                jj++;
             }
            }
           #endif
 
        if(mpi_rank==0) std::cout << "Total sigma diff = " << sigma_d_.diff(sigma_d_*0) << std::endl;

        // check convergence
        auto gd_new = df_sc_mix/(1.0/gd0_ - sigma_d_) + gd_*(1.0-df_sc_mix); // Dyson eq;
        diff_gd = gd_new.diff(gd_);
        if (diff_gd<diff_gd_min-df_sc_cutoff/10.) { diff_gd_min = diff_gd; diff_gd_min_count = 0; }
        else diff_gd_min_count++;
        if(mpi_rank==0) std::cout << "Dual gd diff = " << diff_gd << std::endl;

        if (diff_gd_min_count > 12 && std::abs(df_sc_mix-0.05)>1e-3 && update_df_sc_mixing) {
            if(mpi_rank==0) std::cerr << "\n\tCaught loop cycle. Reducing DF mixing to " << df_sc_mix/2 << " .\n" << std::endl;
            df_sc_mix=std::max(df_sc_mix/1.5, 0.05);
            gd_new = gd_initial;
            sigma_d_ = 0.0;
            diff_gd_min = diff_gd;
            diff_gd_min_count = 0;
            }

        if(mpi_rank==0) diffDF_stream.open(p["diff_stream"].as<std::string>(),std::ios::app);
        if(mpi_rank==0) diffDF_stream << diff_gd << "  " << df_sc_mix << " " << min_det << std::endl;
        if(mpi_rank==0) diffDF_stream.close();

        gd_=gd_new;
        gd_.set_tail(gd0_.tail()); // assume DMFT asymptotics are good 
        sigma_d_ = 0.0;

        for (auto iw : fgrid_.points()) { gd_sum[iw] = std::abs(gd_[iw].sum())/knorm; }; 
        if(mpi_rank==0) std::cout << "GD sum = " << std::abs(gd_sum.sum())/double(fgrid_.size()) << std::endl; 
        #ifdef OPENDF_ENABLE_MPI
        MPI_Barrier(MPI_COMM_WORLD);
        #endif
        }
    if(mpi_rank==0) std::cout << "Finished DF iterations" << std::endl;
        
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
    if (store_full_diag_vertex) { 
      #ifdef OPENDF_ENABLE_MPI
       MPI_Barrier(MPI_COMM_WORLD);
       int size_full_vertex=bgrid.size()*sigma_d_.size();
       //std::array<std::complex<double>,size_full_vertex> vector_full_vertex;
       //std::array<std::complex<double>,size_full_vertex> reduced_full_vertex;
       std::complex<double> vector_full_vertex[size_full_vertex];
       std::complex<double> reduced_full_vertex[size_full_vertex];
       std::complex<double> czero(0.0,0.0);
       std::fill(&reduced_full_vertex[0],&reduced_full_vertex[size_full_vertex-1],czero);
       int jj=0;
       for(auto iw1 : bgrid.points()){
        for(auto iw2 : fgrid_.points()){
         for (auto pts_it = all_kpts.begin(); pts_it != all_kpts.end(); pts_it++) { 
            std::array<kmesh::point, NDim> q = *pts_it;    
            auto wk = std::tuple_cat(std::make_tuple(iw1),std::make_tuple(iw2),q);
            vector_full_vertex[jj]=(*full_diag_vertex_ptr_)(wk);
            jj++;
         }
        }
       }

       MPI_Barrier(MPI_COMM_WORLD);                     
       MPI_Allreduce(&vector_full_vertex[0],&reduced_full_vertex[0],size_full_vertex,MPI_DOUBLE_COMPLEX,MPI_SUM,MPI_COMM_WORLD);
       MPI_Barrier(MPI_COMM_WORLD); 

       jj=0;
       for(auto iw1 : bgrid.points()){
        for(auto iw2 : fgrid_.points()){
         for (auto pts_it = all_kpts.begin(); pts_it != all_kpts.end(); pts_it++) { 
            std::array<kmesh::point, NDim> q = *pts_it;    
            auto wk = std::tuple_cat(std::make_tuple(iw1),std::make_tuple(iw2),q);
            (*full_diag_vertex_ptr_)(wk)=reduced_full_vertex[jj];
            jj++;
         }
        }
       }        

     MPI_Barrier(MPI_COMM_WORLD);
    #endif
    }



    return delta_out;

}

template <typename LatticeT>
typename df_hubbard<LatticeT>::disp_type df_hubbard<LatticeT>::get_susc_(vertex_type const& vertex, bmatsubara_grid::point W, double norm, bool add_lattice_bubble) const
{
    disp_type susc_q_data(disp_.grids()); 

    gk_type Lwk = this->glat_dmft()/this->gd0()*(-1.0);
    Lwk.set_tail(gftools::tools::fun_traits<typename gk_type::function_type>::constant(-1.0));
    auto GDL = this->gd()*Lwk;

    const matrix_type v_matrix = vertex[W].as_matrix(); 

    gk_type dual_bubbles = diagram_traits::calc_bubbles(gd_, W); 
    gk_type gdl_bubbles = diagram_traits::calc_bubbles(GDL, W); 
    gk_type lattice_bubbles = diagram_traits::calc_bubbles(glat_, W); 

    gw_type dual_bubble(fgrid_), lattice_bubble(fgrid_), gdl_bubble(fgrid_);

    int kpts = kgrid_.size();
    //int totalkpts = boost::math::pow<NDim>(kpts);
    //double knorm = double(totalkpts);
    const auto unique_kpts = lattice_t::getUniqueBZPoints(kgrid_);

    size_t nq = 1;
    for (auto pts_it = unique_kpts.begin(); pts_it != unique_kpts.end(); pts_it++) { 
        std::array<kmesh::point, NDim> q = pts_it->first; // point
        auto other_qpts = pts_it -> second; // other points, equivalent by symmetry
        nq++;

        std::complex<double> susc = 0.0;

        dual_bubble.fill([&](typename fmatsubara_grid::point w){return dual_bubbles(std::tuple_cat(std::make_tuple(w), q)); });
        lattice_bubble.fill([&](typename fmatsubara_grid::point w){return lattice_bubbles(std::tuple_cat(std::make_tuple(w), q)); });
        gdl_bubble.fill([&](typename fmatsubara_grid::point w){return gdl_bubbles(std::tuple_cat(std::make_tuple(w), q)); });

        auto gdl_bubble_vector = gdl_bubble.data().as_vector();
        //DEBUG("gdl \n" << gdl_bubble_vector);
        matrix_type dual_bubble_matrix = dual_bubble.data().as_diagonal_matrix(); 
        //DEBUG(" db \n" << dual_bubble_matrix);
        //DEBUG("m_v \n" << v_matrix);

        forward_bs bs(dual_bubble_matrix, v_matrix, 0);
        //matrix_type full_m = bs.solve_inversion();
        matrix_type full_m = bs.solve_inversion();
        //DEBUG("full m : \n" << full_m);
        double m_det = bs.determinant().real();

        if (std::imag(m_det)<1e-7 && std::real(m_det)>0) { 
            susc = (gdl_bubble_vector.transpose()*full_m*gdl_bubble_vector)(0,0)*norm;
            //DEBUG(gftools::tuple_tools::print_array(q));
            //DEBUG(susc);
            susc+=lattice_bubble.sum()*norm*double(add_lattice_bubble);
            //DEBUG(lattice_bubble.sum());
            }
        else susc = -1.;

        for (auto q1 : other_qpts){
            // this is an intel compiler bugfix. Intel can't convert std::array to std::tuple, so we do that by extracting indices and then finding the corresponding tuple
            std::array<size_t, NDim> indices; for (int i = 0; i<NDim; i++) indices[i]=q1[i].index();
            typename disp_type::point_tuple q2 = gftools::tools::grid_tuple_traits<typename disp_type::grid_tuple>::points(indices, susc_q_data.grids());
            susc_q_data(q2) = susc;                                               
            }
        };

    return susc_q_data;
} 

template <typename LatticeT>
void df_hubbard<LatticeT>::calc_full_diag_vertex(alps::params p) //, std::vector<bz_point> kpoints)
{
    p["store_full_diag_vertex"] = true;
    p["df_sc_mix"] = 0;
    p["df_sc_iter"] = 1;
    p["diff_stream"] = "diff_full_vertex.dat";
    // rerun self-consistency and get the full vertex for diagnostics
    this->operator()(p);
}

template <typename LatticeT>
typename df_hubbard<LatticeT>::full_diag_vertex_type const& df_hubbard<LatticeT>::full_diag_vertex() const
{
    if (!full_diag_vertex_ptr_) throw std::logic_error("No full diag vertex stored -> run calc_full_diag_vertex first");
    return *full_diag_vertex_ptr_; 
}

template <typename LatticeT>
std::tuple<std::vector<typename df_hubbard<LatticeT>::full_diag_vertex_type>, std::vector<typename df_hubbard<LatticeT>::full_diag_vertex_type>> 
   df_hubbard<LatticeT>::fluctuation_diagnostics(std::vector<bz_point> kpoints, bool self_check) const
{
    full_diag_vertex_type const& full_diag_vertex = this->full_diag_vertex();

    std::vector<full_diag_vertex_type> out_sigma_d ( kpoints.size(), full_diag_vertex * 0 );
    std::vector<full_diag_vertex_type> out_sigma_lat ( kpoints.size(), full_diag_vertex * 0 );

    bmatsubara_grid const& bgrid = std::get<0>(full_diag_vertex.grids());

    gk_type gd_shift(gd_.grids());
    gk_type sigma_contrib(gd_.grids());
    gk_type sigma_lat_contrib(gd_.grids());
    gk_type sigma_check(gd_.grids()); sigma_check = 0;
    gk_type sigma_d2(gd_.grids()); sigma_d2 = 0;
    //const auto unique_kpts = lattice_t::getUniqueBZPoints(kgrid_);
    const auto all_kpts = lattice_t::getAllBZPoints(kgrid_);
    double knorm = all_kpts.size();
    double beta = fgrid_.beta();
    double T = 1.0/beta;

    if (!kpoints.size()) { std::cerr << "No points for fluctuation diagnostics" << std::endl; return std::forward_as_tuple(out_sigma_d, out_sigma_lat); }

    for (typename bmatsubara_grid::point W : bgrid.points()) { 
        //gk_type dual_bubbles = diagram_traits::calc_bubbles(gd_, W); 
        std::cout << "W (bosonic) = " << W << std::endl;
        size_t nq = 1;
        if (full_diag_vertex[W].diff(full_diag_vertex[W]*0) == 0 ) continue;
        for (auto pts_it = all_kpts.begin(); pts_it != all_kpts.end(); pts_it++) { 
            std::array<kmesh::point, NDim> q = *pts_it;//->first; // point
            std::cout << nq++ << "/" << all_kpts.size() << ": [" << std::flush;
            typename gk_type::point_tuple Wq_args = std::tuple_cat(std::make_tuple(W),q);
            typename gk_type::arg_tuple Wq_args_print = std::tuple_cat(std::make_tuple(W),q);
            std::cout << tuple_tools::print_tuple(Wq_args_print) << "]. Weight : " << 1 << ". " << std::endl;
            
            // T * V(Omega, q, omega, omega) * G(omega + Omega, k + q) 
            gd_shift = gd_.shift(Wq_args_print);
            //std::cout << full_diag_vertex[W].diff(full_diag_vertex[W]*0) << std::endl;

            for (auto iw : fgrid_.points()) { 
                auto WwQ = std::tuple_cat(std::make_tuple(W), std::make_tuple(iw), q);
                std::complex<double> vert_val = full_diag_vertex(WwQ);
                sigma_contrib[iw] = T * vert_val * gd_shift[iw] / knorm;
                // crude, may have to fix it
                sigma_lat_contrib[iw] = sigma_contrib[iw] / ( sigma_d_[iw] * gw_[iw] + 1.0); 
                
                for (int k_ = 0; k_ < kpoints.size(); ++k_) { 
                    bz_point k = kpoints[k_];
                    auto wk = std::tuple_cat(std::make_tuple(iw), k);
                    out_sigma_d[k_](WwQ) = sigma_contrib(wk);

                    // Fluctuation diagnostics for lattice sigma is more involved, as 
                    // Sigma_{lat} - Sigma^{DMFT} = \sum_q \Sigma_q / ( 1 - g * \sum_q \Sigma_q)
                    // so independent components for q can not be straightforwardly obtained 
                    // so we rewrite this as 
                    // \Sigma' = \Sigma_q + \sum_q Sigma_q * g * \Sigma_q 
                    // for now just stick to \Sigma_q / (1 + \Sigma g)

                    out_sigma_lat[k_](WwQ) = sigma_lat_contrib(wk); 
                    }

             }
            sigma_check += sigma_contrib;
        }

        if (self_check) { 
            std::array<double, NDim> q1; q1.fill(0.0);
            typename gk_type::arg_tuple W_shift = std::tuple_cat(std::make_tuple(W.value()),q1);
            gk_type gd_shift1 = gd_.shift(W_shift);
            for (auto iw1 : fgrid_.points()) {
                auto v4r = run_fft(full_diag_vertex[W][iw1.index()], FFTW_FORWARD)/knorm;
                auto gdr = run_fft(gd_shift1[iw1], FFTW_BACKWARD);
                // in chosen notation - a.k.a horizontal ladder with (-0.25 \gamma^4 f^+ f f^+ f ) the sign in +
                sigma_d2[iw1]+= (1.0*T)*run_fft(v4r*gdr, FFTW_FORWARD); 
                };
        }
    }
    std::cout << "dual sigma diff = " << sigma_check.diff(sigma_d_) << std::endl;
    if (self_check) { 
        std::cout << "dual sigma check diff2 = " << sigma_d2.diff(sigma_d_) << std::endl;
        double check_diff = sigma_check.diff(sigma_d2);
        std::cout << "diff between fft and direct = " << check_diff << std::endl;
        if (!gftools::tools::is_float_equal(check_diff, 0, 1e-8)) throw std::logic_error("Problem with the fluctuation diagnostics : the self-energy does not match the calculated value");
        }
    return std::forward_as_tuple(out_sigma_d, out_sigma_lat);
}

OPENDF_INSTANTIATE_LATTICE_OBJECT(df_hubbard);

} // end of namespace open_df
