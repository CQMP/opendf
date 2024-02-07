#pragma once 
#include "opendf/df_base.hpp"


namespace open_df { 

/// Dual fermion evaluation of a spin-symmetric Hubbard model without spatial symmetry breaking
template <typename LatticeT> 
class df_hubbard : public df_base<LatticeT>
{
public:
    typedef df_base<LatticeT> base;
    using typename base::diagram_traits;
    using typename base::lattice_t;
    using typename base::vertex_type;
    using typename base::fvertex_type;
    using typename base::gw_type;
    using typename base::gk_type;
    using typename base::disp_type;
    using typename base::bz_point;
    using base::NDim;
    using typename base::vertex_eval_type;
    typedef diagrams::BetheSalpeter<matrix_type, true> forward_bs;
    // storage for the full equal-fermionic freq vertex, a.k.a w, w+W, w+W, w
    typedef typename diagram_traits::full_diag_vertex_type full_diag_vertex_type;

    /// Constructor
    df_hubbard(gw_type gw, gw_type Delta, lattice_t lattice, kmesh kgrid, vertex_type d_vertex, vertex_type m_vertex):
        base(gw,Delta,lattice,kgrid),
        density_vertex_(d_vertex),
        magnetic_vertex_(m_vertex),
        d_evals_(std::tuple_cat(std::forward_as_tuple(this->magnetic_vertex_.grid()), this->disp_.grids())),
        m_evals_(d_evals_.grids()) 
        {}

    /// Define common parameters
    static alps::params& define_parameters(alps::params &p);
    /// Run the df calculation and return the updated hybridization function
    virtual gw_type operator()(alps::params p);
    /// Get spin susceptibility at fixed frequency
    disp_type spin_susc(bmatsubara_grid::point W, bool add_lattice_bubble = false) const { return std::forward<disp_type>(get_susc_(magnetic_vertex_, W, 0.5, add_lattice_bubble)); }
    /// Get charge susceptibility at fixed frequency
    disp_type charge_susc(bmatsubara_grid::point W, bool add_lattice_bubble = false) const { return std::forward<disp_type>(get_susc_(density_vertex_, W, 1.0, add_lattice_bubble)); }
    /// The grid of bosonic frequencies of the vertex (note - the number of bosonic freqs for the calculation is controlled by a separate parameter)
    bmatsubara_grid const& bgrid() const { return std::get<0>(magnetic_vertex_.grids()); }
    /// Calculate and save the equal fermionic-frequency component of the full dual vertex
    void calc_full_diag_vertex(alps::params p); 
    /// Retrieve the equal fermionic-frequency component of the full dual vertex
    full_diag_vertex_type const& full_diag_vertex() const;
    /// Return the fluctuation diagnostics (Q-dependent contributions to 1) dual and 2) lattice self-energy) for a given set of points in the Brilloin zone
    std::tuple<std::vector<full_diag_vertex_type>, std::vector<full_diag_vertex_type>> fluctuation_diagnostics(std::vector<bz_point> kpoints, bool self_check = false) const;

protected:
    disp_type get_susc_(vertex_type const& in, bmatsubara_grid::point W, double norm, bool add_lattice_bubble) const;
    using base::fgrid_;
    using base::kgrid_;
    using base::gd_;
    using base::gd0_;
    using base::sigma_d_;
    using base::disp_;
    using base::gw_;
    using base::delta_;
    using base::glat_;
    // a pointer to store the diagonal (in freq) component of the full vertex, that is used in the Schwinger-Dyson equation for the self-energy
    std::unique_ptr<typename diagram_traits::full_diag_vertex_type> full_diag_vertex_ptr_; 

    /// An impurity vertex in the density channel
    vertex_type density_vertex_;
    /// An impurity vertex in the magnetic channel
    vertex_type magnetic_vertex_;
    /// Largest eigenvalue of density vertex at different W,Q
    vertex_eval_type d_evals_;
    /// Largest eigenvalue of magnetic vertex at different W,Q
    vertex_eval_type m_evals_;
};

} // end of namespace open_df
