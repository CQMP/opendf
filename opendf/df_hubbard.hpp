#pragma once 
#include "df_base.hpp"

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
    using base::NDim;
    using typename base::vertex_eval_type;
    typedef diagrams::BetheSalpeter<matrix_type, true> forward_bs;

    /// Constructor
    df_hubbard(gw_type gw, gw_type Delta, lattice_t lattice, kmesh kgrid, vertex_type d_vertex, vertex_type m_vertex):
        base(gw,Delta,lattice,kgrid),
        density_vertex_(d_vertex),
        magnetic_vertex_(m_vertex),
        d_evals_(std::tuple_cat(std::forward_as_tuple(this->magnetic_vertex_.grid()), this->disp_.grids())),
        m_evals_(d_evals_.grids()) 
        {}
    /// Run the df calculation and return the updated hybridization function
    virtual gw_type operator()(alps::params p);
    /// Get spin susceptibility at fixed frequency
    disp_type spin_susc(bmatsubara_grid::point W) const { return std::forward<disp_type>(get_susc_(magnetic_vertex_, W, 0.5)); }
    disp_type charge_susc(bmatsubara_grid::point W) const { return std::forward<disp_type>(get_susc_(density_vertex_, W, 1.0)); }

    bmatsubara_grid const& bgrid() const { return std::get<0>(magnetic_vertex_.grids()); }

protected:
    disp_type get_susc_(vertex_type const& in, bmatsubara_grid::point W, double norm) const;
    using base::fgrid_;
    using base::kgrid_;
    using base::gd_;
    using base::gd0_;
    using base::sigma_d_;
    using base::disp_;
    using base::gw_;
    using base::delta_;
    using base::glat_;
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
