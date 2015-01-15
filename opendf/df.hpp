#pragma once 
#include "config.hpp"
#include <alps/params.hpp>

namespace open_df { 

template <typename LatticeT>
class df_base { 
public:
    typedef LatticeT lattice_t;
    typedef grid_object<std::complex<double>, bmatsubara_grid, fmatsubara_grid, fmatsubara_grid> vertex_type;
    typedef grid_object<std::complex<double>, fmatsubara_grid, fmatsubara_grid> fvertex_type;
    typedef grid_object<std::complex<double>, fmatsubara_grid> gw_type;

    static constexpr int NDim = lattice_t::NDim; 

    typedef typename gftools::tools::ArgBackGenerator<NDim,kmesh,grid_object,complex_type,fmatsubara_grid>::type gk_type;
    typedef typename gftools::tools::ArgBackGenerator<NDim,kmesh,grid_object,complex_type>::type disp_type;
        
    df_base(gw_type gw, gw_type Delta, lattice_t lattice, kmesh kgrid);
            
    virtual gw_type operator()(alps::params p)=0;
    gk_type const& gd0() const { return this->gd0_;}
    gk_type const& gd() const { return this->gd_;}

    /// Bare lattice k-dependent GF
    gk_type glat_dmft() const;

    gk_type glat() const { return this->glat_; }
    gk_type lattice_selfenergy_correction() const;
    gk_type const& dual_selfenergy() const { return this->sigma_d_; }
    gw_type glat_loc() const;

protected: 
    /// Lattice to evaluate k-dependent integrals
    lattice_t lattice_;
    /// Fermionic Matsubara grid
    fmatsubara_grid fgrid_;
    /// Mesh in k-space to evaluate k-dependent integrals 
    kmesh kgrid_;
    /// Local GF of the impurity problem (qmc result)
    gw_type gw_;
    /// Hybridization function
    gw_type delta_;
    /// Lattice dispersion
    disp_type disp_; 
    /// Bare dual GF
    gk_type gd0_;
    /// Dressed dual GF
    gk_type gd_;
    /// Dual self-energy
    gk_type sigma_d_; 
    /// Lattice k-dependent GF
    gk_type glat_;
};

template <typename LatticeT> 
class df_hubbard : public df_base<LatticeT>
{
public:
    typedef df_base<LatticeT> base;
    using typename base::lattice_t;
    using typename base::vertex_type;
    using typename base::fvertex_type;
    using typename base::gw_type;
    using typename base::gk_type;
    using typename base::disp_type;
    using base::NDim;

    df_hubbard(gw_type gw, gw_type Delta, lattice_t lattice, kmesh kgrid, vertex_type d_vertex, vertex_type m_vertex):
        base(gw,Delta,lattice,kgrid),
        density_vertex_(d_vertex),
        magnetic_vertex_(m_vertex)
        {}
    virtual gw_type operator()(alps::params p);

protected:
    using base::fgrid_;
    using base::kgrid_;
    using base::gd_;
    using base::gd0_;
    using base::sigma_d_;
    using base::disp_;
    using base::gw_;
    using base::delta_;
    using base::glat_;
    vertex_type density_vertex_;
    vertex_type magnetic_vertex_;
};

} // end of namespace open_df
