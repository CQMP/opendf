#pragma once 
#include <alps/params.hpp>

#include <opendf/config.hpp>
#include <opendf/diagrams.hpp>

namespace open_df { 
            
template <typename LatticeT>
class df_base { 
public:
    typedef diagrams::diagram_traits<LatticeT> diagram_traits;
    typedef typename diagram_traits::lattice_t lattice_t;
    typedef typename diagram_traits::vertex_type vertex_type;
    typedef typename diagram_traits::fvertex_type fvertex_type;
    typedef typename diagram_traits::gw_type gw_type;
    static constexpr int NDim = diagram_traits::NDim; 
    typedef typename diagram_traits::gk_type gk_type;
    typedef typename diagram_traits::disp_type disp_type;
    typedef typename diagram_traits::vertex_eval_type vertex_eval_type;

    df_base(gw_type gw, gw_type Delta, lattice_t lattice, kmesh kgrid);
    void reload(gw_type gw, gw_type Delta);
            
    virtual gw_type operator()(alps::params p)=0;
    gk_type const& gd0() const { return this->gd0_;}
    gk_type const& gd() const { return this->gd_;}

    /// Bare lattice k-dependent GF
    gk_type glat_dmft() const;
    gw_type sigma_dmft(double mu = 0) const;
    gw_type delta() const { return delta_; }

    gk_type glat() const { return this->glat_; }
    gk_type sigma_lat(double mu = 0) const;
    gk_type const& sigma_d() const { return this->sigma_d_; }
    gw_type glat_loc() const;

    disp_type dispersion() const { return disp_; }

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

} // end of namespace open_df
