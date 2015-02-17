#pragma once 
#include <alps/params.hpp>

#include <opendf/config.hpp>
#include <opendf/diagrams.hpp>

namespace open_df { 
            
/// A base class for dual fermion calculations for translational- and spin-invariant systems
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

    /// Constructor
    /// \param[im] gw $g_\omega$ Green's function of the impurity 
    /// \param[im] Delta $\Delta_\omega$ Hybridization function 
    /// \param[in] lattice A LatticeTraits class that defines the lattice
    /// \param[in] kgrid A grid of kpoints that samples one dimension of reciprocal space
    df_base(gw_type gw, gw_type Delta, lattice_t lattice, kmesh kgrid);

    /// Set the gd manually. Useful for providing initial guess with better convergence
    void set_gd(gk_type&& gd_initial); 
    /// Reset all calculated quantitities
    void reload(gw_type gw, gw_type Delta, bool flush_gd = true);
            
    /// Perform the DF calculation - return updated hybridization function
    virtual gw_type operator()(alps::params p)=0;
    /// Return bare dual Green's function
    gk_type const& gd0() const { return this->gd0_;}
    /// Return dressed dual Green's function
    gk_type const& gd() const { return this->gd_;}

    /// Bare lattice k-dependent GF
    gk_type glat_dmft() const;
    /// Return dmft self-energy  
    gw_type sigma_dmft(double mu = 0) const;
    /// Return stored hybridization function
    gw_type delta() const { return delta_; }

    /// Return dressed lattice Green's function
    gk_type glat() const { return this->glat_; }
    /// Return lattice self-energy
    gk_type sigma_lat(double mu = 0) const;
    /// Return dual self-energy
    gk_type const& sigma_d() const { return this->sigma_d_; }
    /// Return local part of the lattice GF
    gw_type glat_loc() const;

    /// Return lattice dispersion
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
