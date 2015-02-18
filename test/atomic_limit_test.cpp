#include <alps/params.hpp>
#include <gtest/gtest.h>

#include "opendf/lattice_traits.hpp"
#include "opendf/df_hubbard.hpp"

using namespace open_df;

static constexpr int D = 2; // work in 2 dimensions
typedef cubic_traits<D> lattice_t; 
typedef df_hubbard<lattice_t> df_type; 

typedef df_type::vertex_type vertex_type;
typedef df_type::gw_type gw_type;
typedef df_type::gk_type gk_type;
typedef df_type::disp_type disp_type;

int main(int argc, char *argv[])
{
    alps::params p; 

    double U = 16; 
    double beta = 1.0;
    double mu = U/2;
    int wmax = 16; //16;
    int kpts = 16;

    typedef std::complex<double> complex_type;
    typedef typename fmatsubara_grid::point fpoint; 
    typedef typename bmatsubara_grid::point bpoint; 

    // grids
    fmatsubara_grid fgrid(-wmax,wmax,beta);
    bmatsubara_grid bgrid(-1,2,beta);
    DEBUG(bgrid);
        
    auto mult = beta*U*U/4.;
    gw_type Lambda(fgrid);
    typename gw_type::function_type lambdaf = [mult,U](std::complex<double> w){return 1. - U*U/4./w/w;};
    Lambda.fill(lambdaf);

    // magnetic vertex
    vertex_type magnetic_vertex(std::forward_as_tuple(bgrid,fgrid,fgrid));
    vertex_type::point_function_type VertexF;
    VertexF = [&](bpoint W,fpoint w1, fpoint w2)->std::complex<double>{
            return  double(is_float_equal(W.value(), 0))*mult*Lambda(w1)*Lambda(w2)*(2. + double(w1.index() == w2.index()));
    };
    magnetic_vertex.fill(VertexF);

    // density vertex
    vertex_type density_vertex(magnetic_vertex.grids());
    density_vertex = 0;
    VertexF = [&](bpoint W,fpoint w1, fpoint w2)->std::complex<double>{
            return  double(is_float_equal(W.value(), 0))*mult*Lambda(w1)*Lambda(w2)*(-3.0)*double(w1.index() == w2.index());
    };
    //density_vertex = mult*Lambda*Lambda*(-3.);  // Triplet vertex S_z = 1, S = 1
    density_vertex.fill(VertexF);

    // local gf
    std::function<complex_type(fpoint)> gw_f = [U](fpoint w) { return 0.5 / (w.value() - U/2.) + 0.5 / (w.value() + U/2.); };
    gw_type gw(fgrid);
    gw.fill(gw_f);

    gw_type delta(gw.grids());
    delta = gw * 2 * double(D);

    gw.savetxt("gw_test.dat");
    delta.savetxt("delta_test.dat");
    
    // parameters
    double hopping_t = 1.0;
    double T = 1.0/beta;

    p["df_sc_mix"] = 1.0;
    p["df_sc_iter"] = 1;
    p["nbosonic"] = 1;
    //std::cout << p << std::flush;
    std::cout << "temperature = " << T << std::endl;
    std::cout << std::endl;

    // create a grid over Brilloin zone for evaluation 
    kmesh kgrid(kpts);
    
    // define lattice
    lattice_t lattice(hopping_t);
    // get dispersion
    disp_type disp(std::forward_as_tuple(kgrid, kgrid));
    disp.fill(lattice.get_dispersion());  
        
    // construct a df run 
    df_hubbard<cubic_traits<2>> DF(gw, delta, lattice, kgrid, density_vertex, magnetic_vertex);
    // run df
    gw_type delta_upd = DF(p);
    gw_type glat_loc = DF.glat_loc();
    glat_loc.savetxt("gloc_test.dat");

    auto w0 = fgrid.find_nearest(I*PI/beta);

    gw_type gloc_comp(fgrid);
    std::vector<double> v = {{  
        -4.175286161764e-02,
        -6.151136778808e-02,
        -5.052277618879e-02,
        -4.015155827793e-02,
        -3.274400701650e-02,
        -2.746448171460e-02,
        -2.358018909282e-02,
        -2.062600757087e-02
            }};
    for (int i=0; i<std::min(v.size(), glat_loc.size()/2); i++) { 
        std::cout << i << " : " << glat_loc[w0.index() + i] << " == " << v[i]*I << std::endl;
        EXPECT_EQ(is_float_equal(glat_loc[w0.index() + i], v[i]*I, 1e-4), true);
        };
    std::cout << "SUCCESS" << std::endl;

    // get spin susceptibility
    disp_type spin_susc = DF.spin_susc(bgrid.find_nearest(0.0));
    std::cout << "susc at [pi, pi] = " << spin_susc(PI,PI) << " == " << 3.468082768315e-01 <<  std::endl;
    EXPECT_NEAR(spin_susc(PI,PI).real(), 3.468082768315e-01, 1e-3);
    EXPECT_NEAR(spin_susc(PI,PI).imag(), 0.0, 1e-4);
    std::cout << "susc at [0 , pi] = " << spin_susc(0, PI) << std::endl;
    EXPECT_NEAR(spin_susc(0,PI).real(), 2.503368337900e-01, 1e-3);
    EXPECT_NEAR(spin_susc(0,PI).imag(), 0.0, 1e-4);
    std::cout << "susc at [0 , 0 ] = " << spin_susc(0, 0)  << " == " << 1.977592985438e-01 << std::endl;
    EXPECT_NEAR(spin_susc(0,0).real(), 1.977592985438e-01, 1e-3);
    EXPECT_NEAR(spin_susc(0,0).imag(), 0.0, 1e-4);
    std::cout << "SUCCESS" << std::endl;
}

