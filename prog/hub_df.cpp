#include <opendf/config.hpp>

#ifdef OPENDF_ENABLE_MPI
#include <alps/utilities/boost_mpi.hpp>
#define mpi_cout if (!comm.rank()) std::cout
#else
#define mpi_cout std::cout
#endif

#include <chrono>
#include <alps/params.hpp>
//#include "alps/hdf5.hpp"


#include <opendf/lattice_traits.hpp>
#include <opendf/df_hubbard.hpp>
#include <boost/filesystem.hpp>

#include "data_save.hpp"

#ifdef LATTICE_cubic1d
    static constexpr int D = 1; 
    typedef open_df::cubic_traits<D> lattice_t; 
    #define LATTICE_PARAMS 1
#elif LATTICE_cubic2d
    static constexpr int D = 2; 
    typedef open_df::cubic_traits<D> lattice_t; 
    #define LATTICE_PARAMS 1
#elif LATTICE_cubic3d
    static constexpr int D = 3; 
    typedef open_df::cubic_traits<D> lattice_t; 
    #define LATTICE_PARAMS 1
#elif LATTICE_cubic4d
    static constexpr int D = 4; 
    typedef open_df::cubic_traits<D> lattice_t; 
    #define LATTICE_PARAMS 1
#elif LATTICE_triangular
    static constexpr int D = 2; 
    typedef open_df::triangular_traits lattice_t; 
    #define LATTICE_PARAMS 2
#elif LATTICE_square_nnn
    static constexpr int D = 2; 
    typedef open_df::square_nnn_traits lattice_t; 
    #define LATTICE_PARAMS 2
#else 
#error Undefined lattice
#endif

using namespace open_df;
using namespace std::chrono;

typedef df_hubbard<lattice_t> df_type; 
typedef df_type::vertex_type vertex_type;
typedef df_type::gw_type gw_type;
typedef df_type::gk_type gk_type;
typedef df_type::disp_type disp_type;

#ifdef OPENDF_ENABLE_MPI
  alps::mpi::communicator comm;
#endif 

inline void print_section (const std::string& str)
{
  mpi_cout << std::string(str.size(),'=') << std::endl;
  mpi_cout << str << std::endl;
  mpi_cout << std::string(str.size(),'=') << std::endl;
}

void run(alps::params p)
{
    #ifdef OPENDF_ENABLE_MPI     
      mpi_cout<<"number of process "<<comm.size()<<std::endl;
      mpi_cout<<"rank id "<<comm.rank()<<std::endl;
    #endif

    print_section("DF ladder in " + std::to_string(D) + " dimensions.");
    // read input data
    std::string input_name = p["input"];
    std::string top = p["inp_section"];
    mpi_cout << "Reading data from " << input_name << " /" << top << std::endl;

    alps::hdf5::archive ar(input_name, "r");

    std::array<vertex_type, 2> vertex_input = {{ 
        load_grid_object<vertex_type>(ar, top + "/F00"), 
        load_grid_object<vertex_type>(ar, top + "/F01") }}; 

    std::array<gw_type, 2> gw_arr = {{ 
        load_grid_object<gw_type>(ar, top + "/gw0"), 
        load_grid_object<gw_type>(ar, top + "/gw1") }} ;

    std::array<gw_type, 2> delta_arr = {{ 
        load_grid_object<gw_type>(ar, top + "/delta0"), 
        load_grid_object<gw_type>(ar, top + "/delta1") }} ;

    // fix spin symmerty
    if (gw_arr[0].diff(gw_arr[1]) > 1e-8) 
        throw std::logic_error("Spin asymmetry is not implemented.");

    gw_type gw = gw_arr[0];
    gw_type Delta = delta_arr[0]; 

    vertex_type density_vertex = vertex_input[0] + vertex_input[1];
    vertex_type magnetic_vertex = vertex_input[0] - vertex_input[1];

    if (gw.grid() != vertex_input[0].template grid<1>()) 
        throw std::logic_error("Green's function and vertex are defined on different grids. Exiting.");


    // parameters
    double beta = gw.grid().beta();
    double T = 1.0/beta;
    int kpts = p["kpts"];
    p["beta"] = beta;

    print_section("parameters");
    mpi_cout << p << std::flush;
    mpi_cout << "temperature = " << T << std::endl;
    mpi_cout << std::endl;

    // create a grid over Brilloin zone for evaluation 
    kmesh kgrid(kpts);
    
    // define lattice
    #if LATTICE_PARAMS==1
    lattice_t lattice(p["t"].as<double>());
    #elif LATTICE_PARAMS==2
    lattice_t lattice(p["t"], p["tp"]);
    #endif
    // get dispersion
    disp_type disp(gftools::tuple_tools::repeater<kmesh, D>::get_tuple(kgrid));
    disp.fill(lattice.get_dispersion());  
        
    // construct a df run 
    df_type DF(gw, Delta, lattice, kgrid, density_vertex, magnetic_vertex);

    // check if a resume is requested and try to resume 
    if (p["resume"].as<bool>()) { 
        mpi_cout << "Trying to resume" << std::endl;
        std::string output_file = p["output"];
        bool resume = boost::filesystem::exists(output_file);
        if (!resume) { ERROR("Can't resume - no file " << output_file); }
        else { 
            std::string top = "/df";
            alps::hdf5::archive ar(output_file, "r");
            resume = ar.is_group(top + "/gd");
            if (!resume) { ERROR("Can't resume - no gd found in" << output_file << top << "/gd"); } 
            else { 
                gk_type gd_initial = load_grid_object<gk_type>(ar, top + "/gd"); 
                DF.set_gd(std::move(gd_initial));
                }
            }
        }
    // run df
    steady_clock::time_point start, end;
    start = steady_clock::now();
    for (int i = 0; i < 1; i++) { 
        DF.reload(gw, Delta, false);
        gw_type delta_upd = DF(p);
        Delta = delta_upd;
        }
    end = steady_clock::now();

    #ifdef OPENDF_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    if (!comm.rank()) 
    #endif
        { 
        //std::cout<<"comm.rank "<<comm.rank()<<std::endl;
        mpi_cout << "Calculation lasted : " 
            << duration_cast<hours>(end-start).count() << "h " 
            << duration_cast<minutes>(end-start).count()%60 << "m " 
            << duration_cast<seconds>(end-start).count()%60 << "s " 
            << duration_cast<milliseconds>(end-start).count()%1000 << "ms " 
            << std::endl;
        p["run_time"] = int(duration_cast<milliseconds>(end-start).count());

        if (p["fluct_diag"].as<bool>()) { 
            std::cout << "Extracting full equal-fermionic frequency vertex" << std::endl;
            DF.calc_full_diag_vertex(p);
            }

        save_data(DF, Delta, p); 

        if (p["plaintext"].as<int>() > 0) { 
            magnetic_vertex.savetxt("magnetic_vertex_input.dat");
            density_vertex.savetxt("density_vertex_input.dat");
            fmatsubara_grid const& fgrid = magnetic_vertex.template grid<1>();
            bmatsubara_grid const& bgrid = magnetic_vertex.template grid<0>();
            bmatsubara_grid::point W0 = bgrid.find_nearest(0.0);
            typename df_type::diagram_traits::fvertex_type m0(fgrid, fgrid);
            typename df_type::diagram_traits::fvertex_type d0(fgrid, fgrid);
            m0.data() = magnetic_vertex[W0];
            d0.data() = density_vertex[W0];
            m0.savetxt("magnetic_vertex_input_W0.dat");
            d0.savetxt("density_vertex_input_W0.dat");
            }
        }

    #ifdef OPENDF_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    #endif
}


//
// build command line executable
// 

#ifndef BUILD_PYTHON_MODULE
alps::params cmdline_params(int argc, char** argv);

int main(int argc, char *argv[])
{
    #ifdef OPENDF_ENABLE_MPI
    MPI_Init(&argc, &argv);
    #endif
    try { 
        alps::params p = cmdline_params(argc, argv); 
        run(p); 
        }
    catch (std::exception &e) { std::cerr << e.what() << std::endl; exit(1); };

    #ifdef OPENDF_ENABLE_MPI
    MPI_Barrier(MPI_COMM_WORLD);
    MPI_Finalize();
    #endif    
}


alps::params cmdline_params(int argc, char** argv)
{
    alps::params p(argc, argv);
    p.description("Dual fermions for the Hubbard model in " + std::to_string(D) + " dimensions.");
    df_type::define_parameters(p);



    save_define_parameters(p);

    p
     .define<double>("mu",           0.0,   "chemical potential")
     .define<int>("kpts",           16,     "number of points on a single axis in Brilloin zone")
     .define<std::string>("input",          "output.h5", "input file with vertices and gf")
     .define<std::string>("inp_section",    "dmft",      "input section (default : 'dmft')")
     .define<std::string>("output",         "output.h5", "output file")
     .define<bool>("resume", 0, "try resuming calculation - load dual gf from 'output' hdf5 file : /df/gd section");

    #if LATTICE_PARAMS == 1
    p.define<double>("t",  1.0,   "hopping on a lattice");
    #elif LATTICE_PARAMS == 2
    p.define<double>("t",  1.0,   "nearest neighbor hopping on a lattice");
    p.define<double>("tp", 0.0,   "next-nearest neighbor hopping on a lattice");
    #else
    #error Undefined lattice
    #endif

    if (p.help_requested(std::cerr)) { exit(1); };

    return p;
}
#endif 

//
// build python module
//

#ifdef BUILD_PYTHON_MODULE
#include <boost/python.hpp>
//compile it as a python module (requires boost::python library)

void solve(boost::python::dict py_parms)
{
    //alps::params p(parms_);
    //std::string output_file = boost::lexical_cast<std::string>(parms["BASENAME"]|"results")+std::string(".out.h5");
    alps::params p;
    #define PYCONV(x, T) p[x] = boost::python::extract<T>(py_parms.attr(x));
    PYCONV("df_sc_mix", double)
    PYCONV("df_sc_cutoff", double)
    PYCONV("bs_mix", double)
    PYCONV("hopping", double)
    
    PYCONV("kpts", int);
    PYCONV("df_sc_iter", int);
    PYCONV("nbosonic", int);
    PYCONV("n_bs_iter", int);
    PYCONV("plaintext", int);

    PYCONV("input", std::string);
    PYCONV("output", std::string);

    PYCONV("update_df_mixing", bool);
    PYCONV("eval_bs_sc", bool);
    PYCONV("resume", bool);
    #undef PYCONV
    run(p);
};

    BOOST_PYTHON_MODULE(libpyhub_df)
    {
        using namespace boost::python;
        def("solve",&solve); //define python-callable run method
    };
#endif


