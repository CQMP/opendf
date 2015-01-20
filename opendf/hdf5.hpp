#pragma once

#include <string>
#include <boost/algorithm/string/classification.hpp>
#include <boost/algorithm/string/split.hpp>

#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wunused-variable" 
    #include <alps/hdf5.hpp>
    //#include "multi_array.hpp"
#pragma GCC diagnostic pop

#include <gftools.hpp>
#include "multi_array.hpp"


namespace alps{ 
    namespace hdf5 { 
        inline void save(alps::hdf5::archive & ar, std::string const & path, const gftools::enum_grid& x ); 
        inline void save(alps::hdf5::archive & ar, std::string const & path, const gftools::real_grid& x ); 
    }
}

namespace gftools {
// expand alps::hdf5::load to bold_hyb objects

// loader of various objects 
template <typename> struct hdf5_loader;
// load objects from hdf5 archive
template <typename T>
T load(alps::hdf5::archive & ar, std::string const & path) { return hdf5_loader<T>::load(ar,path); }
//template <typename T>
//void save(alps::hdf5::archive & ar, std::string const & path, const T& o) { return hdf5_loader<T>::save(ar,path,o); }

//
// gftools - related
//
// load gftools::container
template <typename T, size_t D>
struct hdf5_loader<container<T,D>> {
    typedef container<T,D> type;
    typedef typename container<T,D>::boost_t boost_type;
    static type load (alps::hdf5::archive & ar, std::string const & path) {
        boost_type c1; ar >> alps::make_pvp(path,c1); return container_ref<T,D>(c1); }
    static void save (alps::hdf5::archive & ar, std::string const & path, const container<T,D>& c) {
        ar << alps::make_pvp(path, c.boost_container_()); }
};

// grid
template <typename Grid, typename R = Grid>
using isGrid = typename std::enable_if<
    (sizeof(typename Grid::point) > 0 && 
    sizeof(decltype(std::declval<Grid>()[0]))>0), R>::type;

/*
template <class Grid>
inline isGrid<Grid, void> save(alps::hdf5::archive & ar, std::string const & path, const Grid& c)
{
    ar << alps::make_pvp(path+"/values", c.values());
}
*/

namespace extra { 
template <typename G>
inline G load_grid (alps::hdf5::archive & ar, std::string const & path)
{
    std::vector<typename G::point::value_type> g;
    ar[path+"/values"] >> g;
    return G(g);
}
}

#define GRID_HDF5_LOADER(T,R)                                                                           \
    template <> struct hdf5_loader<T> {                                                                 \
        static void save(alps::hdf5::archive & ar, std::string const & path, const T& x) {              \
            std::vector<R> y(x.size()); for (int i=0; i<y.size(); ++i) y[i] = x[i].val_;                \
            ar << alps::make_pvp(path+"/values",y); }                                                   \
        static T load(alps::hdf5::archive & ar, std::string const & path) {                             \
            std::vector<R> g; ar[path+"/values"] >> g; return T(g); }                                   \
        };                                                                                                                                                  
GRID_HDF5_LOADER(real_grid, typename real_grid::value_type);
GRID_HDF5_LOADER(enum_grid, int);
GRID_HDF5_LOADER(matsubara_grid<true>, typename matsubara_grid<true>::value_type);
GRID_HDF5_LOADER(matsubara_grid<false>, typename matsubara_grid<false>::value_type);
GRID_HDF5_LOADER(kmesh, typename kmesh::value_type);
#undef GRID_HDF5_LOADER 




// tuple of grids
template <typename, int N=0> struct hdf5_grid_tuple;
template <typename G1, typename ... Grids, int N> 
struct hdf5_grid_tuple<std::tuple<G1, Grids...>,N> 
{
    typedef std::tuple<G1, Grids...> grid_tuple;
    static void save(alps::hdf5::archive & ar, std::string const & path, const grid_tuple& t) { 
        std::string index_str = std::to_string(N); 
        hdf5_loader<G1>::save(ar, path + "/"+index_str, std::get<0>(t)); 
        hdf5_grid_tuple<std::tuple<Grids...>,N+1>::save(ar,path,tuple_tools::tuple_tail(t));
        }

    static grid_tuple load(alps::hdf5::archive & ar, std::string const & path) { 
        std::string index_str = std::to_string(N); 
        G1 grid = gftools::load<G1>(ar, path + "/"+index_str);
        return std::tuple_cat(std::make_tuple(grid), hdf5_grid_tuple<std::tuple<Grids...>,N+1>::load(ar,path));
        }
};

template <typename Grid, int N> 
struct hdf5_grid_tuple<std::tuple<Grid>,N>
{
    typedef std::tuple<Grid> grid_tuple;
    static void save(alps::hdf5::archive & ar, std::string const & path, const std::tuple<Grid>& t) { 
        std::string index_str = std::to_string(N); 
        hdf5_loader<Grid>::save(ar, path + "/"+index_str, std::get<0>(t)); 
    }
    static grid_tuple load(alps::hdf5::archive & ar, std::string const & path) { 
        std::string index_str = std::to_string(N); 
        return std::make_tuple(gftools::load<Grid>(ar, path + "/"+index_str)); 
    }
};

// gridobject
template <typename T> 
inline void save_grid_object(alps::hdf5::archive & ar, std::string const & path, const T& c, bool plaintext = false) 
{
    std::cout << "hdf5 : saving " << typeid(T).name() << " to " << path << std::endl;
    hdf5_grid_tuple<typename T::grid_tuple>::save(ar,path+"/grids",c.grids());
    save(ar,path+"/data", c.data());
    std::string p2(path);
    std::vector<std::string> split_vec; 
    boost::algorithm::split(split_vec, path, boost::is_any_of("/"), boost::token_compress_on );
    if (plaintext) c.savetxt(split_vec[split_vec.size()-1]+".dat");
}

template <typename T> 
inline T load_grid_object(alps::hdf5::archive & ar, std::string const & path) {  
    std::cout << "hdf5 : loading " << typeid(T).name() << " from " << path << std::endl;
    auto grids = hdf5_grid_tuple<typename T::grid_tuple>::load(ar,path+"/grids");
    auto data = hdf5_loader<typename T::container_type>::load(ar,path+"/data");
    return T(grids,data);
}
} // end of namespace bold_hyb

namespace alps{ 
    namespace hdf5 { 
        template <typename T, size_t D>
        inline void save(alps::hdf5::archive & ar, std::string const & path, const gftools::container<T,D>& c) { 
            gftools::hdf5_loader<gftools::container<T,D>>::save(ar, path, c); }
        inline void save(alps::hdf5::archive & ar, std::string const & path, const gftools::enum_grid& x ) { 
            gftools::hdf5_loader<gftools::enum_grid>::save(ar, path, x); }  
        inline void save(alps::hdf5::archive & ar, std::string const & path, const gftools::real_grid& x ) { 
            gftools::hdf5_loader<gftools::real_grid>::save(ar, path, x); }  
    }
}
