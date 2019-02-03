#
# dual_diagrams cmake configuration file
#

cmake_minimum_required (VERSION 2.8)
# title and version of the project
project (opendf CXX)
message ("\n${PROJECT_NAME} configuration\n")
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) # find additional cmake module in cmake dir
# check ./cmake/CommondDefs.cmake for all of the macro/functions below
include(CommonDefs)

# Disable build in source
no_source_builds()
fix_rpath()
# Workarounds for bugs in compilers
compiler_workarounds()
# C++11
option(AutoSetCXX11 "Auto set c++11 flags" ON)
if (AutoSetCXX11)
    set_cxx11()
endif (AutoSetCXX11)
# Print build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")

# Find ALPSCore package for alpscore dependency
find_package(ALPSCore REQUIRED COMPONENTS hdf5 accumulators mc params HINTS ${ALPS_ROOT})

# Dependencies
add_alpscore()
if (NOT ALPSCore_HAS_EIGEN_VERSION)
    add_eigen3()
endif (NOT ALPSCore_HAS_EIGEN_VERSION)
add_gftools()
option(BuildPython "Build python modules" OFF)
if (BuildPython)
    add_python()
endif (BuildPython)

# MPI
option(EnableMPI "Use MPI" ON)
if (EnableMPI AND ALPSCore_HAS_MPI)
     set(OPENDF_ENABLE_MPI TRUE)
endif()

# fftw3
add_fftw3()

# https://stackoverflow.com/questions/25365160/boostmultiprecisionfloat128-and-c11
add_definitions("-fext-numeric-literals")

### add source/binary root of the project to included path
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}
)

### configure and build everything 
# Check which lattices should be built
# List of all known lattices
list(APPEND known_lattices cubic1d cubic2d cubic3d cubic4d triangular square_nnn)
# List of lattices, compiled by default
list(APPEND default_lattices ${known_lattices})
# if no lattices specified - build all 
list(LENGTH DF_LATTICES comp_len)
if (${comp_len} EQUAL 0) 
    set(DF_LATTICES ${default_lattices})
endif()

unset(df_lattices)
foreach(lattice ${DF_LATTICES})
    list(FIND known_lattices ${lattice} has_lattice)
    if (${has_lattice} EQUAL -1) 
        message(FATAL_ERROR "opendf : Unknown lattice ${lattice}")
    else()
        list(FIND df_lattices ${lattice} already_included)
        if (${already_included} EQUAL -1) 
            list(APPEND df_lattices ${lattice})
        endif()
    endif()
endforeach()

message(STATUS "Lattices : ${df_lattices}")
mark_as_advanced(${df_lattices})
set (DF_LATTICES ${df_lattices} CACHE STRING "Lattices for DF calculation" FORCE)

# configure header
configure_file(${CMAKE_SOURCE_DIR}/include/opendf/config.hpp.in ${CMAKE_BINARY_DIR}/include/opendf/config.hpp)
include_directories(${CMAKE_SOURCE_DIR}/include/ ${CMAKE_BINARY_DIR}/include/)

# build library
set (opendf_name opendf${DF_POSTFIX})
add_subdirectory(src)
# build executables
option(Execs "Enable executables" ON)
if (Execs)
    add_subdirectory(prog)
endif (Execs)

# build tests
option(Testing "Enable testing" OFF)
if (Testing)
    enable_testing()
    add_subdirectory(test)
endif (Testing)
# build misc tools - converters, etc
add_subdirectory(tools)

# install headers
install(DIRECTORY ${CMAKE_SOURCE_DIR}/include DESTINATION include FILES_MATCHING PATTERN "*.hpp")
install(DIRECTORY ${CMAKE_BINARY_DIR}/include DESTINATION include FILES_MATCHING PATTERN "*.hpp")


### Additional installation files
configure_file("${CMAKE_SOURCE_DIR}/opendf.lmod.in" "${CMAKE_BINARY_DIR}/opendf.lmod")
