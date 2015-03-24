#
# dual_diagrams cmake configuration file
#

cmake_minimum_required (VERSION 2.8)
message ("\ndual diagrams configuration\n")
# mac os rpath policy
if (POLICY CMP0042)
    cmake_policy(SET CMP0042 NEW)
endif()

# Disable build in source
message(STATUS ${CMAKE_MODULE_PATH})
set(CMAKE_DISABLE_SOURCE_CHANGES ON)
list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake) # find additional cmake module in cmake dir

# title and version of the project
project (opendf CXX)
# Print build type
message(STATUS "Build type: ${CMAKE_BUILD_TYPE}")
# some compiler flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wall -Wno-deprecated-register")
# workaround definitions for different compilers
if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
    # disable "undefined template warning" frequently shown by intel
    add_definitions("-wd488")
endif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")

# set c++11 flags automatically or let the user do it
option(AutoSetCXX11 "Auto set c++11 flags" ON)
if (AutoSetCXX11)
    set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11")

    if ("${CMAKE_CXX_COMPILER_ID}" MATCHES "GNU")
        execute_process(
            COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE GCC_VERSION)
            set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -Wno-unused-local-typedefs")
        if (NOT (GCC_VERSION VERSION_GREATER 4.7 OR GCC_VERSION VERSION_EQUAL 4.7))
            message(FATAL_ERROR "${PROJECT_NAME} requires g++ 4.7 or greater.")
        endif ()
    elseif ("${CMAKE_CXX_COMPILER_ID}" MATCHES "Clang" OR "${CMAKE_CXX_COMPILER_ID}" MATCHES "Intel")
        if ("${CMAKE_SYSTEM_NAME}" MATCHES "Darwin")
            message( STATUS "Adding -stdlib=libc++ flag")
            set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -stdlib=libc++")
        endif()
    else ()
        message(FATAL_ERROR "Your C++ compiler does not support C++11.")
    endif ()
endif (AutoSetCXX11)

# RPATH settings (see http://www.cmake.org/Wiki/CMake_RPATH_handling) 
#SET(CMAKE_SKIP_BUILD_RPATH  FALSE)
#SET(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE) 
#SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#SET(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)
#LIST(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
#    "${CMAKE_INSTALL_PREFIX}/lib" isSystemDir)
#IF("${isSystemDir}" STREQUAL "-1")
#    SET(CMAKE_INSTALL_RPATH "${CMAKE_INSTALL_PREFIX}/lib")
#ENDIF("${isSystemDir}" STREQUAL "-1")


### Dependencies
# Eigen
find_package (Eigen3 3.1 REQUIRED)
    message(STATUS "Eigen3 includes: " ${EIGEN3_INCLUDE_DIR} )
    include_directories(${EIGEN3_INCLUDE_DIR})
# Boost

find_package (Boost REQUIRED COMPONENTS program_options)
include_directories(${Boost_INCLUDE_DIRS})
message(STATUS "Boost includes: " ${Boost_INCLUDE_DIRS} )
message(STATUS "Boost libraries: " ${Boost_LIBRARIES} )
link_libraries(${Boost_LIBRARIES})

# gftools 
find_package(GFTools QUIET)
    if (NOT GFTOOLS_FOUND)
        message(STATUS "Fetching gftools")
        include(ExtGFTools)
    endif()

    message(STATUS "GFTools includes : ${GFTOOLS_INCLUDE_DIR}") 
    include_directories(${GFTOOLS_INCLUDE_DIR})
# HDF5
find_package (HDF5 REQUIRED)
    message(STATUS "HDF5 includes: " ${HDF5_INCLUDE_DIRS} )
    message(STATUS "HDF5 libs: " ${HDF5_LIBRARIES} )
    include_directories(${HDF5_INCLUDE_DIRS})
    link_libraries(${HDF5_LIBRARIES})
# ALPSCore 
find_package(ALPSCore REQUIRED COMPONENTS hdf5 accumulators mc params)
    include_directories(${ALPSCore_INCLUDE_DIRS})
    link_libraries(${ALPSCore_LIBRARIES}) 

# Boost + optional : python  + boost::python + alpscore::python 
find_package (Boost REQUIRED COMPONENTS serialization filesystem system chrono program_options)
    message(STATUS "Boost includes: " ${Boost_INCLUDE_DIRS} )
    message(STATUS "Boost libraries: " ${Boost_LIBRARIES} )
    include_directories(${Boost_INCLUDE_DIRS})
    link_libraries(${Boost_LIBRARIES})

# MPI
option(EnableMPI "Use MPI" ON)
if (EnableMPI)

    find_package (MPI)
        message(STATUS "MPI includes: " ${MPI_INCLUDE_DIRS} )
        message(STATUS "MPI CXX libs: " ${MPI_CXX_LIBRARIES} )
        message(STATUS "MPI definitions : " ${MPI_CXX_COMPILE_FLAGS} ${MPI_C_COMPILE_FLAGS})
        add_definitions(${MPI_CXX_COMPILE_FLAGS}  ${MPI_C_COMPILE_FLAGS})
        include_directories(${MPI_INCLUDE_DIRS})
        link_libraries(${MPI_CXX_LIBRARIES})

    find_package (Boost REQUIRED COMPONENTS mpi)
    message(STATUS "Boost mpi includes: " ${Boost_INCLUDE_DIRS} )
    message(STATUS "Boost mpi libraries: " ${Boost_LIBRARIES} )
    include_directories(${Boost_INCLUDE_DIRS})
    link_libraries(${Boost_LIBRARIES})

    set(OPENDF_ENABLE_MPI TRUE)
endif(EnableMPI)

option(BuildPython "Build python modules" OFF)
if (BuildPython) 
    find_package (Boost REQUIRED COMPONENTS python)
    include_directories(${Boost_INCLUDE_DIRS})
    link_libraries(${Boost_LIBRARIES})
    #find_package(ALPSCore REQUIRED COMPONENTS hdf5 accumulators mc params python)
    find_package (PythonLibs REQUIRED)
    include_directories(${PYTHON_INCLUDE_DIRS})
    message(STATUS "Python includes: " ${PYTHON_INCLUDE_DIRS})
    message(STATUS "Python libraries: " ${PYTHON_LIBRARIES})
    link_libraries(${PYTHON_LIBRARIES})
    set(BOLD_HYB_BUILD_PYTHON True)
endif(BuildPython)
    

# fftw
find_package (fftw REQUIRED)
    message(STATUS "FFTW includes: " ${FFTW_INCLUDE_DIR} )
	include_directories(${FFTW_INCLUDE_DIR})
    link_libraries(${FFTW_LIBRARIES})

### add source/binary root of the project to included path
include_directories(
    ${CMAKE_SOURCE_DIR}
    ${CMAKE_BINARY_DIR}
)

### configure and build everything 
# Check which lattices should be built
# List of all known lattices
list(APPEND known_lattices cubic1d cubic2d cubic3d cubic4d)
# List of lattices, compiled by default
list(APPEND default_lattices cubic2d cubic3d)
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

# build library
set (opendf_name opendf${DF_POSTFIX})
add_subdirectory(opendf)
# build executables
option(Execs "Enable executables" ON)
if (Execs)
    add_subdirectory(prog)
endif (Execs)

# build tests
option(Testing "Enable testing" ON)
if (Testing)
    enable_testing()
    add_subdirectory(test)
endif (Testing)
# build misc tools - converters, etc
add_subdirectory(tools)

### Additional installation files
configure_file("${CMAKE_SOURCE_DIR}/opendf.lmod.in" "${CMAKE_BINARY_DIR}/opendf.lmod")
