
#fk_diagrams 
set (execs 
hub_df
)

foreach(exec ${execs})
    add_executable(${exec} ${exec}.cpp)
    target_link_libraries(${exec} ${opendf_name})
    if (NOT GFTOOLS_FOUND)
        add_dependencies(${exec} gftools)
    endif()
    install ( TARGETS ${exec} DESTINATION bin )
endforeach()

if (BuildPython) 
    # build python module
    foreach(exec ${execs})
        add_library(py${exec} MODULE ${exec}.cpp)
        target_link_libraries(py${exec} ${opendf_name})
        target_compile_definitions(py${exec} PUBLIC -DBUILD_PYTHON_MODULE -DLATTICE_${df_lattice})
        install ( TARGETS py${exec} DESTINATION lib/python )
    endforeach()
endif (BuildPython) 
