
#fk_diagrams 
set (execs 
hub_df
)


foreach (df_lattice ${df_lattices})
    message(STATUS "DF: ${df_lattice}")
    foreach(exec_name ${execs})
        set(exec ${exec_name}_${df_lattice})
        add_executable(${exec} ${exec_name}.cpp)
        set_target_properties(${exec} PROPERTIES COMPILE_DEFINITIONS "LATTICE_${df_lattice}") 
        target_link_libraries(${exec} ${opendf_name})
        if (NOT GFTOOLS_FOUND)
            add_dependencies(${exec} gftools)
        endif()
        install (TARGETS ${exec} DESTINATION bin )
    endforeach(exec_name)

    if (BuildPython) 
        # build python module
        foreach(exec_name ${execs})
            set(exec ${exec_name}_${df_lattice})
            add_library(py${exec} MODULE ${exec_name}.cpp)
            target_link_libraries(py${exec} ${opendf_name})
            target_compile_definitions(py${exec} PUBLIC -DBUILD_PYTHON_MODULE -DLATTICE_${df_lattice})
            install ( TARGETS py${exec} DESTINATION lib/python )
        endforeach()
    endif (BuildPython) 

endforeach(df_lattice)
