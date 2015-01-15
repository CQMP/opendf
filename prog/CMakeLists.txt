
#fk_diagrams 
set (execs 
hub_diagrams
)

foreach(exec ${execs})
    add_executable(${exec} ${exec}.cpp)
    target_link_libraries(${exec} ${opendf_name})
    if (NOT GFTOOLS_FOUND)
        add_dependencies(${exec} gftools)
    endif()
    install ( TARGETS ${exec} DESTINATION bin )
endforeach()


