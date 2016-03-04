set (df_src 
    diagrams.cpp
    df_base.cpp 
    df_hubbard.cpp 
    )

#add_library(${opendf_name} SHARED ${df_src})
add_library(${opendf_name} STATIC ${df_src})
set_target_properties(${opendf_name} PROPERTIES POSITION_INDEPENDENT_CODE ON)

install(TARGETS ${opendf_name}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    )

