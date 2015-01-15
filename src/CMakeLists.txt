include_directories (.)

set (df_src 
    diagrams.hpp
    df.hpp df.cpp 
    )

add_library(open_df${DF_POSTFIX} SHARED ${df_src})

install(TARGETS open_df${DF_POSTFIX}
    ARCHIVE DESTINATION lib
    LIBRARY DESTINATION lib
    )

install(DIRECTORY . DESTINATION include/open_df/
          FILES_MATCHING PATTERN "*.h *.hpp")
