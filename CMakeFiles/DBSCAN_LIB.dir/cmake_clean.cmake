file(REMOVE_RECURSE
  "libDBSCAN_LIB.a"
  "libDBSCAN_LIB.pdb"
)

# Per-language clean rules from dependency scanning.
foreach(lang )
  include(CMakeFiles/DBSCAN_LIB.dir/cmake_clean_${lang}.cmake OPTIONAL)
endforeach()
