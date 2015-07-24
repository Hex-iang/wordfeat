#==================================================================================================
# Cmake file for providing handy function with target property setup
# 

#--------------------------------------------------------------------------------------------------
# Function to set runtime directory for target file
function(wordfeat_set_runtime_directory target dir)
  set_target_properties(${target} PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY "${dir}")
endfunction()