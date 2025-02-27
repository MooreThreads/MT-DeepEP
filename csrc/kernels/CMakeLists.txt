function(add_deep_ep_library target_name source_file)
    add_library(${target_name} STATIC ${source_file})
    set_target_properties(${target_name} PROPERTIES
            POSITION_INDEPENDENT_CODE ON
            CXX_STANDARD_REQUIRED ON
            MUSA_STANDARD_REQUIRED ON
            CXX_STANDARD 14
            MUSA_STANDARD 14
            MUSA_SEPARABLE_COMPILATION ON
    )
    target_link_libraries(${target_name} PUBLIC mtshmem musart musadevrt mlx5)
endfunction()

add_deep_ep_library(intranode_musa intranode.cu)
add_deep_ep_library(runtime_musa runtime.cu)
add_deep_ep_library(internode_musa internode.cu)
add_deep_ep_library(internode_ll_musa internode_ll.cu)

# Later, we should link all libraries in `EP_MUSA_LIBRARIES`
set(EP_MUSA_LIBRARIES intranode_musa runtime_musa internode_musa internode_ll_musa PARENT_SCOPE)
