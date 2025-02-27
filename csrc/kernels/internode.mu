#include "configs.muh"
#include "buffer.muh"
#include "exception.muh"
#include "launch.muh"
// #include "utils.muh"

namespace deep_ep {

namespace internode {


template<int kNumThreads, int kNumExpertsPerSM, int kNumRanksPerSM>
__global__ void __launch_bounds__(kNumThreads, 1)
get_dispatch_layout(const int64_t* topk_idx,
                    int* num_tokens_per_rank, int* num_tokens_per_rdma_rank,
                    int* num_tokens_per_expert, bool* is_token_in_rank,
                    int num_tokens, int num_topk, int num_ranks, int num_experts) {
    auto sm_id = static_cast<int>(blockIdx.x);
    auto thread_id = static_cast<int>(threadIdx.x);

    // Count expert statistics
    __shared__ int num_tokens_per_expert_per_thread[kNumThreads][kNumExpertsPerSM];
    int expert_begin_idx = sm_id * kNumExpertsPerSM, expert_end_idx = min(expert_begin_idx + kNumExpertsPerSM, num_experts);
    if (expert_begin_idx < expert_end_idx) {
        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumExpertsPerSM; ++ i)
            num_tokens_per_expert_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;
            #pragma unroll
            for (int j = 0, expert_idx; j < num_topk; ++ j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin_idx <= expert_idx and expert_idx < expert_end_idx)
                    ++ num_tokens_per_expert_per_thread[thread_id][expert_idx - expert_begin_idx];
            }
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumExpertsPerSM <= kNumThreads, "Too many experts per SM");
        if (expert_begin_idx + thread_id < expert_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++ i)
                sum += num_tokens_per_expert_per_thread[i][thread_id];
            num_tokens_per_expert[expert_begin_idx + thread_id] = sum;
        }
        return;
    }

    if (num_tokens_per_rdma_rank != nullptr)
        EP_DEVICE_ASSERT(num_ranks % NUM_MAX_MTL_PEERS == 0 and num_ranks > NUM_MAX_MTL_PEERS);

    // Count rank statistics
    constexpr int kNumRDMARanksPerSM = kNumRanksPerSM / NUM_MAX_MTL_PEERS;
    __shared__ int num_tokens_per_rank_per_thread[kNumThreads][kNumRanksPerSM];
    __shared__ int num_tokens_per_rdma_rank_per_thread[kNumThreads][kNumRDMARanksPerSM];
    auto sm_begin = (num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM;
    int rank_begin_idx = (sm_id - sm_begin) * kNumRanksPerSM, rank_end_idx = min(rank_begin_idx + kNumRanksPerSM, num_ranks);
    int rdma_rank_begin_idx = rank_begin_idx / NUM_MAX_MTL_PEERS, rdma_rank_end_idx = rank_end_idx / NUM_MAX_MTL_PEERS;
    if (rank_begin_idx < rank_end_idx) {
        const auto num_expert_per_rank = num_experts / num_ranks;
        auto expert_begin = rank_begin_idx * num_expert_per_rank;
        auto expert_end = rank_end_idx * num_expert_per_rank;

        // Per-thread count
        #pragma unroll
        for (int i = 0; i < kNumRanksPerSM; ++ i)
            num_tokens_per_rank_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = 0; i < kNumRDMARanksPerSM; ++ i)
            num_tokens_per_rdma_rank_per_thread[thread_id][i] = 0;
        #pragma unroll
        for (int i = thread_id; i < num_tokens; i += kNumThreads) {
            auto shifted_topk_idx = topk_idx + i * num_topk;
            int is_in_rank[kNumRanksPerSM] = {0}, is_in_rdma_rank[kNumRDMARanksPerSM] = {0};
            #pragma unroll
            for (int j = 0, expert_idx, rank_idx; j < num_topk; ++j) {
                expert_idx = static_cast<int>(shifted_topk_idx[j]);
                if (expert_begin <= expert_idx and expert_idx < expert_end) {
                    // Count single rank
                    rank_idx = expert_idx / num_expert_per_rank - rank_begin_idx;
                    is_in_rank[rank_idx] ++, is_in_rdma_rank[rank_idx / NUM_MAX_MTL_PEERS] ++;
                }
            }

            auto shifted_is_token_in_rank = is_token_in_rank + i * num_ranks;
            #pragma unroll
            for (int j = 0; j + rank_begin_idx < rank_end_idx; ++ j) {
                shifted_is_token_in_rank[j + rank_begin_idx] = (is_in_rank[j] > 0);
                num_tokens_per_rank_per_thread[thread_id][j] += (is_in_rank[j] > 0);
            }

            // #pragma unroll
            /*FIXME*/
            for (int j = 0; j + rdma_rank_begin_idx < rdma_rank_end_idx; ++ j)
                num_tokens_per_rdma_rank_per_thread[thread_id][j] += (is_in_rdma_rank[j] > 0);
        }
        __syncthreads();

        // Sum up
        EP_STATIC_ASSERT(kNumRanksPerSM <= kNumThreads, "Too many ranks per SM");
        if (rank_begin_idx + thread_id < rank_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++ i)
                sum += num_tokens_per_rank_per_thread[i][thread_id];
            num_tokens_per_rank[rank_begin_idx + thread_id] = sum;
        }

        if (num_tokens_per_rdma_rank != nullptr and rdma_rank_begin_idx + thread_id < rdma_rank_end_idx) {
            int sum = 0;
            #pragma unroll
            for (int i = 0; i < kNumThreads; ++ i)
                sum += num_tokens_per_rdma_rank_per_thread[i][thread_id];
            num_tokens_per_rdma_rank[rdma_rank_begin_idx + thread_id] = sum;
        }
    }
}

void get_dispatch_layout(const int64_t* topk_idx,
                         int* num_tokens_per_rank, int* num_tokens_per_rdma_rank,
                         int* num_tokens_per_expert, bool* is_token_in_rank,
                         int num_tokens, int num_topk, int num_ranks, int num_experts,
                         musaStream_t stream) {
    constexpr int kNumThreads = 256, kNumExpertsPerSM = 32, kNumRanksPerSM = 8;
    int num_sms = ((num_experts + kNumExpertsPerSM - 1) / kNumExpertsPerSM) + (num_ranks + kNumRanksPerSM - 1) / kNumRanksPerSM;
    EP_STATIC_ASSERT(kNumExpertsPerSM % NUM_MAX_MTL_PEERS == 0, "Invalid number of experts per SM");
    // /*FIXME me*/
    // get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM><<<num_sms,kNumThreads,0,stream>>>(
    //     topk_idx, num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank,
    //               num_tokens, num_topk, num_ranks, num_experts
    // );
    SETUP_LAUNCH_CONFIG(num_sms, kNumThreads, stream);
    LAUNCH_KERNEL(&cfg, (get_dispatch_layout<kNumThreads, kNumExpertsPerSM, kNumRanksPerSM>),
                  topk_idx, num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank,
                  num_tokens, num_topk, num_ranks, num_experts);
}

struct SourceMeta {
    // int src_rdma_rank, is_token_in_mtl_rank_bits;

    // EP_STATIC_ASSERT(NUM_MAX_MTL_PEERS == 8, "Invalid number of maximum MTL peers");

    // __forceinline__ SourceMeta() = default;

    // // TODO: faster encoding
    // __device__ __forceinline__ SourceMeta(int rdma_rank, const bool* is_token_in_mtl_ranks) {
    //     src_rdma_rank = rdma_rank;
    //     is_token_in_mtl_rank_bits = is_token_in_mtl_ranks[0];
    //     #pragma unroll
    //     for (int i = 1; i < NUM_MAX_MTL_PEERS; ++ i)
    //         is_token_in_mtl_rank_bits |= is_token_in_mtl_ranks[i] << i;
    // }

    // __device__ __forceinline__ bool is_token_in_mtl_rank(int mtl_rank) const {
    //     return (is_token_in_mtl_rank_bits >> mtl_rank) & 1;
    // }
};

// EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");

int get_source_meta_bytes() {
    return sizeof(SourceMeta);
}

__host__ __device__ __forceinline__
int get_num_bytes_per_rdma_token(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights) {
    // return static_cast<int>(align(hidden_int4 * sizeof(int4) + sizeof(SourceMeta) + num_scales * sizeof(float) + num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float), sizeof(int4)));
}

__host__ __device__ __forceinline__
std::pair<int, int> get_rdma_clean_meta(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights, int num_rdma_ranks, int num_rdma_recv_buffer_tokens, int num_sms) {
    // Return `int32_t` offset and count to clean
    // return {
    //     (get_num_bytes_per_rdma_token(hidden_int4, num_scales, num_topk_idx, num_topk_weights) * num_rdma_recv_buffer_tokens * num_rdma_ranks * 2 * num_sms) / sizeof(int),
    //     (NUM_MAX_MTL_PEERS * 2 + 4) * num_rdma_ranks * 2 * num_sms
    // };
}

__host__ __device__ __forceinline__
std::pair<int, int> get_mtl_clean_meta(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights, int num_rdma_ranks, int num_mtl_ranks, int num_mtl_recv_buffer_tokens, int num_sms) {
    // Return `int32_t` offset and to clean
    // EP_STATIC_ASSERT(sizeof(SourceMeta) % sizeof(int) == 0, "Invalid size of `SourceMeta`");
    // return {
    //     (num_mtl_recv_buffer_tokens * (hidden_int4 * sizeof(int4) + num_scales * sizeof(float) + num_topk_idx * sizeof(int) + num_topk_weights * sizeof(float) + sizeof(SourceMeta)) * num_mtl_ranks * num_sms) / sizeof(int),
    //     num_mtl_ranks * (2 * num_rdma_ranks + 2) * num_sms,
    // };
}

template <bool kLowLatencyMode>
__forceinline__ __device__ int translate_dst_rdma_rank(const int dst_rdma_rank, const int mtl_rank) {
    // return kLowLatencyMode ? (dst_rdma_rank * NUM_MAX_MTL_PEERS + mtl_rank) : dst_rdma_rank;
}


template <bool kLowLatencyMode, int kNumRDMARanks>
__global__ void
notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                const int* num_tokens_per_rdma_rank, int* moe_recv_rdma_counter_mapped,
                const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                const bool* is_token_in_rank, int num_tokens, int num_channels, int expert_alignment,
                const int rdma_clean_offset, const int rdma_num_int_clean,
                const int mtl_clean_offset, const int mtl_num_int_clean,
                int* rdma_channel_prefix_matrix, int* recv_rdma_rank_prefix_sum,
                int* gbl_channel_prefix_matrix, int* recv_gbl_rank_prefix_sum,
                void* rdma_buffer_ptr,
                void** buffer_ptrs, int** task_fifo_ptrs, int head, int rank,
                const int rdma_team) {
   
}

void notify_dispatch(const int* num_tokens_per_rank, int* moe_recv_counter_mapped, int num_ranks,
                     const int* num_tokens_per_rdma_rank, int* moe_recv_rdma_counter_mapped,
                     const int* num_tokens_per_expert, int* moe_recv_expert_counter_mapped, int num_experts,
                     const bool* is_token_in_rank, int num_tokens, int num_channels,
                     int hidden_int4, int num_scales, int num_topk, int expert_alignment,
                     int* rdma_channel_prefix_matrix, int* recv_rdma_rank_prefix_sum,
                     int* gbl_channel_prefix_matrix, int* recv_gbl_rank_prefix_sum,
                     void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                     void** buffer_ptrs, int num_max_mtl_chunked_recv_tokens,
                     int** task_fifo_ptrs, int head, int rank,
                     musaStream_t stream, int64_t num_rdma_bytes, int64_t num_mtl_bytes,
                     bool low_latency_mode) {
// #define NOTIFY_DISPATCH_LAUNCH_CASE(num_rdma_ranks) { \
//     auto notify_dispatch_func = low_latency_mode ? \
//         notify_dispatch<true, num_rdma_ranks> : notify_dispatch<false, num_rdma_ranks>; \
//     LAUNCH_KERNEL(&cfg, notify_dispatch_func, \
//                   num_tokens_per_rank, moe_recv_counter_mapped, num_ranks, \
//                   num_tokens_per_rdma_rank, moe_recv_rdma_counter_mapped, \
//                   num_tokens_per_expert, moe_recv_expert_counter_mapped, num_experts, \
//                   is_token_in_rank, num_tokens, num_channels, expert_alignment, \
//                   rdma_clean_meta.first, rdma_clean_meta.second, \
//                   mtl_clean_meta.first, mtl_clean_meta.second, \
//                   rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
//                   gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
//                   rdma_buffer_ptr, \
//                   buffer_ptrs, task_fifo_ptrs, head, rank, \
//                   cpu_rdma_team); } break

//     constexpr int kNumThreads = 256;
//     const auto num_rdma_ranks = num_ranks / NUM_MAX_MTL_PEERS;

//     // Get clean meta
//     auto rdma_clean_meta = get_rdma_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
//     auto mtl_clean_meta = get_mtl_clean_meta(hidden_int4, num_scales, num_topk, num_topk, num_rdma_ranks, NUM_MAX_MTL_PEERS, num_max_mtl_chunked_recv_tokens, num_channels);
//     EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
//     EP_HOST_ASSERT((mtl_clean_meta.first + mtl_clean_meta.second) * sizeof(int) <= num_mtl_bytes);
//     EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
//     EP_HOST_ASSERT(num_mtl_bytes < std::numeric_limits<int>::max());

//     // Launch kernel
//     SETUP_LAUNCH_CONFIG(1 + num_rdma_ranks, kNumThreads, stream);
//     SWITCH_RDMA_RANKS(NOTIFY_DISPATCH_LAUNCH_CASE);
// #undef NOTIFY_DISPATCH_LAUNCH_CASE
}

// At most 8 RDMA ranks to be sent
constexpr int get_num_topk_rdma_ranks(int num_rdma_ranks) {
    return num_rdma_ranks < 8 ? num_rdma_ranks : 8;
}

template <bool kLowLatencyMode, int kNumRDMARanks, bool kCachedMode,
          int kNumDispatchRDMASenderWarps, int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks)>
__global__ void __launch_bounds__(((kNumDispatchRDMASenderWarps + 1 + NUM_MAX_MTL_PEERS) * 32), 1)
dispatch(int4* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights, SourceMeta* recv_src_meta,
         const int4* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
         int* send_rdma_head, int* send_mtl_head,
         int* recv_rdma_channel_prefix_matrix, int* recv_gbl_channel_prefix_matrix,
         const int* rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
         const int* gbl_channel_prefix_matrix, const int* recv_gbl_rank_prefix_sum,
         int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,
         const bool* is_token_in_rank,
         void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
         void** buffer_ptrs, int num_max_mtl_chunked_send_tokens, int num_max_mtl_chunked_recv_tokens,
         int rank, int num_ranks) {
 
}

void dispatch(void* recv_x, float* recv_x_scales, int64_t* recv_topk_idx, float* recv_topk_weights, void* recv_src_meta,
              const void* x, const float* x_scales, const int64_t* topk_idx, const float* topk_weights,
              int* send_rdma_head, int* send_mtl_head,
              int* recv_rdma_channel_prefix_matrix, int* recv_gbl_channel_prefix_matrix,
              const int* rdma_channel_prefix_matrix, const int* recv_rdma_rank_prefix_sum,
              const int* gbl_channel_prefix_matrix, const int* recv_gbl_rank_prefix_sum,
              int num_tokens, int hidden_int4, int num_scales, int num_topk, int num_experts,
              const bool* is_token_in_rank,
              void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
              void** buffer_ptrs, int num_max_mtl_chunked_send_tokens, int num_max_mtl_chunked_recv_tokens,
              int rank, int num_ranks, bool is_cached_dispatch,
              musaStream_t stream, int num_channels, bool low_latency_mode) {
//     constexpr int kNumDispatchRDMASenderWarps = 7;

// #define DISPATCH_LAUNCH_CASE(num_rdma_ranks) { \
//     auto dispatch_func = low_latency_mode ? \
//         (is_cached_dispatch ? dispatch<true, num_rdma_ranks, true, kNumDispatchRDMASenderWarps> : dispatch<true, num_rdma_ranks, false, kNumDispatchRDMASenderWarps>) : \
//         (is_cached_dispatch ? dispatch<false, num_rdma_ranks, true, kNumDispatchRDMASenderWarps> : dispatch<false, num_rdma_ranks, false, kNumDispatchRDMASenderWarps>); \
//     LAUNCH_KERNEL(&cfg, dispatch_func, \
//                   reinterpret_cast<int4*>(recv_x), recv_x_scales, recv_topk_idx, recv_topk_weights, reinterpret_cast<SourceMeta*>(recv_src_meta), \
//                   reinterpret_cast<const int4*>(x), x_scales, topk_idx, topk_weights, \
//                   send_rdma_head, send_mtl_head, \
//                   recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix, \
//                   rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, \
//                   gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum, \
//                   num_tokens, hidden_int4, num_scales, num_topk, num_experts, \
//                   is_token_in_rank, \
//                   rdma_buffer_ptr, num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, \
//                   buffer_ptrs, num_max_mtl_chunked_send_tokens, num_max_mtl_chunked_recv_tokens, \
//                   rank, num_ranks); } break

//     EP_HOST_ASSERT((topk_idx == nullptr)  == (topk_weights == nullptr));
//     EP_HOST_ASSERT((recv_topk_idx == nullptr) == (recv_topk_weights == nullptr));

//     SETUP_LAUNCH_CONFIG(num_channels * 2, (kNumDispatchRDMASenderWarps + 1 + NUM_MAX_MTL_PEERS) * 32, stream);
//     SWITCH_RDMA_RANKS(DISPATCH_LAUNCH_CASE);
// #undef DISPATCH_LAUNCH_CASE
}

template <bool kLowLatencyMode>
__global__ void cached_notify(const int rdma_clean_offset, const int rdma_num_int_clean,
                              const int mtl_clean_offset, const int mtl_num_int_clean,
                              int* combined_rdma_head, int num_combined_tokens, int num_channels,
                              const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, int* combined_mtl_head,
                              void* rdma_buffer_ptr,
                              void** buffer_ptrs, int** task_fifo_ptrs, int head, int rank, int num_ranks,
                              bool is_cached_dispatch, const int rdma_team) {
   
}

void cached_notify(int hidden_int4, int num_scales, int num_topk_idx, int num_topk_weights,
                   int num_ranks, int num_channels, int num_combined_tokens, int* combined_rdma_head,
                   const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, int* combined_mtl_head,
                   void* rdma_buffer_ptr, int num_max_rdma_chunked_recv_tokens,
                   void** buffer_ptrs, int num_max_mtl_chunked_recv_tokens,
                   int** task_fifo_ptrs, int head, int rank, musaStream_t stream,
                   int64_t num_rdma_bytes, int64_t num_mtl_bytes,
                   bool is_cached_dispatch, bool low_latency_mode) {
    // const int num_threads = std::max(128, 32 * num_channels);
    // const auto num_rdma_ranks = num_ranks / NUM_MAX_MTL_PEERS;

    // // Get clean meta
    // auto rdma_clean_meta = get_rdma_clean_meta(hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks, num_max_rdma_chunked_recv_tokens, num_channels);
    // auto mtl_clean_meta = get_mtl_clean_meta(hidden_int4, num_scales, num_topk_idx, num_topk_weights, num_rdma_ranks, NUM_MAX_MTL_PEERS, num_max_mtl_chunked_recv_tokens, num_channels);
    // EP_HOST_ASSERT((rdma_clean_meta.first + rdma_clean_meta.second) * sizeof(int) <= num_rdma_bytes);
    // EP_HOST_ASSERT((mtl_clean_meta.first + mtl_clean_meta.second) * sizeof(int) <= num_mtl_bytes);
    // EP_HOST_ASSERT(num_rdma_bytes < std::numeric_limits<int>::max());
    // EP_HOST_ASSERT(num_mtl_bytes < std::numeric_limits<int>::max());
    // EP_HOST_ASSERT(num_channels * 2 > 3);

    // // Launch kernel
    // auto cached_notify_func = low_latency_mode ? cached_notify<true> : cached_notify<false>;
    // SETUP_LAUNCH_CONFIG(num_channels * 2, num_threads, stream);
    // LAUNCH_KERNEL(&cfg, cached_notify_func,
    //               rdma_clean_meta.first, rdma_clean_meta.second,
    //               mtl_clean_meta.first, mtl_clean_meta.second,
    //               combined_rdma_head, num_combined_tokens, num_channels,
    //               rdma_channel_prefix_matrix, rdma_rank_prefix_sum, combined_mtl_head,
    //               rdma_buffer_ptr,
    //               buffer_ptrs, task_fifo_ptrs, head, rank, num_ranks,
    //               is_cached_dispatch, cpu_rdma_team);
}

template <int kNumRanks, typename dtype_t, int kMaxNumRanks, typename ReceiveFn, typename ReceiveTWFn>
__device__ int combine_token(bool is_token_in_rank, int head_idx,
                             int lane_id, int hidden_int4, int num_topk,
                             int4* combined_row, float* combined_topk_weights,
                             int num_max_recv_tokens, const ReceiveFn& recv_fn, const ReceiveTWFn& recv_tw_fn) {
    // constexpr auto kDtypePerInt4 = sizeof(int4) / sizeof(dtype_t);

    // // Broadcast current heads
    // // Lane `i` holds the head of rank `i` and `is_token_in_rank`
    // EP_STATIC_ASSERT(kMaxNumRanks <= 32, "Too many ranks");
    // int num_topk_ranks = 0, topk_ranks[kMaxNumRanks], slot_indices[kMaxNumRanks];
    // #pragma unroll
    // for (int i = 0; i < kNumRanks; ++ i) if (__shfl_sync(0xffffffff, is_token_in_rank, i)) {
    //     slot_indices[num_topk_ranks] = __shfl_sync(0xffffffff, head_idx, i) % num_max_recv_tokens;
    //     topk_ranks[num_topk_ranks ++] = i;
    // }
    // EP_DEVICE_ASSERT(num_topk_ranks <= kMaxNumRanks);

    // // Reduce data
    // #pragma unroll
    // for (int i = lane_id; i < hidden_int4; i += 32) {
    //     // Read buffers
    //     // TODO: maybe too many registers here
    //     int4 recv_value_int4[kMaxNumRanks];
    //     #pragma unroll
    //     for (int j = 0; j < num_topk_ranks; ++ j)
    //         recv_value_int4[j] = recv_fn(topk_ranks[j], slot_indices[j], i);

    //     // Reduce all-to-all results
    //     float values[kDtypePerInt4] = {0};
    //     #pragma unroll
    //     for (int j = 0; j < num_topk_ranks; ++ j) {
    //         auto recv_value_dtypes = reinterpret_cast<const dtype_t*>(&recv_value_int4[j]);
    //         #pragma unroll
    //         for (int k = 0; k < kDtypePerInt4; ++ k)
    //             values[k] += static_cast<float>(recv_value_dtypes[k]);
    //     }

    //     // Cast back to `dtype_t` and write
    //     int4 out_int4;
    //     auto out_dtypes = reinterpret_cast<dtype_t*>(&out_int4);
    //     #pragma unroll
    //     for (int j = 0; j < kDtypePerInt4; ++ j)
    //         out_dtypes[j] = static_cast<dtype_t>(values[j]);
    //     st_na_global(combined_row + i, out_int4);
    // }

    // // Reduce `topk_weights`
    // if (lane_id < num_topk) {
    //     float value = 0;
    //     #pragma unroll
    //     for (int i = 0; i < num_topk_ranks; ++ i)
    //         value += recv_tw_fn(topk_ranks[i], slot_indices[i], lane_id);
    //     st_na_global(combined_topk_weights + lane_id, value);
    // }

    // // Return the minimum top-k rank
    // return topk_ranks[0];
}

template<bool kLowLatencyMode,
         int kNumRDMARanks, typename dtype_t,
         int kNumCombineForwarderWarps,
         int kNumTopkRDMARanks = get_num_topk_rdma_ranks(kNumRDMARanks),
         int kNumWarpsPerForwarder = (kNumCombineForwarderWarps / kNumRDMARanks > 0) ? kNumCombineForwarderWarps / kNumRDMARanks : 1,
         int kNumForwarders = kNumRDMARanks * kNumWarpsPerForwarder,
         int kNumRDMAReceivers = kNumForwarders + NUM_MAX_MTL_PEERS>
__global__ void __launch_bounds__((NUM_MAX_MTL_PEERS + 1 + kNumForwarders) * 32, 1)
combine(int4* combined_x, float* combined_topk_weights,
        const bool* is_combined_token_in_rank,
        const int4* x, const float* topk_weights,
        const int* combined_rdma_head, const int* combined_mtl_head,
        const SourceMeta* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, const int* gbl_channel_prefix_matrix,
        int num_tokens, int num_combined_tokens, int hidden, int num_topk,
        void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
        void** buffer_ptrs, int num_max_mtl_chunked_send_tokens, int num_max_mtl_chunked_recv_tokens,
        int rank, int num_ranks) {
  
}

void combine(musaDataType_t type,
             void* combined_x, float* combined_topk_weights,
             const bool* is_combined_token_in_rank,
             const void* x, const float* topk_weights,
             const int* combined_rdma_head, const int* combined_mtl_head,
             const void* src_meta, const int* rdma_channel_prefix_matrix, const int* rdma_rank_prefix_sum, const int* gbl_channel_prefix_matrix,
             int num_tokens, int num_combined_tokens, int hidden, int num_topk,
             void* rdma_buffer_ptr, int num_max_rdma_chunked_send_tokens, int num_max_rdma_chunked_recv_tokens,
             void** buffer_ptrs, int num_max_mtl_chunked_send_tokens, int num_max_mtl_chunked_recv_tokens,
             int rank, int num_ranks, musaStream_t stream, int num_channels, bool low_latency_mode) {
//     constexpr int kNumCombineForwarderWarps = 16;

// #define COMBINE_LAUNCH_CASE(num_rdma_ranks) { \
//     auto combine_func = low_latency_mode ? \
//         combine<true, num_rdma_ranks, mt_bfloat16, kNumCombineForwarderWarps> : combine<false, num_rdma_ranks, mt_bfloat16, kNumCombineForwarderWarps>; \
//     LAUNCH_KERNEL(&cfg, combine_func, \
//                   reinterpret_cast<int4*>(combined_x), combined_topk_weights, is_combined_token_in_rank, \
//                   reinterpret_cast<const int4*>(x), topk_weights, \
//                   combined_rdma_head, combined_mtl_head, \
//                   reinterpret_cast<const SourceMeta*>(src_meta), rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix, \
//                   num_tokens, num_combined_tokens, hidden, num_topk, \
//                   rdma_buffer_ptr, num_max_rdma_chunked_send_tokens, num_max_rdma_chunked_recv_tokens, \
//                   buffer_ptrs, num_max_mtl_chunked_send_tokens, num_max_mtl_chunked_recv_tokens, \
//                   rank, num_ranks); } break

//     int num_rdma_ranks = num_ranks / NUM_MAX_MTL_PEERS;
//     auto num_warps_per_forwarder = std::max(kNumCombineForwarderWarps / num_rdma_ranks, 1);
//     int num_forwarder_warps = num_rdma_ranks * num_warps_per_forwarder;
//     EP_HOST_ASSERT(num_forwarder_warps > 0 and num_forwarder_warps % num_rdma_ranks == 0);
//     EP_HOST_ASSERT(num_max_mtl_chunked_recv_tokens % num_rdma_ranks == 0);
//     EP_HOST_ASSERT(num_max_mtl_chunked_recv_tokens / num_rdma_ranks > std::max(num_max_rdma_chunked_send_tokens, num_max_mtl_chunked_send_tokens));
//     EP_HOST_ASSERT(type == MUSA_R_16BF);

//     SETUP_LAUNCH_CONFIG(num_channels * 2, (NUM_MAX_MTL_PEERS + num_forwarder_warps + 1) * 32, stream);
//     SWITCH_RDMA_RANKS(COMBINE_LAUNCH_CASE);
// #undef COMBINE_LAUNCH_CASE
}

} // namespace internode

} // namespace deep_ep
