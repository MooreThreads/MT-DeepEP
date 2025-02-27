#include <vector>
#include <cstring>

#include "configs.muh"
#include "exception.muh"
#include "launch.muh"
#include "utils.muh"
#include "ibgda_device.muh"

namespace deep_ep {

namespace intranode {

template<int kNumRanks>
__global__ void barrier(int** task_fifo_ptrs, int head, int rank) {
    // barrier_device<kNumRanks>(task_fifo_ptrs, head, rank);
}

void barrier(int** task_fifo_ptrs, int head, int rank, int num_ranks, musaStream_t stream) {
// #define BARRIER_LAUNCH_CASE(ranks) \
//     LAUNCH_KERNEL(&cfg, barrier<ranks>, task_fifo_ptrs, head, rank); \
//     break

//     SETUP_LAUNCH_CONFIG(1, 32, stream);
//     SWITCH_RANKS(BARRIER_LAUNCH_CASE);
// #undef BARRIER_LAUNCH_CASE
}

} // namespace intranode

namespace internode {


std::vector<uint8_t> get_unique_id() {
    return {};
}

__global__ void ibgda_initialize_recv_queue(int rank) {
}

int init(const std::vector<uint8_t> &root_unique_id_val, int rank, int num_ranks, bool low_latency_mode) {
}

void* alloc(size_t size, size_t alignment) {
}

void free(void* ptr) {
}

void barrier() {
}

void finalize() {
}

} // namespace internode

} // namespace deep_ep
