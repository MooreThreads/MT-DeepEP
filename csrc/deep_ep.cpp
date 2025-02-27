// #include <ATen/ATen.h>
#include "torch_musa/csrc/aten/musa/MUSAContext.h"
// #include <ATen/musa/MUSADataType.h>
#include <atomic>
#include <chrono>
#include <musa_runtime.h>
#include <memory>
#include <pybind11/functional.h>
#include <torch/python.h>

#include "deep_ep.hpp"
#include "kernels/api.muh"
#include "kernels/configs.muh"

namespace deep_ep {

Buffer::Buffer(int rank, int num_ranks, int64_t num_mtl_bytes, int64_t num_rdma_bytes, bool low_latency_mode):
        rank(rank), num_ranks(num_ranks),
        num_mtl_bytes(num_mtl_bytes), num_rdma_bytes(num_rdma_bytes),
        low_latency_mode(low_latency_mode),
        comm_stream(at::musa::getStreamFromPool(true)) {
    // Task fifo memory
    int64_t fifo_bytes = sizeof(int) * NUM_MAX_FIFO_SLOTS;
    int64_t buffer_ptr_bytes = sizeof(void*) * NUM_MAX_MTL_PEERS;
    int64_t task_ptr_bytes = sizeof(int*) * NUM_MAX_MTL_PEERS;

    // Common checks
    EP_HOST_ASSERT(num_mtl_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (num_mtl_bytes <= std::numeric_limits<int>::max() or num_rdma_bytes == 0));
    EP_HOST_ASSERT(num_rdma_bytes % NUM_BUFFER_ALIGNMENT_BYTES == 0 and (low_latency_mode or num_rdma_bytes <= std::numeric_limits<int>::max()));
    EP_HOST_ASSERT(0 <= rank and rank < num_ranks and (num_ranks <= NUM_MAX_MTL_PEERS * NUM_MAX_RDMA_PEERS or low_latency_mode));
    EP_HOST_ASSERT(num_ranks < NUM_MAX_MTL_PEERS or num_ranks % NUM_MAX_MTL_PEERS == 0);
    if (num_rdma_bytes > 0)
        EP_HOST_ASSERT(num_ranks > NUM_MAX_MTL_PEERS or low_latency_mode);

    // Get ranks
    MUSA_CHECK(musaGetDevice(&device_id));
    rdma_rank = rank / NUM_MAX_MTL_PEERS, mtl_rank = rank % NUM_MAX_MTL_PEERS;
    num_rdma_ranks = std::max(1, num_ranks / NUM_MAX_MTL_PEERS), num_mtl_ranks = std::min(num_ranks, NUM_MAX_MTL_PEERS);

    // Get device info
    musaDeviceProp device_prop = {};
    MUSA_CHECK(musaGetDeviceProperties(&device_prop, device_id));

    if (num_mtl_bytes > 0) {
        // Local IPC: alloc local memory and set local IPC handle
        MUSA_CHECK(musaMalloc(&buffer_ptrs[mtl_rank], num_mtl_bytes + fifo_bytes + buffer_ptr_bytes + task_ptr_bytes));
        MUSA_CHECK(musaIpcGetMemHandle(&ipc_handles[mtl_rank], buffer_ptrs[mtl_rank]));
        buffer_ptrs_gpu = reinterpret_cast<void**>(reinterpret_cast<uint8_t*>(buffer_ptrs[mtl_rank]) + num_mtl_bytes + fifo_bytes);

        // Set task fifo
        EP_HOST_ASSERT(NUM_MAX_FIFO_SLOTS % num_mtl_ranks == 0);
        task_fifo_ptrs[mtl_rank] = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buffer_ptrs[mtl_rank]) + num_mtl_bytes);
        task_fifo_ptrs_gpu = reinterpret_cast<int**>(reinterpret_cast<uint8_t*>(buffer_ptrs[mtl_rank]) + num_mtl_bytes + fifo_bytes + buffer_ptr_bytes);

        // No need to synchronize, will do a full device sync during `sync`
        MUSA_CHECK(musaMemsetAsync(task_fifo_ptrs[mtl_rank], 0, fifo_bytes, comm_stream));
    }

    // Create 32 MiB workspace
    MUSA_CHECK(musaMalloc(&workspace, NUM_WORKSPACE_BYTES));
    MUSA_CHECK(musaMemsetAsync(workspace, 0, NUM_WORKSPACE_BYTES, comm_stream));

    // MoE counter
    MUSA_CHECK(musaMallocHost(&moe_recv_counter, sizeof(int64_t), musaHostAllocMapped));
    MUSA_CHECK(musaHostGetDevicePointer(&moe_recv_counter_mapped, const_cast<int*>(moe_recv_counter), 0));
    *moe_recv_counter = -1;

    // MoE expert-level counter
    MUSA_CHECK(musaMallocHost(&moe_recv_expert_counter, sizeof(int) * NUM_MAX_LOCAL_EXPERTS, musaHostAllocMapped));
    MUSA_CHECK(musaHostGetDevicePointer(&moe_recv_expert_counter_mapped, const_cast<int*>(moe_recv_expert_counter), 0));
    for (int i = 0; i < NUM_MAX_LOCAL_EXPERTS; ++ i)
        moe_recv_expert_counter[i] = -1;

    // MoE RDMA-level counter
    if (num_rdma_ranks > 0) {
        MUSA_CHECK(musaMallocHost(&moe_recv_rdma_counter, sizeof(int), musaHostAllocMapped));
        MUSA_CHECK(musaHostGetDevicePointer(&moe_recv_rdma_counter_mapped, const_cast<int*>(moe_recv_rdma_counter), 0));
        *moe_recv_rdma_counter = -1;
    }
}

Buffer::~Buffer() noexcept(false) {
    // Synchronize
    MUSA_CHECK(musaDeviceSynchronize());

    if (num_mtl_bytes > 0) {
        // Barrier
        intranode::barrier(task_fifo_ptrs_gpu, head, mtl_rank, num_mtl_ranks, comm_stream);
        move_fifo_slots();
        MUSA_CHECK(musaDeviceSynchronize());

        // Close remote IPC
        if (is_available()) {
            for (int i = 0; i < num_mtl_ranks; ++ i) if (i != mtl_rank)
                MUSA_CHECK(musaIpcCloseMemHandle(buffer_ptrs[i]));
        }

        // Free local buffer and error flag
        MUSA_CHECK(musaFree(buffer_ptrs[mtl_rank]));
    }

    // Free MTSHMEM
    if (num_rdma_bytes > 0) {
        MUSA_CHECK(musaDeviceSynchronize());
        internode::barrier();
        internode::free(rdma_buffer_ptr);
        internode::finalize();
    }

    // Free cuBLAS handle, workspace and MoE counter
    MUSA_CHECK(musaFree(workspace));
    MUSA_CHECK(musaFreeHost(const_cast<int*>(moe_recv_counter)));

    // Free chunked mode staffs
    MUSA_CHECK(musaFreeHost(const_cast<int*>(moe_recv_expert_counter)));
}

void Buffer::move_fifo_slots(int num_slots) {
    head = (head + num_ranks * num_slots) % NUM_MAX_FIFO_SLOTS;
}

bool Buffer::is_available() const {
    return available;
}

bool Buffer::is_internode_available() const {
    return is_available() and num_ranks > NUM_MAX_MTL_PEERS;
}

int Buffer::get_num_rdma_ranks() const {
    return num_rdma_ranks;
}

int Buffer::get_rdma_rank() const {
    return rdma_rank;
}

int Buffer::get_root_rdma_rank(bool global) const {
    return global ? mtl_rank : 0;
}

int Buffer::get_local_device_id() const {
    return device_id;
}

pybind11::bytearray Buffer::get_local_ipc_handle() const {
    return {ipc_handles[mtl_rank].reserved, MUSA_IPC_HANDLE_SIZE};
}

pybind11::bytearray Buffer::get_local_mtshmem_unique_id() const {
    EP_HOST_ASSERT(rdma_rank == 0 and "Only RDMA rank 0 can get MTSHMEM unique ID");
    auto unique_id = internode::get_unique_id();
    return {reinterpret_cast<const char*>(unique_id.data()), unique_id.size()};
}

torch::Tensor Buffer::get_local_buffer_tensor(const pybind11::object& dtype, int64_t offset, bool use_rdma_buffer) const {
    torch::ScalarType casted_dtype = torch::python::detail::py_object_to_dtype(dtype);
    auto element_bytes = static_cast<int64_t>(elementSize(casted_dtype));
    auto base_ptr = reinterpret_cast<uint8_t*>(use_rdma_buffer ? rdma_buffer_ptr : buffer_ptrs[mtl_rank]) + offset;
    auto num_bytes = use_rdma_buffer ? num_rdma_bytes : num_mtl_bytes;
    return torch::from_blob(base_ptr, num_bytes / element_bytes, torch::TensorOptions().dtype(casted_dtype).device(at::kMUSA));
}

void Buffer::sync(const std::vector<int> &device_ids,
                  const std::vector<std::optional<pybind11::bytearray>> &all_gathered_handles,
                  const std::optional<pybind11::bytearray>& root_unique_id_opt) {
    EP_HOST_ASSERT(not is_available());

    // Sync IPC handles
    if (num_mtl_bytes > 0) {
        EP_HOST_ASSERT(num_ranks == device_ids.size());
        EP_HOST_ASSERT(device_ids.size() == all_gathered_handles.size());
        for (int i = 0, offset = rdma_rank * num_mtl_ranks; i < num_mtl_ranks; ++ i) {
            EP_HOST_ASSERT(all_gathered_handles[offset + i].has_value());
            auto handle_str = std::string(all_gathered_handles[offset + i].value());
            EP_HOST_ASSERT(handle_str.size() == MUSA_IPC_HANDLE_SIZE);
            if (offset + i != rank) {
                std::memcpy(ipc_handles[i].reserved, handle_str.c_str(), MUSA_IPC_HANDLE_SIZE);
                MUSA_CHECK(musaIpcOpenMemHandle(&buffer_ptrs[i], ipc_handles[i], musaIpcMemLazyEnablePeerAccess));
                task_fifo_ptrs[i] = reinterpret_cast<int*>(reinterpret_cast<uint8_t*>(buffer_ptrs[i]) + num_mtl_bytes);
            } else {
                EP_HOST_ASSERT(std::memcmp(ipc_handles[i].reserved, handle_str.c_str(), MUSA_IPC_HANDLE_SIZE) == 0);
            }
        }

        // Copy all buffer and task pointers to GPU
        MUSA_CHECK(musaMemcpy(buffer_ptrs_gpu, buffer_ptrs, sizeof(void*) * NUM_MAX_MTL_PEERS, musaMemcpyHostToDevice));
        MUSA_CHECK(musaMemcpy(task_fifo_ptrs_gpu, task_fifo_ptrs, sizeof(int*) * NUM_MAX_MTL_PEERS, musaMemcpyHostToDevice));
        MUSA_CHECK(musaDeviceSynchronize());
    }

    // Sync MTSHMEM handles and allocate memory
    if (num_rdma_bytes > 0) {
        // Initialize MTSHMEM
        EP_HOST_ASSERT(root_unique_id_opt.has_value());
        std::vector<uint8_t> root_unique_id(root_unique_id_opt->size());
        auto root_unique_id_str = root_unique_id_opt->cast<std::string>();
        std::memcpy(root_unique_id.data(), root_unique_id_str.c_str(), root_unique_id_opt->size());
        auto mtshmem_rank = low_latency_mode ? rank : rdma_rank;
        auto num_mtshmem_ranks = low_latency_mode ? num_ranks : num_rdma_ranks;
        EP_HOST_ASSERT(mtshmem_rank == internode::init(root_unique_id, mtshmem_rank, num_mtshmem_ranks, low_latency_mode));
        internode::barrier();

        // Allocate
        rdma_buffer_ptr = internode::alloc(num_rdma_bytes, NUM_BUFFER_ALIGNMENT_BYTES);

        // Clean buffer (mainly for low-latency mode)
        MUSA_CHECK(musaMemset(rdma_buffer_ptr, 0, num_rdma_bytes));

        // Barrier
        internode::barrier();
        MUSA_CHECK(musaDeviceSynchronize());
    }

    // Ready to use
    available = true;
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::get_dispatch_layout(const torch::Tensor& topk_idx, int num_experts,
                            std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(topk_idx.dim() == 2);
    EP_HOST_ASSERT(topk_idx.is_contiguous());
    EP_HOST_ASSERT(num_experts > 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::musa::getCurrentMUSAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::musa::setCurrentMUSAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    auto num_tokens = static_cast<int>(topk_idx.size(0)), num_topk = static_cast<int>(topk_idx.size(1));
    auto num_tokens_per_rank = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kMUSA));
    auto num_tokens_per_rdma_rank = std::optional<torch::Tensor>();
    auto num_tokens_per_expert = torch::empty({num_experts}, dtype(torch::kInt32).device(torch::kMUSA));
    auto is_token_in_rank = torch::empty({num_tokens, num_ranks}, dtype(torch::kBool).device(torch::kMUSA));
    if (is_internode_available())
        num_tokens_per_rdma_rank = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kMUSA));

    internode::get_dispatch_layout(topk_idx.data_ptr<int64_t>(),
                                   num_tokens_per_rank.data_ptr<int>(),
                                   num_tokens_per_rdma_rank.has_value() ? num_tokens_per_rdma_rank.value().data_ptr<int>() : nullptr,
                                   num_tokens_per_expert.data_ptr<int>(),
                                   is_token_in_rank.data_ptr<bool>(),
                                   num_tokens, num_topk, num_ranks, num_experts,
                                   comm_stream);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {topk_idx, num_tokens_per_rank, num_tokens_per_expert, is_token_in_rank}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {num_tokens_per_rdma_rank}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::musa::setCurrentMUSAStream(compute_stream);

    return {num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert, is_token_in_rank, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>>
Buffer::intranode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank, const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens, const std::optional<torch::Tensor>& cached_rank_prefix_matrix, const std::optional<torch::Tensor>& cached_channel_prefix_matrix,
                           int expert_alignment, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    bool cached_mode = cached_rank_prefix_matrix.has_value();

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_channel_prefix_matrix.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    EP_HOST_ASSERT(is_token_in_rank.scalar_type() == torch::kBool);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(is_token_in_rank.dim() == 2 and is_token_in_rank.is_contiguous());
    EP_HOST_ASSERT(is_token_in_rank.size(0) == x.size(0) and is_token_in_rank.size(1) == num_ranks);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rank_prefix_matrix->dim() == 2 and cached_rank_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rank_prefix_matrix->size(0) == num_ranks and cached_rank_prefix_matrix->size(1) == num_ranks);
        EP_HOST_ASSERT(cached_channel_prefix_matrix->dim() == 2 and cached_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_channel_prefix_matrix->size(0) == num_ranks and cached_channel_prefix_matrix->size(1) == num_channels);
    } else {
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    int64_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<int64_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32);
        EP_HOST_ASSERT(x_scales->dim() > 0 and x_scales->dim() < 3 and x_scales->is_contiguous());
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = x_scales->data_ptr<float>();
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::musa::getCurrentMUSAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::musa::setCurrentMUSAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1;
    auto rank_prefix_matrix = torch::Tensor();
    auto channel_prefix_matrix = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    // To clean: channel start/end offset, head and tail
    int num_memset_int = num_channels * num_ranks * 4;
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        rank_prefix_matrix = cached_rank_prefix_matrix.value();
        channel_prefix_matrix = cached_channel_prefix_matrix.value();

        // Copy rank prefix matrix and clean flags
        intranode::cached_notify_dispatch(rank_prefix_matrix.data_ptr<int>(), num_memset_int,
                                          buffer_ptrs_gpu, task_fifo_ptrs_gpu, head, rank, num_ranks,
                                          comm_stream);
        move_fifo_slots(2);
    } else {
        rank_prefix_matrix = torch::empty({num_ranks, num_ranks}, dtype(torch::kInt32).device(torch::kMUSA));
        channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kMUSA));

        // Send sizes
        // Meta information:
        //  - Size prefix by ranks, shaped as `[num_ranks, num_ranks]`
        //  - Size prefix by experts (not used later), shaped as `[num_ranks, num_local_experts]`
        // NOTES: no more token dropping in this version
        *moe_recv_counter = -1;
        for (int i = 0; i < num_local_experts; ++ i)
            moe_recv_expert_counter[i] = -1;
        EP_HOST_ASSERT(num_ranks * (num_ranks + num_local_experts) * sizeof(int) <= num_mtl_bytes);
        intranode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
                                   num_tokens_per_expert->data_ptr<int>(), moe_recv_expert_counter_mapped, num_experts,
                                   num_tokens, is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
                                   rank_prefix_matrix.data_ptr<int>(),
                                   num_memset_int, expert_alignment,
                                   buffer_ptrs_gpu, task_fifo_ptrs_gpu, head, rank,
                                   comm_stream, num_channels);
        move_fifo_slots(3);

        // Synchronize total received tokens and tokens per expert
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            // Read total count
            num_recv_tokens = static_cast<int>(*moe_recv_counter);

            // Read per-expert count
            bool ready = (num_recv_tokens >= 0);
            for (int i = 0; i < num_local_experts and ready; ++i)
                ready &= moe_recv_expert_counter[i] >= 0;

            if (ready)
                break;

            // Timeout check
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() > NUM_CPU_TIMEOUT_SECS)
                throw std::runtime_error("DeepEP error: CPU recv timeout");
        }
        num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_src_idx = torch::empty({num_recv_tokens}, dtype(torch::kInt32).device(torch::kMUSA));
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(), recv_x_scales = std::optional<torch::Tensor>();
    auto recv_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kMUSA));
    auto send_head = torch::empty({num_tokens, num_ranks}, dtype(torch::kInt32).device(torch::kMUSA));

    // Assign pointers
    int64_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<int64_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ?
                        torch::empty({num_recv_tokens}, x_scales->options()) :
                        torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = recv_x_scales->data_ptr<float>();
    }

    // Dispatch
    EP_HOST_ASSERT(num_ranks * num_ranks * sizeof(int) +                                                                    // Size prefix matrix
                   num_channels * num_ranks * sizeof(int) +                                                                 // Channel start offset
                   num_channels * num_ranks * sizeof(int) +                                                                 // Channel end offset
                   num_channels * num_ranks * sizeof(int) * 2 +                                                             // Queue head and tail
                   num_channels * num_ranks * config.num_max_mtl_chunked_recv_tokens * hidden * recv_x.element_size() +     // Data buffer
                   num_channels * num_ranks * config.num_max_mtl_chunked_recv_tokens * sizeof(int) +                        // Source index buffer
                   num_channels * num_ranks * config.num_max_mtl_chunked_recv_tokens * num_topk * sizeof(int64_t) +         // Top-k index buffer
                   num_channels * num_ranks * config.num_max_mtl_chunked_recv_tokens * num_topk * sizeof(float) +           // Top-k weight buffer
                   num_channels * num_ranks * config.num_max_mtl_chunked_recv_tokens * sizeof(float) * num_scales           // FP8 scale buffer
                   <= num_mtl_bytes);
    intranode::dispatch(recv_x.data_ptr(), recv_x_scales_ptr, recv_src_idx.data_ptr<int>(), recv_topk_idx_ptr, recv_topk_weights_ptr, recv_channel_prefix_matrix.data_ptr<int>(),
                        send_head.data_ptr<int>(),
                        x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
                        is_token_in_rank.data_ptr<bool>(), channel_prefix_matrix.data_ptr<int>(),
                        num_tokens, static_cast<int>(hidden * recv_x.element_size() / sizeof(int4)), num_topk, num_experts, num_scales,
                        buffer_ptrs_gpu, rank, num_ranks, comm_stream, config.num_sms,
                        config.num_max_mtl_chunked_send_tokens, config.num_max_mtl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, is_token_in_rank, rank_prefix_matrix, channel_prefix_matrix, recv_x, recv_src_idx, recv_channel_prefix_matrix, send_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {x_scales, topk_idx, topk_weights, num_tokens_per_rank, num_tokens_per_expert, cached_channel_prefix_matrix, cached_rank_prefix_matrix, recv_topk_idx, recv_topk_weights, recv_x_scales}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::musa::setCurrentMUSAStream(compute_stream);

    // Return values
    return {recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list, rank_prefix_matrix, channel_prefix_matrix, recv_channel_prefix_matrix, recv_src_idx, send_head, event};
}

/*FIXME not found in pytorch musa*/
static musaDataType  ScalarTypeToMusaDataType(const c10::ScalarType& scalar_type){
    switch (scalar_type) {
    case c10::ScalarType::BFloat16:
      return MUSA_R_16BF;
    // case c10::ScalarType::Float8_e4m3fn:
    //   return MUSA_R_8F_E4M3;
    // case c10::ScalarType::Float8_e5m2:
    //   return MUSA_R_8F_E5M2;
    default:
        EP_HOST_ASSERT(0);  
    }
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::intranode_combine(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                          const torch::Tensor& src_idx, const torch::Tensor& rank_prefix_matrix, const torch::Tensor& channel_prefix_matrix,
                          const torch::Tensor& send_head, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_idx.dim() == 1 and src_idx.is_contiguous() and src_idx.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(send_head.dim() == 2 and send_head.is_contiguous() and send_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rank_prefix_matrix.dim() == 2 and rank_prefix_matrix.is_contiguous() and rank_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(channel_prefix_matrix.dim() == 2 and channel_prefix_matrix.is_contiguous() and channel_prefix_matrix.scalar_type() == torch::kInt32);

    // One channel use two blocks, even-numbered blocks for sending, odd-numbered blocks for receiving.
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    int num_channels = config.num_sms / 2;

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_recv_tokens = static_cast<int>(send_head.size(0));
    EP_HOST_ASSERT(src_idx.size(0) == num_tokens);
    EP_HOST_ASSERT(send_head.size(1) == num_ranks);
    EP_HOST_ASSERT(rank_prefix_matrix.size(0) == num_ranks and rank_prefix_matrix.size(1) == num_ranks);
    EP_HOST_ASSERT(channel_prefix_matrix.size(0) == num_ranks and channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::musa::getCurrentMUSAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::musa::setCurrentMUSAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    int num_topk = 0;
    auto recv_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }

    // Launch barrier and reset queue head and tail
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 <= num_mtl_bytes);
    intranode::cached_notify_combine(buffer_ptrs_gpu, send_head.data_ptr<int>(),
                                     num_channels, num_recv_tokens, num_channels * num_ranks * 2,
                                     task_fifo_ptrs_gpu, head, rank, num_ranks,
                                     comm_stream);
    // NOTES: this function uses two FIFO slots (barrier before and after)
    move_fifo_slots(2);

    // Combine data
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    EP_HOST_ASSERT(num_channels * num_ranks * sizeof(int) * 2 +                                                      // Queue head and tail
                   num_channels * num_ranks * config.num_max_mtl_chunked_recv_tokens * hidden * x.element_size() +    // Data buffer
                   num_channels * num_ranks * config.num_max_mtl_chunked_recv_tokens * sizeof(int) +                  // Source index buffer
                   num_channels * num_ranks * config.num_max_mtl_chunked_recv_tokens * num_topk * sizeof(float)       // Top-k weight buffer
                   <= num_mtl_bytes);
    /*FIXME not support*/
    intranode::combine(ScalarTypeToMusaDataType(x.scalar_type()),
                       recv_x.data_ptr(), recv_topk_weights_ptr,
                       x.data_ptr(), topk_weights_ptr,
                       src_idx.data_ptr<int>(), rank_prefix_matrix.data_ptr<int>(), channel_prefix_matrix.data_ptr<int>(),
                       send_head.data_ptr<int>(), num_tokens, num_recv_tokens, hidden, num_topk,
                       buffer_ptrs_gpu, rank, num_ranks,
                       comm_stream, config.num_sms,
                       config.num_max_mtl_chunked_send_tokens, config.num_max_mtl_chunked_recv_tokens);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, src_idx, send_head, rank_prefix_matrix, channel_prefix_matrix, recv_x}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {topk_weights, recv_topk_weights}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::musa::setCurrentMUSAStream(compute_stream);

    return {recv_x, recv_topk_weights, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::vector<int>, torch::Tensor, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, torch::Tensor, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::internode_dispatch(const torch::Tensor& x, const std::optional<torch::Tensor>& x_scales,
                           const std::optional<torch::Tensor>& topk_idx, const std::optional<torch::Tensor>& topk_weights,
                           const std::optional<torch::Tensor>& num_tokens_per_rank, const std::optional<torch::Tensor>& num_tokens_per_rdma_rank,
                           const torch::Tensor& is_token_in_rank, const std::optional<torch::Tensor>& num_tokens_per_expert,
                           int cached_num_recv_tokens, int cached_num_rdma_recv_tokens,
                           const std::optional<torch::Tensor>& cached_rdma_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_rdma_rank_prefix_sum,
                           const std::optional<torch::Tensor>& cached_gbl_channel_prefix_matrix, const std::optional<torch::Tensor>& cached_recv_gbl_rank_prefix_sum,
                           int expert_alignment, const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);
    EP_HOST_ASSERT(0 < get_num_rdma_ranks() and get_num_rdma_ranks() <= NUM_MAX_RDMA_PEERS);

    bool cached_mode = cached_rdma_channel_prefix_matrix.has_value();
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum.has_value());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix.has_value());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum.has_value());
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank.has_value());
        EP_HOST_ASSERT(num_tokens_per_expert.has_value());
    }

    // Type checks
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->scalar_type() == torch::kInt32);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->scalar_type() == torch::kInt32);
        EP_HOST_ASSERT(num_tokens_per_expert->scalar_type() == torch::kInt32);
    }

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT((x.size(1) * x.element_size()) % sizeof(int4) == 0);
    if (cached_mode) {
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->dim() == 2 and cached_rdma_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_rdma_channel_prefix_matrix->size(0) == num_rdma_ranks and cached_rdma_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->dim() == 1 and cached_recv_rdma_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_rdma_rank_prefix_sum->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->dim() == 2 and cached_gbl_channel_prefix_matrix->is_contiguous());
        EP_HOST_ASSERT(cached_gbl_channel_prefix_matrix->size(0) == num_ranks and cached_gbl_channel_prefix_matrix->size(1) == num_channels);
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->dim() == 1 and cached_recv_gbl_rank_prefix_sum->is_contiguous());
        EP_HOST_ASSERT(cached_recv_gbl_rank_prefix_sum->size(0) == num_ranks);
    } else {
        EP_HOST_ASSERT(num_tokens_per_rank->dim() == 1 and num_tokens_per_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->dim() == 1 and num_tokens_per_rdma_rank->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_expert->dim() == 1 and num_tokens_per_expert->is_contiguous());
        EP_HOST_ASSERT(num_tokens_per_rank->size(0) == num_ranks);
        EP_HOST_ASSERT(num_tokens_per_rdma_rank->size(0) == num_rdma_ranks);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) % num_ranks == 0);
        EP_HOST_ASSERT(num_tokens_per_expert->size(0) / num_ranks <= NUM_MAX_LOCAL_EXPERTS);
    }

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)), hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_experts = cached_mode ? 0 : static_cast<int>(num_tokens_per_expert->size(0)), num_local_experts = num_experts / num_ranks;

    // Top-k checks
    int num_topk = 0;
    int64_t* topk_idx_ptr = nullptr;
    float* topk_weights_ptr = nullptr;
    EP_HOST_ASSERT(topk_idx.has_value() == topk_weights.has_value());
    if (topk_idx.has_value()) {
        num_topk = static_cast<int>(topk_idx->size(1));
        EP_HOST_ASSERT(num_experts > 0);
        EP_HOST_ASSERT(topk_idx->dim() == 2 and topk_idx->is_contiguous());
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(num_tokens == topk_idx->size(0) and num_tokens == topk_weights->size(0));
        EP_HOST_ASSERT(num_topk == topk_weights->size(1));
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        topk_idx_ptr = topk_idx->data_ptr<int64_t>();
        topk_weights_ptr = topk_weights->data_ptr<float>();
    }

    // FP8 scales checks
    float* x_scales_ptr = nullptr;
    int num_scales = 0;
    if (x_scales.has_value()) {
        EP_HOST_ASSERT(x.element_size() == 1);
        EP_HOST_ASSERT(x_scales->scalar_type() == torch::kFloat32);
        EP_HOST_ASSERT(x_scales->dim() > 0 and x_scales->dim() < 3 and x_scales->is_contiguous());
        EP_HOST_ASSERT(x_scales->size(0) == num_tokens);
        num_scales = x_scales->dim() == 1 ? 1 : static_cast<int>(x_scales->size(1));
        x_scales_ptr = x_scales->data_ptr<float>();
    }

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::musa::getCurrentMUSAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::musa::setCurrentMUSAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Create handles (only return for non-cached mode)
    int num_recv_tokens = -1, num_rdma_recv_tokens = -1;
    auto rdma_channel_prefix_matrix = torch::Tensor();
    auto recv_rdma_rank_prefix_sum = torch::Tensor();
    auto gbl_channel_prefix_matrix = torch::Tensor();
    auto recv_gbl_rank_prefix_sum = torch::Tensor();
    std::vector<int> num_recv_tokens_per_expert_list;

    // Barrier or send sizes
    if (cached_mode) {
        num_recv_tokens = cached_num_recv_tokens;
        num_rdma_recv_tokens = cached_num_rdma_recv_tokens;
        rdma_channel_prefix_matrix = cached_rdma_channel_prefix_matrix.value();
        recv_rdma_rank_prefix_sum = cached_recv_rdma_rank_prefix_sum.value();
        gbl_channel_prefix_matrix = cached_gbl_channel_prefix_matrix.value();
        recv_gbl_rank_prefix_sum = cached_recv_gbl_rank_prefix_sum.value();

        // Just a barrier and clean flags
        internode::cached_notify(hidden_int4, num_scales, num_topk, num_topk,
                                 num_ranks, num_channels, 0, nullptr,
                                 nullptr, nullptr, nullptr,
                                 rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                                 buffer_ptrs_gpu, config.num_max_mtl_chunked_recv_tokens,
                                 task_fifo_ptrs_gpu, head, rank, comm_stream,
                                 config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                 num_mtl_bytes, true, low_latency_mode);
        move_fifo_slots(2);
    } else {
        rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kMUSA));
        recv_rdma_rank_prefix_sum = torch::empty({num_rdma_ranks}, dtype(torch::kInt32).device(torch::kMUSA));
        gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kMUSA));
        recv_gbl_rank_prefix_sum = torch::empty({num_ranks}, dtype(torch::kInt32).device(torch::kMUSA));

        // Send sizes
        *moe_recv_counter = -1, *moe_recv_rdma_counter = -1;
        for (int i = 0; i < num_local_experts; ++ i)
            moe_recv_expert_counter[i] = -1;
        internode::notify_dispatch(num_tokens_per_rank->data_ptr<int>(), moe_recv_counter_mapped, num_ranks,
                                   num_tokens_per_rdma_rank->data_ptr<int>(), moe_recv_rdma_counter_mapped,
                                   num_tokens_per_expert->data_ptr<int>(), moe_recv_expert_counter_mapped, num_experts,
                                   is_token_in_rank.data_ptr<bool>(), num_tokens, num_channels,
                                   hidden_int4, num_scales, num_topk, expert_alignment,
                                   rdma_channel_prefix_matrix.data_ptr<int>(), recv_rdma_rank_prefix_sum.data_ptr<int>(),
                                   gbl_channel_prefix_matrix.data_ptr<int>(), recv_gbl_rank_prefix_sum.data_ptr<int>(),
                                   rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                                   buffer_ptrs_gpu, config.num_max_mtl_chunked_recv_tokens,
                                   task_fifo_ptrs_gpu, head, rank, comm_stream,
                                   config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                                   num_mtl_bytes, low_latency_mode);
        move_fifo_slots(3);

        // Synchronize total received tokens and tokens per expert
        auto start_time = std::chrono::high_resolution_clock::now();
        while (true) {
            // Read total count
            num_recv_tokens = static_cast<int>(*moe_recv_counter);
            num_rdma_recv_tokens = static_cast<int>(*moe_recv_rdma_counter);

            // Read per-expert count
            bool ready = (num_recv_tokens >= 0) and (num_rdma_recv_tokens >= 0);
            for (int i = 0; i < num_local_experts and ready; ++ i)
                ready &= moe_recv_expert_counter[i] >= 0;

            if (ready)
                break;

            // Timeout check
            if (std::chrono::duration_cast<std::chrono::seconds>(std::chrono::high_resolution_clock::now() - start_time).count() > NUM_CPU_TIMEOUT_SECS) {
                printf("Global rank: %d, num_recv_tokens: %d, num_rdma_recv_tokens: %d\n", rank, num_recv_tokens, num_rdma_recv_tokens);
                for (int i = 0; i < num_local_experts; ++ i)
                    printf("moe_recv_expert_counter[%d]: %d\n", i, moe_recv_expert_counter[i]);
                throw std::runtime_error("DeepEP error: timeout (dispatch CPU)");
            }
        }
        num_recv_tokens_per_expert_list = std::vector<int>(moe_recv_expert_counter, moe_recv_expert_counter + num_local_experts);
    }

    // Allocate new tensors
    auto recv_x = torch::empty({num_recv_tokens, hidden}, x.options());
    auto recv_topk_idx = std::optional<torch::Tensor>(), recv_topk_weights = std::optional<torch::Tensor>(), recv_x_scales = std::optional<torch::Tensor>();
    auto recv_src_meta = std::optional<torch::Tensor>();
    auto recv_rdma_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto recv_gbl_channel_prefix_matrix = std::optional<torch::Tensor>();
    auto send_rdma_head = std::optional<torch::Tensor>();
    auto send_mtl_head = std::optional<torch::Tensor>();
    if (not cached_mode) {
        recv_src_meta = torch::empty({num_recv_tokens, internode::get_source_meta_bytes()}, dtype(torch::kByte).device(torch::kMUSA));
        recv_rdma_channel_prefix_matrix = torch::empty({num_rdma_ranks, num_channels}, dtype(torch::kInt32).device(torch::kMUSA));
        recv_gbl_channel_prefix_matrix = torch::empty({num_ranks, num_channels}, dtype(torch::kInt32).device(torch::kMUSA));
        send_rdma_head = torch::empty({num_tokens, num_rdma_ranks}, dtype(torch::kInt32).device(torch::kMUSA));
        send_mtl_head = torch::empty({num_rdma_recv_tokens, NUM_MAX_MTL_PEERS}, dtype(torch::kInt32).device(torch::kMUSA));
    }

    // Assign pointers
    int64_t* recv_topk_idx_ptr = nullptr;
    float* recv_topk_weights_ptr = nullptr;
    float* recv_x_scales_ptr = nullptr;
    if (topk_idx.has_value()) {
        recv_topk_idx = torch::empty({num_recv_tokens, num_topk}, topk_idx->options());
        recv_topk_weights = torch::empty({num_recv_tokens, num_topk}, topk_weights->options());
        recv_topk_idx_ptr = recv_topk_idx->data_ptr<int64_t>();
        recv_topk_weights_ptr = recv_topk_weights->data_ptr<float>();
    }
    if (x_scales.has_value()) {
        recv_x_scales = x_scales->dim() == 1 ?
                        torch::empty({num_recv_tokens}, x_scales->options()) :
                        torch::empty({num_recv_tokens, num_scales}, x_scales->options());
        recv_x_scales_ptr = recv_x_scales->data_ptr<float>();
    }

    // Launch data dispatch
    // NOTES: the buffer size checks are moved into the `.mu` file
    internode::dispatch(recv_x.data_ptr(), recv_x_scales_ptr, recv_topk_idx_ptr, recv_topk_weights_ptr,
                        cached_mode ? nullptr : recv_src_meta->data_ptr(),
                        x.data_ptr(), x_scales_ptr, topk_idx_ptr, topk_weights_ptr,
                        cached_mode ? nullptr : send_rdma_head->data_ptr<int>(), cached_mode ? nullptr : send_mtl_head->data_ptr<int>(),
                        cached_mode ? nullptr : recv_rdma_channel_prefix_matrix->data_ptr<int>(),
                        cached_mode ? nullptr : recv_gbl_channel_prefix_matrix->data_ptr<int>(),
                        rdma_channel_prefix_matrix.data_ptr<int>(), recv_rdma_rank_prefix_sum.data_ptr<int>(),
                        gbl_channel_prefix_matrix.data_ptr<int>(), recv_gbl_rank_prefix_sum.data_ptr<int>(),
                        num_tokens, hidden_int4, num_scales, num_topk, num_experts,
                        is_token_in_rank.data_ptr<bool>(),
                        rdma_buffer_ptr, config.num_max_rdma_chunked_send_tokens, config.num_max_rdma_chunked_recv_tokens,
                        buffer_ptrs_gpu, config.num_max_mtl_chunked_send_tokens, config.num_max_mtl_chunked_recv_tokens,
                        rank, num_ranks, cached_mode,
                        comm_stream, num_channels, low_latency_mode);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, is_token_in_rank, recv_x,
                       rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum, gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {x_scales, topk_idx, topk_weights,
                        num_tokens_per_rank, num_tokens_per_rdma_rank, num_tokens_per_expert,
                        cached_rdma_channel_prefix_matrix, cached_recv_rdma_rank_prefix_sum,
                        cached_gbl_channel_prefix_matrix, cached_recv_gbl_rank_prefix_sum,
                        recv_topk_idx, recv_topk_weights, recv_x_scales,
                        recv_rdma_channel_prefix_matrix, recv_gbl_channel_prefix_matrix, send_rdma_head, send_mtl_head,
                        recv_src_meta}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::musa::setCurrentMUSAStream(compute_stream);

    // Return values
    return {recv_x, recv_x_scales, recv_topk_idx, recv_topk_weights, num_recv_tokens_per_expert_list,
            rdma_channel_prefix_matrix, gbl_channel_prefix_matrix,
            recv_rdma_channel_prefix_matrix, recv_rdma_rank_prefix_sum,
            recv_gbl_channel_prefix_matrix, recv_gbl_rank_prefix_sum,
            recv_src_meta, send_rdma_head, send_mtl_head, event};
}

std::tuple<torch::Tensor, std::optional<torch::Tensor>, std::optional<EventHandle>>
Buffer::internode_combine(const torch::Tensor& x, const std::optional<torch::Tensor>& topk_weights,
                          const torch::Tensor& src_meta, const torch::Tensor& is_combined_token_in_rank,
                          const torch::Tensor& rdma_channel_prefix_matrix, const torch::Tensor& rdma_rank_prefix_sum, const torch::Tensor& gbl_channel_prefix_matrix,
                          const torch::Tensor& combined_rdma_head, const torch::Tensor& combined_mtl_head,
                          const Config& config, std::optional<EventHandle>& previous_event, bool async, bool allocate_on_comm_stream) {
    const int num_channels = config.num_sms / 2;
    EP_HOST_ASSERT(config.num_sms % 2 == 0);

    // Shape and contiguous checks
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous());
    EP_HOST_ASSERT(src_meta.dim() == 2 and src_meta.is_contiguous() and src_meta.scalar_type() == torch::kByte);
    EP_HOST_ASSERT(is_combined_token_in_rank.dim() == 2 and is_combined_token_in_rank.is_contiguous() and is_combined_token_in_rank.scalar_type() == torch::kBool);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.dim() == 2 and rdma_channel_prefix_matrix.is_contiguous() and rdma_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.dim() == 1 and rdma_rank_prefix_sum.is_contiguous() and rdma_rank_prefix_sum.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.dim() == 2 and gbl_channel_prefix_matrix.is_contiguous() and gbl_channel_prefix_matrix.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.is_contiguous() and combined_rdma_head.scalar_type() == torch::kInt32);
    EP_HOST_ASSERT(combined_mtl_head.dim() == 2 and combined_mtl_head.is_contiguous() and combined_mtl_head.scalar_type() == torch::kInt32);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1)), hidden_int4 = static_cast<int>(x.size(1) * x.element_size() / sizeof(int4));
    auto num_combined_tokens = static_cast<int>(is_combined_token_in_rank.size(0));
    EP_HOST_ASSERT((hidden * x.element_size()) % sizeof(int4) == 0);
    EP_HOST_ASSERT(src_meta.size(1) == internode::get_source_meta_bytes());
    EP_HOST_ASSERT(is_combined_token_in_rank.size(1) == num_ranks);
    EP_HOST_ASSERT(rdma_channel_prefix_matrix.size(0) == num_rdma_ranks and rdma_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(rdma_rank_prefix_sum.size(0) == num_rdma_ranks);
    EP_HOST_ASSERT(gbl_channel_prefix_matrix.size(0) == num_ranks and gbl_channel_prefix_matrix.size(1) == num_channels);
    EP_HOST_ASSERT(combined_rdma_head.dim() == 2 and combined_rdma_head.size(0) == num_combined_tokens and combined_rdma_head.size(1) == num_rdma_ranks);
    EP_HOST_ASSERT(combined_mtl_head.dim() == 2 and combined_mtl_head.size(1) == NUM_MAX_MTL_PEERS);

    // Allocate all tensors on comm stream if set
    // NOTES: do not allocate tensors upfront!
    auto compute_stream = at::musa::getCurrentMUSAStream();
    if (allocate_on_comm_stream) {
        EP_HOST_ASSERT(previous_event.has_value() and async);
        at::musa::setCurrentMUSAStream(comm_stream);
    }

    // Wait previous tasks to be finished
    if (previous_event.has_value()) {
        stream_wait(comm_stream, previous_event.value());
    } else {
        stream_wait(comm_stream, compute_stream);
    }

    // Top-k checks
    int num_topk = 0;
    auto combined_topk_weights = std::optional<torch::Tensor>();
    float* topk_weights_ptr = nullptr;
    float* combined_topk_weights_ptr = nullptr;
    if (topk_weights.has_value()) {
        EP_HOST_ASSERT(topk_weights->dim() == 2 and topk_weights->is_contiguous());
        EP_HOST_ASSERT(topk_weights->size(0) == num_tokens);
        EP_HOST_ASSERT(topk_weights->scalar_type() == torch::kFloat32);
        num_topk = static_cast<int>(topk_weights->size(1));
        topk_weights_ptr = topk_weights->data_ptr<float>();
        combined_topk_weights = torch::empty({num_combined_tokens, num_topk}, topk_weights->options());
        combined_topk_weights_ptr = combined_topk_weights->data_ptr<float>();
    }

    // Extra check for avoid-dead-lock design
    EP_HOST_ASSERT(config.num_max_mtl_chunked_recv_tokens % num_rdma_ranks == 0);
    EP_HOST_ASSERT(config.num_max_mtl_chunked_send_tokens <= config.num_max_mtl_chunked_recv_tokens / num_rdma_ranks);

    // Launch barrier and reset queue head and tail
    internode::cached_notify(hidden_int4, 0, 0, num_topk,
                             num_ranks, num_channels,
                             num_combined_tokens, combined_rdma_head.data_ptr<int>(),
                             rdma_channel_prefix_matrix.data_ptr<int>(), rdma_rank_prefix_sum.data_ptr<int>(), combined_mtl_head.data_ptr<int>(),
                             rdma_buffer_ptr, config.num_max_rdma_chunked_recv_tokens,
                             buffer_ptrs_gpu, config.num_max_mtl_chunked_recv_tokens,
                             task_fifo_ptrs_gpu, head, rank, comm_stream,
                             config.get_rdma_buffer_size_hint(hidden_int4 * sizeof(int4), num_ranks),
                             num_mtl_bytes, false, low_latency_mode);
    move_fifo_slots(2);

    // Launch data combine
    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());
    internode::combine(ScalarTypeToMusaDataType(x.scalar_type()),
                       combined_x.data_ptr(), combined_topk_weights_ptr,
                       is_combined_token_in_rank.data_ptr<bool>(),
                       x.data_ptr(), topk_weights_ptr,
                       combined_rdma_head.data_ptr<int>(), combined_mtl_head.data_ptr<int>(),
                       src_meta.data_ptr(), rdma_channel_prefix_matrix.data_ptr<int>(), rdma_rank_prefix_sum.data_ptr<int>(), gbl_channel_prefix_matrix.data_ptr<int>(),
                       num_tokens, num_combined_tokens, hidden, num_topk,
                       rdma_buffer_ptr, config.num_max_rdma_chunked_send_tokens, config.num_max_rdma_chunked_recv_tokens,
                       buffer_ptrs_gpu, config.num_max_mtl_chunked_send_tokens, config.num_max_mtl_chunked_recv_tokens,
                       rank, num_ranks, comm_stream, num_channels, low_latency_mode);

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        event = EventHandle(comm_stream);
        for (auto& t: {x, src_meta,
                       is_combined_token_in_rank, rdma_channel_prefix_matrix, rdma_rank_prefix_sum, gbl_channel_prefix_matrix,
                       combined_x, combined_rdma_head, combined_mtl_head}) {
            t.record_stream(comm_stream);
            if (allocate_on_comm_stream)
                t.record_stream(compute_stream);
        }
        for (auto& to: {topk_weights, combined_topk_weights}) {
            to.has_value() ? to->record_stream(comm_stream) : void();
            if (allocate_on_comm_stream)
                to.has_value() ? to->record_stream(compute_stream) : void();
        }
    } else {
        stream_wait(compute_stream, comm_stream);
    }

    // Switch back compute stream
    if (allocate_on_comm_stream)
        at::musa::setCurrentMUSAStream(compute_stream);

    // Return values
    return {combined_x, combined_topk_weights, event};
}

void Buffer::clean_low_latency_buffer(int num_max_dispatch_tokens_per_rank, int hidden, int num_experts) {
    EP_HOST_ASSERT(low_latency_mode);

    auto layout = LowLatencyLayout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    auto clean_meta_0 = layout.buffers[0].clean_meta();
    auto clean_meta_1 = layout.buffers[1].clean_meta();

    auto check_boundary = [=](void* ptr, size_t num_bytes) {
        auto offset = reinterpret_cast<int64_t>(ptr) - reinterpret_cast<int64_t>(rdma_buffer_ptr);
        EP_HOST_ASSERT(0 <= offset and offset + num_bytes <= num_rdma_bytes);
    };
    check_boundary(clean_meta_0.first, clean_meta_0.second * sizeof(int));
    check_boundary(clean_meta_1.first, clean_meta_1.second * sizeof(int));

    internode_ll::clean_low_latency_buffer(clean_meta_0.first, clean_meta_0.second,
                                           clean_meta_1.first, clean_meta_1.second,
                                           at::musa::getCurrentMUSAStream());
}

std::tuple<torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::low_latency_dispatch(const torch::Tensor& x, const torch::Tensor& topk_idx,
                             int num_max_dispatch_tokens_per_rank, int num_experts,
                             bool async, bool return_recv_hook) {
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    // By default using `ptp128c` FP8 cast
    EP_HOST_ASSERT(x.dim() == 2 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(1) % sizeof(int4) == 0 and x.size(1) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(x.size(0) == topk_idx.size(0) and x.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(num_experts % num_ranks == 0);

    auto num_tokens = static_cast<int>(x.size(0)), hidden = static_cast<int>(x.size(1));
    auto num_scales = hidden / 128, num_topk = static_cast<int>(topk_idx.size(1));
    int num_local_experts = num_experts / num_ranks;

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = at::musa::getCurrentMUSAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not (async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate packed tensors
    auto packed_recv_x = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank, hidden}, x.options().dtype(torch::kFloat8_e4m3fn));
    auto packed_recv_src_info = torch::empty({num_local_experts, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kInt32).device(torch::kMUSA));
    auto packed_recv_layout_range = torch::empty({num_local_experts, num_ranks}, torch::dtype(torch::kInt64).device(torch::kMUSA));
    auto packed_recv_count = torch::from_blob(buffer.dispatch_rdma_atomic_token_counter,
                                              {num_local_experts}, torch::dtype(torch::kInt32).device(torch::kMUSA));

    // Allocate column-majored scales
    EP_HOST_ASSERT((num_ranks * num_max_dispatch_tokens_per_rank) % 4 == 0 and "TMA requires the number of tokens to be multiple of 4");
    auto packed_recv_x_scales = torch::empty({num_local_experts, num_scales, num_ranks * num_max_dispatch_tokens_per_rank}, torch::dtype(torch::kFloat32).device(torch::kMUSA));
    packed_recv_x_scales = torch::transpose(packed_recv_x_scales, 1, 2);

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::dispatch(packed_recv_x.data_ptr(), packed_recv_x_scales.data_ptr<float>(),
                               packed_recv_src_info.data_ptr<int>(), packed_recv_layout_range.data_ptr<int64_t>(),
                               buffer.dispatch_rdma_recv_data_buffer, buffer.dispatch_rdma_recv_count_buffer,
                               buffer.dispatch_rdma_send_buffer,
                               x.data_ptr(), topk_idx.data_ptr<int64_t>(),
                               next_clean_meta.first, next_clean_meta.second,
                               num_tokens, hidden, num_max_dispatch_tokens_per_rank,
                               num_topk, num_experts, rank, num_ranks,
                               workspace, launch_stream, phases);
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {packed_recv_x, packed_recv_x_scales, packed_recv_count, packed_recv_src_info, packed_recv_layout_range, event, recv_hook};
}

std::tuple<torch::Tensor, std::optional<EventHandle>, std::optional<std::function<void()>>>
Buffer::low_latency_combine(const torch::Tensor& x, const torch::Tensor& topk_idx, const torch::Tensor& topk_weights,
                            const torch::Tensor& src_info, const torch::Tensor& layout_range,
                            int num_max_dispatch_tokens_per_rank, int num_experts,
                            bool async, bool return_recv_hook) {
    EP_HOST_ASSERT(low_latency_mode);

    // Tensor checks
    EP_HOST_ASSERT(x.dim() == 3 and x.is_contiguous() and x.scalar_type() == torch::kBFloat16);
    EP_HOST_ASSERT(x.size(0) == num_experts / num_ranks);
    EP_HOST_ASSERT(x.size(1) == num_ranks * num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(x.size(2) % sizeof(int4) == 0 and x.size(2) % 128 == 0);
    EP_HOST_ASSERT(topk_idx.dim() == 2 and topk_idx.is_contiguous());
    EP_HOST_ASSERT(topk_idx.size(0) == topk_weights.size(0) and topk_idx.size(1) == topk_weights.size(1));
    EP_HOST_ASSERT(topk_idx.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(topk_weights.dim() == 2 and topk_weights.is_contiguous());
    EP_HOST_ASSERT(topk_weights.size(0) <= num_max_dispatch_tokens_per_rank);
    EP_HOST_ASSERT(topk_weights.scalar_type() == torch::kFloat32);
    EP_HOST_ASSERT(src_info.dim() == 2 and src_info.is_contiguous());
    EP_HOST_ASSERT(src_info.scalar_type() == torch::kInt32 and x.size(0) == src_info.size(0));
    EP_HOST_ASSERT(layout_range.dim() == 2 and layout_range.is_contiguous());
    EP_HOST_ASSERT(layout_range.scalar_type() == torch::kInt64);
    EP_HOST_ASSERT(layout_range.size(0) == num_experts / num_ranks and layout_range.size(1) == num_ranks);
    auto hidden = static_cast<int>(x.size(2));
    auto num_local_experts = num_experts / num_ranks, num_topk = static_cast<int>(topk_weights.size(1));
    auto num_combined_tokens = static_cast<int>(topk_weights.size(0));

    // Buffer control
    LowLatencyLayout layout(rdma_buffer_ptr, num_max_dispatch_tokens_per_rank, hidden, num_ranks, num_experts);
    EP_HOST_ASSERT(layout.total_bytes <= num_rdma_bytes);
    auto buffer = layout.buffers[low_latency_buffer_idx];
    auto next_buffer = layout.buffers[low_latency_buffer_idx ^= 1];

    // Wait previous tasks to be finished
    // NOTES: the hook mode will always use the default stream
    auto compute_stream = at::musa::getCurrentMUSAStream();
    auto launch_stream = return_recv_hook ? compute_stream : comm_stream;
    EP_HOST_ASSERT(not (async and return_recv_hook));
    if (not return_recv_hook)
        stream_wait(launch_stream, compute_stream);

    // Allocate output tensor
    auto combined_x = torch::empty({num_combined_tokens, hidden}, x.options());

    // Kernel launch
    auto next_clean_meta = next_buffer.clean_meta();
    auto launcher = [=](int phases) {
        internode_ll::combine(combined_x.data_ptr(),
                              buffer.combine_rdma_recv_data_buffer, buffer.combine_rdma_recv_flag_buffer,
                              buffer.combine_rdma_send_buffer,
                              x.data_ptr(), topk_idx.data_ptr<int64_t>(), topk_weights.data_ptr<float>(),
                              src_info.data_ptr<int>(), layout_range.data_ptr<int64_t>(),
                              next_clean_meta.first, next_clean_meta.second,
                              num_combined_tokens, hidden, num_max_dispatch_tokens_per_rank,
                              num_topk, num_experts, rank, num_ranks,
                              workspace, launch_stream, phases);
    };
    launcher(return_recv_hook ? LOW_LATENCY_SEND_PHASE : (LOW_LATENCY_SEND_PHASE | LOW_LATENCY_RECV_PHASE));

    // Wait streams
    std::optional<EventHandle> event;
    if (async) {
        // NOTES: we must ensure the all tensors will not be deallocated before the stream-wait happens,
        // so in Python API, we must wrap all tensors into the event handle.
        event = EventHandle(launch_stream);
    } else if (not return_recv_hook) {
        stream_wait(compute_stream, launch_stream);
    }

    // Receiver callback
    std::optional<std::function<void()>> recv_hook = std::nullopt;
    if (return_recv_hook)
        recv_hook = [=]() { launcher(LOW_LATENCY_RECV_PHASE); };

    // Return values
    return {combined_x, event, recv_hook};
}

} // namespace deep_ep

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.doc() = "DeepEP: an efficient expert-parallel communication library";

    pybind11::class_<deep_ep::Config>(m, "Config")
        .def(pybind11::init<int, int, int, int, int>(),
             py::arg("num_sms") = 20,
             py::arg("num_max_mtl_chunked_send_tokens") = 6, py::arg("num_max_mtl_chunked_recv_tokens") = 256,
             py::arg("num_max_rdma_chunked_send_tokens") = 6, py::arg("num_max_rdma_chunked_recv_tokens") = 256)
        .def("get_mtl_buffer_size_hint", &deep_ep::Config::get_mtl_buffer_size_hint)
        .def("get_rdma_buffer_size_hint", &deep_ep::Config::get_rdma_buffer_size_hint);
    m.def("get_low_latency_rdma_size_hint", &deep_ep::get_low_latency_rdma_size_hint);

    pybind11::class_<deep_ep::EventHandle>(m, "EventHandle")
        .def(pybind11::init<>())
        .def("current_stream_wait", &deep_ep::EventHandle::current_stream_wait);

    pybind11::class_<deep_ep::Buffer>(m, "Buffer")
        .def(pybind11::init<int, int, int64_t, int64_t, bool>())
        .def("is_available", &deep_ep::Buffer::is_available)
        .def("get_num_rdma_ranks", &deep_ep::Buffer::get_num_rdma_ranks)
        .def("get_rdma_rank", &deep_ep::Buffer::get_rdma_rank)
        .def("get_root_rdma_rank", &deep_ep::Buffer::get_root_rdma_rank)
        .def("get_local_device_id", &deep_ep::Buffer::get_local_device_id)
        .def("get_local_ipc_handle", &deep_ep::Buffer::get_local_ipc_handle)
        .def("get_local_mtshmem_unique_id", &deep_ep::Buffer::get_local_mtshmem_unique_id)
        .def("get_local_buffer_tensor", &deep_ep::Buffer::get_local_buffer_tensor)
        .def("sync", &deep_ep::Buffer::sync)
        .def("get_dispatch_layout", &deep_ep::Buffer::get_dispatch_layout)
        .def("intranode_dispatch", &deep_ep::Buffer::intranode_dispatch)
        .def("intranode_combine", &deep_ep::Buffer::intranode_combine)
        .def("internode_dispatch", &deep_ep::Buffer::internode_dispatch)
        .def("internode_combine", &deep_ep::Buffer::internode_combine)
        .def("clean_low_latency_buffer", &deep_ep::Buffer::clean_low_latency_buffer)
        .def("low_latency_dispatch", &deep_ep::Buffer::low_latency_dispatch)
        .def("low_latency_combine", &deep_ep::Buffer::low_latency_combine);
}
