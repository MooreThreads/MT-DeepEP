#include "torch_musa/csrc/aten/musa/MUSAContext.h"
#include <memory>

#include "kernels/exception.muh"

namespace deep_ep {

struct EventHandle {
    std::shared_ptr<torch::Event> event;

    EventHandle() {
        event = std::make_shared<torch::Event>(torch::kMUSA);
        event->record(at::musa::getCurrentMUSAStream());
    }

    explicit EventHandle(const at::musa::MUSAStream& stream) {
        event = std::make_shared<torch::Event>(torch::kMUSA);
        event->record(stream);
    }

    EventHandle(const EventHandle& other) = default;

    void current_stream_wait() const {
        at::musa::getCurrentMUSAStream().unwrap().wait(*event);
    }
};

torch::Event create_event(const at::musa::MUSAStream &s) {
    auto event = torch::Event(torch::kMUSA);
    event.record(s);
    return event;
}

void stream_wait(const at::musa::MUSAStream& s_0, const at::musa::MUSAStream& s_1) {
    EP_HOST_ASSERT(s_0.id() != s_1.id());
    s_0.unwrap().wait(create_event(s_1));
}

void stream_wait(const at::musa::MUSAStream& s, const EventHandle& event) {
    s.unwrap().wait(*event.event);
}

} // namespace deep_ep
