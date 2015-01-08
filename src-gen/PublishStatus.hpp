// AUTOGENERATED FILE - DO NOT MODIFY!
// This file generated by Djinni from rocketspeed.djinni

#pragma once

#include "MsgIdImpl.hpp"
#include "Status.hpp"
#include <optional>
#include <utility>

namespace rocketspeed { namespace djinni {

struct PublishStatus final {

    Status status;

    std::optional<MsgIdImpl> message_id;


    PublishStatus(
            Status status,
            std::optional<MsgIdImpl> message_id) :
                status(std::move(status)),
                message_id(std::move(message_id)) {
    }
};

} }  // namespace rocketspeed::djinni
