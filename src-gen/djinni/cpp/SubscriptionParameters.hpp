// AUTOGENERATED FILE - DO NOT MODIFY!
// This file generated by Djinni from rocketspeed.djinni

#pragma once

#include <cstdint>
#include <string>
#include <utility>

namespace rocketspeed { namespace djinni {

struct SubscriptionParameters final {
    int32_t tenant_id;
    std::string namespace_id;
    std::string topic_name;
    int64_t start_seqno;

    friend bool operator==(const SubscriptionParameters& lhs, const SubscriptionParameters& rhs);
    friend bool operator!=(const SubscriptionParameters& lhs, const SubscriptionParameters& rhs);

    SubscriptionParameters(int32_t tenant_id,
                           std::string namespace_id,
                           std::string topic_name,
                           int64_t start_seqno)
    : tenant_id(std::move(tenant_id))
    , namespace_id(std::move(namespace_id))
    , topic_name(std::move(topic_name))
    , start_seqno(std::move(start_seqno))
    {}
};

} }  // namespace rocketspeed::djinni
