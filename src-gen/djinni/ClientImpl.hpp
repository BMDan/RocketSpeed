// AUTOGENERATED FILE - DO NOT MODIFY!
// This file generated by Djinni from rocketspeed.djinni

#pragma once

#include "ConfigurationImpl.hpp"
#include "MsgIdImpl.hpp"
#include "PublishStatus.hpp"
#include "RetentionBase.hpp"
#include "SubscriptionRequestImpl.hpp"
#include "SubscriptionStorage.hpp"
#include <cstdint>
#include <experimental/optional>
#include <memory>
#include <string>
#include <vector>

namespace rocketspeed { namespace djinni {

class PublishCallbackImpl;
class ReceiveCallbackImpl;
class SubscribeCallbackImpl;

class ClientImpl {
public:
    virtual ~ClientImpl() {}

    static std::shared_ptr<ClientImpl> Open(const ConfigurationImpl & config, const std::string & client_id, const std::shared_ptr<SubscribeCallbackImpl> & subscribe_callback, const std::shared_ptr<ReceiveCallbackImpl> & receive_callback, const SubscriptionStorage & storage);

    virtual PublishStatus Publish(int16_t namespace_id, const std::string & topic_name, RetentionBase retention, const std::vector<uint8_t> & data, const std::experimental::optional<MsgIdImpl> & message_id, const std::experimental::optional<std::shared_ptr<PublishCallbackImpl>> & publish_callback) = 0;

    virtual void ListenTopics(const std::vector<SubscriptionRequestImpl> & names) = 0;

    virtual void Acknowledge(int16_t namespace_id, const std::string & topic_name, int64_t sequence_number) = 0;

    virtual void Close() = 0;
};

} }  // namespace rocketspeed::djinni
