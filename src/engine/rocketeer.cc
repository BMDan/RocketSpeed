// Copyright (c) 2014, Facebook, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.
//
#define __STDC_FORMAT_MACROS
#include "include/Rocketeer.h"

#include <functional>
#include <memory>
#include <sstream>
#include <string>

#include "external/folly/move_wrapper.h"

#include "include/Env.h"
#include "include/Logger.h"
#include "include/RocketSpeed.h"
#include "include/Slice.h"
#include "include/Status.h"
#include "include/Types.h"
#include "src/messages/flow_control.h"
#include "src/util/common/flow.h"
#include "src/util/common/retry_later_sink.h"
#include "src/util/common/subscription_id.h"
#include "src/util/common/hash.h"

namespace rocketspeed {

////////////////////////////////////////////////////////////////////////////////
SubscriptionID InboundID::GetSubID() const {
  return SubscriptionID::Unsafe(sub_id);
}

size_t InboundID::GetShard() const {
  return GetSubID().GetShardID();
}

size_t InboundID::Hash() const {
  return MurmurHash2<StreamID, uint64_t>()(stream_id, sub_id);
}

std::string InboundID::ToString() const {
  std::ostringstream ss;
  ss << "InboundID(" << stream_id << ", " << sub_id << ")";
  return ss.str();
}

////////////////////////////////////////////////////////////////////////////////
SubscriptionID RocketeerMessage::GetSubID() const {
  return SubscriptionID::Unsafe(sub_id);
}

////////////////////////////////////////////////////////////////////////////////
Rocketeer::Rocketeer()
: below_rocketeer_(nullptr)
, metadata_sink_(new RetryLaterSink<std::function<BackPressure()>>(
    [] (std::function<BackPressure()>& handler) { return handler(); })) {}

Rocketeer::~Rocketeer() = default;

BackPressure Rocketeer::TryHandleNewSubscription(
    InboundID inbound_id, SubscriptionParameters params) {
  RS_ASSERT(false) << "TryHandleNewSubscription is not implemented.";
  return BackPressure::None();
}

void Rocketeer::HandleNewSubscription(Flow* flow,
                                      InboundID inbound_id,
                                      SubscriptionParameters params) {
  // This is the default implementation of HandleNewSubscription.
  // Most application Rocketeers will implement TryHandleNewSubscription, but
  // internally RocketSpeed calls HandleNewSubscription.
  //
  // The default implementation forward the call to TryHandleNewSubscription
  // through a RetryLaterSink, which will retry the call later if the Try
  // called requested a retry.
  std::function<BackPressure()> cmd(
    [this, inbound_id, params] () {
      return TryHandleNewSubscription(inbound_id, params);
    });
  flow->Write(metadata_sink_.get(), cmd);
}

BackPressure Rocketeer::TryHandleTermination(
    InboundID inbound_id, TerminationSource source) {
  RS_ASSERT(false) << "TryHandleTermination is not implemented.";
  return BackPressure::None();
}

void Rocketeer::HandleTermination(
    Flow* flow, InboundID inbound_id, TerminationSource source) {
  // This is the default implementation of HandleTermination.
  // Most application Rocketeers will implement TryHandleTermination, but
  // internally RocketSpeed calls HandleTermination.
  //
  // The default implementation forward the call to TryHandleTermination
  // through a RetryLaterSink, which will retry the call later if the Try
  // called requested a retry.
  std::function<BackPressure()> cmd(
    [this, inbound_id, source] () {
      return TryHandleTermination(inbound_id, source);
    });
  flow->Write(metadata_sink_.get(), cmd);
}

void Rocketeer::Deliver(Flow* flow,
                        InboundID inbound_id,
                        SequenceNumber seqno,
                        std::string payload,
                        MsgId msg_id) {
  GetBelowRocketeer()->Deliver(
      flow, inbound_id, seqno, std::move(payload), msg_id);
}

void Rocketeer::DeliverBatch(Flow* flow,
                             StreamID stream_id,
                             std::vector<RocketeerMessage> messages) {
  GetBelowRocketeer()->DeliverBatch(flow, stream_id, std::move(messages));
}

void Rocketeer::Advance(Flow* flow,
                        InboundID inbound_id,
                        SequenceNumber seqno) {
  GetBelowRocketeer()->Advance(flow, inbound_id, seqno);
}

void Rocketeer::Terminate(Flow* flow,
                          InboundID inbound_id,
                          UnsubscribeReason reason) {
  GetBelowRocketeer()->Terminate(flow, inbound_id, reason);
}

}  // namespace rocketspeed
