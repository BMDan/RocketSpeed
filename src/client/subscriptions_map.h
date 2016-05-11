/// Copyright (c) 2014, Facebook, Inc.  All rights reserved.
/// This source code is licensed under the BSD-style license found in the
/// LICENSE file in the root directory of this source tree. An additional grant
/// of patent rights can be found in the PATENTS file in the same directory.
#pragma once

#include <cstdint>
#include <functional>
#include <memory>
#include <unordered_map>
#include <unordered_set>

#include "include/Types.h"
#include "src/messages/types.h"
#include "src/util/common/observable_container.h"
#include "src/util/common/ref_count_flyweight.h"

namespace rocketspeed {

class EventLoop;
class Flow;
class MessageDeliver;
class Logger;
class Slice;
template <typename>
class Sink;

/// A flyweight-pattern-based storage for topics and namespaces.
struct TenantAndNamespace {
  TenantID tenant_id;
  NamespaceID namespace_id;

  friend bool operator<(const TenantAndNamespace& lhs,
                        const TenantAndNamespace& rhs) {
    if (lhs.tenant_id == rhs.tenant_id) {
      return lhs.namespace_id < rhs.namespace_id;
    }
    return lhs.tenant_id < rhs.tenant_id;
  }
};
using TenantAndNamespaceFactory = RefCountFlyweightFactory<TenantAndNamespace>;
using TenantAndNamespaceFlyweight = RefCountFlyweight<TenantAndNamespace>;

/// A base information required by the SubscriptionsMap.
///
/// The layout is optimised primarly for memory usage, and secondary for the
/// performance of metadata updates.
template <typename S>
class SubscriptionBase {
 public:
  using SubscriptionID = S;

  SubscriptionBase(TenantAndNamespaceFlyweight tenant_and_namespace,
                   const Slice& topic_name,
                   SubscriptionID sub_id,
                   SequenceNumber initial_seqno)
  : tenant_and_namespace_(std::move(tenant_and_namespace))
  , topic_name_(topic_name.ToString())
  , sub_id_(sub_id)
  , expected_seqno_(initial_seqno) {}

  TenantID GetTenant() const { return tenant_and_namespace_.Get().tenant_id; }

  Slice GetNamespace() const {
    return tenant_and_namespace_.Get().namespace_id;
  }

  Slice GetTopicName() const { return topic_name_; }

  SequenceNumber GetExpectedSeqno() const { return expected_seqno_; }

  /// Returns true if the state transition carried by the update has been
  /// recorded and the update shall be delivered, false if the update could not
  /// be applied due to mismatched sequence numbers.
  bool ProcessUpdate(Logger* info_log,
                     SequenceNumber previous,
                     SequenceNumber current);

  /// This ID shall only be obtained for logging purposes.
  ///
  /// Caller may not rely on the ID being invariant for the whole duration of a
  /// subscription, but the ID is useful for debugging, as this is precisely
  /// what the server uses as an identification of the subscription.
  SubscriptionID GetIDForLogging() const { return sub_id_; }

 private:
  template <typename SubscriptionState>
  friend class SubscriptionsMap;

  const TenantAndNamespaceFlyweight tenant_and_namespace_;
  // TODO(stupaq): NTS
  const std::string topic_name_;
  /// An ID of this subscription known to the remote end.
  SubscriptionID sub_id_;
  /// Next expected sequence number on this subscription.
  SequenceNumber expected_seqno_;

  /// @{
  /// These methods shall not be accessed anyone but the SubscriptionMap that
  /// stores the subscription. No other piece of code may rely on invariance of
  /// a subscription ID stored _inside_ of the SubscriptionBase.
  /// The SubscriptionID can potentially change when a subscription is rewound.
  /// No intrusive map may contain the subscription when it happens.
  SubscriptionID GetSubscriptionID() const { return sub_id_; }

  void Rewind(SubscriptionID sub_id, SequenceNumber expected_seqno) {
    RS_ASSERT(sub_id_ != sub_id);
    sub_id_ = sub_id;
    expected_seqno_ = expected_seqno;
  }
  /// @}
};

/// A map of active subscriptions that replicates itself to the remove end over
/// provided sink and processes messages delivered on a subscription.
///
/// Stores absolutely minimal amount of information (per subscription) that is
/// needed to process updates and handle reconnections. Enabled users to attach
/// arbitrary state and functionality to a subscription.
///
/// The class is optimised for memory usage per subscription and is not
/// thread-safe.
// TODO(stupaq): generalise the sink and reshard by host in the proxy
template <typename SubscriptionState>
class SubscriptionsMap : public StreamReceiver {
 public:
  using SubscriptionID = typename SubscriptionState::SubscriptionID;

  using DeliverCb = std::function<void(
      Flow* flow, SubscriptionState*, std::unique_ptr<MessageDeliver>)>;
  using TerminateCb = std::function<void(
      Flow* flow, SubscriptionState*, std::unique_ptr<MessageUnsubscribe>)>;

  SubscriptionsMap(EventLoop* event_loop,
                   DeliverCb deliver_cb,
                   TerminateCb terminate_cb);

  /// Returns a non-owning pointer to the SubscriptionState.
  ///
  /// The pointer is valid until matching ::Unsubscribe call.
  SubscriptionState* Subscribe(SubscriptionID sub_id,
                               TenantID tenant_id,
                               const Slice& namespace_id,
                               const Slice& topic_name,
                               SequenceNumber initial_seqno);

  /// Returns a non-owning pointer to the SubscriptionState or null if doesn't
  /// exist.
  SubscriptionState* Find(SubscriptionID sub_id) const;

  /// Rewinds provided subscription to a given sequence number.
  void Rewind(SubscriptionState* ptr,
              SubscriptionID new_sub_id,
              SequenceNumber new_seqno);

  /// Returns true if a subscription was terminated, false if it didn't exist.
  void Unsubscribe(SubscriptionState* ptr);

  bool Empty() const;

  /// Forces the map to reestablish communication to the provided host.
  void ReconnectTo(const HostId& host);

 private:
  EventLoop* const event_loop_;
  const DeliverCb deliver_cb_;
  const TerminateCb terminate_cb_;

  TenantAndNamespaceFactory tenant_and_namespace_factory_;

  // TODO(stupaq): intrusive
  using Subscriptions =
      std::unordered_map<SubscriptionID, std::unique_ptr<SubscriptionState>>;
  ObservableContainer<Subscriptions> pending_subscriptions_;
  ObservableContainer<Subscriptions> synced_subscriptions_;
  // TODO(stupaq): sparse_hash_set
  using Unsubscribes = std::unordered_set<SubscriptionID>;
  ObservableContainer<Unsubscribes> pending_unsubscribes_;

  HostId last_host_;
  std::unique_ptr<Sink<SharedTimestampedString>> sink_;

  Logger* GetLogger();

  void HandlePendingSubscription(
      Flow* flow, std::unique_ptr<SubscriptionState> upstream_sub);

  void HandlePendingUnsubscription(Flow* flow, SubscriptionID sub_id);

  void ReceiveGoodbye(StreamReceiveArg<MessageGoodbye>) final override;
  void ReceiveUnsubscribe(StreamReceiveArg<MessageUnsubscribe>) final override;
  void ReceiveDeliver(StreamReceiveArg<MessageDeliver>) final override;
};

}  // namespace rocketspeed