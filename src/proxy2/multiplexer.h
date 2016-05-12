/// Copyright (c) 2014, Facebook, Inc.  All rights reserved.
/// This source code is licensed under the BSD-style license found in the
/// LICENSE file in the root directory of this source tree. An additional grant
/// of patent rights can be found in the PATENTS file in the same directory.
#pragma once

#include <memory>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>

#include "include/Types.h"
#include "src/client/subscriptions_map.h"
#include "src/util/common/hash.h"
#include "src/util/id_allocator.h"

namespace rocketspeed {

class EventLoop;
class Flow;
class Multiplexer;
class PerShard;
class PerStream;
class ProxyServerOptions;
class Slice;
class Stream;
using SubscriptionID = uint64_t;
class UpdatesAccumulator;

// TODO(stupaq):
/// As we already know the shard in the Multiplexer, we can just as well use <8
/// bytes for subscription IDs. Once we loop around the IDs we can:
/// * kill the Stream,
/// * reallocate SubscriptionIDs,
/// * resync all subscriptions using a new Stream.
class UpstreamAllocator : public IDAllocator<uint64_t, UpstreamAllocator> {
  using Base = IDAllocator<uint64_t, UpstreamAllocator>;

 public:
  using Base::Base;
};

class UpstreamSubscription
    : public SubscriptionBase<UpstreamAllocator::IDType> {
  using Base = SubscriptionBase<UpstreamAllocator::IDType>;

 public:
  using Base::Base;
  using DownstreamSubscriptionsSet =
      std::unordered_set<std::pair<PerStream*, SubscriptionID>,
                         MurmurHash2<std::pair<PerStream*, SubscriptionID>>>;

  UpdatesAccumulator* GetAccumulator() const { return accumulator_.get(); }

  void SetAccumulator(std::unique_ptr<UpdatesAccumulator> accumulator) {
    RS_ASSERT(!accumulator_);
    accumulator_ = std::move(accumulator);
  }

  void AddDownstream(PerStream* per_stream,
                     SubscriptionID downstream_sub,
                     SequenceNumber initial_seqno);

  size_t RemoveDownstream(PerStream* per_stream, SubscriptionID downstream_sub);

  void ReceiveDeliver(PerShard* per_shard,
                      Flow* flow,
                      std::unique_ptr<MessageDeliver> deliver);

  void ReceiveTerminate(PerShard* per_shard,
                        Flow* flow,
                        std::unique_ptr<MessageUnsubscribe> unsubscribe);

 private:
  std::unique_ptr<UpdatesAccumulator> accumulator_;
  SequenceNumber expected_seqno_{0};
  // TODO(stupaq): optimise for small cardinality
  DownstreamSubscriptionsSet downstream_subscriptions_;
};

/// A subscription-level proxy (per stream of subscriptions).
///
/// Multiplexer's memory requirements may be linear in the total number of
/// active subscriptions it learns about.
class Multiplexer {
 public:
  explicit Multiplexer(PerShard* per_shard);

  EventLoop* GetLoop() const;
  const ProxyServerOptions& GetOptions() const;

  /// Handles a subscription that was chosen for multiplexing.
  ///
  /// Returned handle to the subscription state is valid until matching
  /// unsubscribe call.
  UpstreamSubscription* Subscribe(Flow* flow,
                                  TenantID tenant_id,
                                  const Slice& namespace_id,
                                  const Slice& topic_name,
                                  SequenceNumber initial_seqno,
                                  PerStream* per_stream,
                                  SubscriptionID downstream_sub);

  void Unsubscribe(Flow* flow,
                   UpstreamSubscription* upstream_sub,
                   PerStream* per_stream,
                   SubscriptionID downstream_sub);

  void ChangeRoute();

  ~Multiplexer();

 private:
  PerShard* const per_shard_;

  UpstreamAllocator upstream_allocator_;
  SubscriptionsMap<UpstreamSubscription> subscriptions_map_;

  // TODO(stupaq): intrusive
  std::unordered_map<std::pair<std::string, std::string>,
                     UpstreamSubscription*,
                     MurmurHash2<std::pair<std::string, std::string>>>
      topic_index_;

  UpstreamSubscription* FindInIndex(const Slice& namespace_id,
                                    const Slice& topic_name);

  void InsertIntoIndex(UpstreamSubscription* upstream_sub);

  void RemoveFromIndex(UpstreamSubscription* upstream_sub);

  void ReceiveDeliver(Flow* flow,
                      UpstreamSubscription*,
                      std::unique_ptr<MessageDeliver>);

  void ReceiveTerminate(Flow* flow,
                        UpstreamSubscription*,
                        std::unique_ptr<MessageUnsubscribe> unsubscribe);
};

}  // namespace rocketspeed
