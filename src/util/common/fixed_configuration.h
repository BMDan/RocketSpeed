// Copyright (c) 2014, Facebook, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.
//
#pragma once

#include "include/Status.h"
#include "include/Types.h"
#include "include/HostId.h"

namespace rocketspeed {

/**
 * Simple implementation of PublisherRouter where there is a single pilot
 * to connect to. This is useful for controlled situations like
 * testing and benchmarking where the hosts are known beforehand, and are
 * unlikely to change.
 */
class FixedPublisherRouter : public PublisherRouter {
 public:
  explicit FixedPublisherRouter(HostId pilot): pilot_(pilot) {}

  Status GetPilot(HostId* host_out) const override;

 private:
  HostId pilot_;
};

/**
 * Implementation of simple ShardingStrategy where there is only one copilot
 * host, representing a single shard.
 */
class FixedShardingStrategy : public ShardingStrategy {
 public:
  explicit FixedShardingStrategy(HostId copilot, size_t shards = 1)
  : copilot_(copilot)
  , shards_(shards) {}

  size_t GetShard(Slice namespace_id, Slice topic_name) const override;
  size_t GetVersion() override { return 0; }
  HostId GetHost(size_t) override { return copilot_; }
  void MarkHostDown(const HostId&) override {}

 private:
  HostId copilot_;
  const size_t shards_;
};

/**
 * Parse config_str and generate fixed configuration strategies.
 * Recognized keys: pilot, copilot
 *
 * Example: "pilot=192.168.1.4;copilot=192.168.1.5"
 */
Status CreateFixedConfiguration(
    const std::string& config_str,
    std::unique_ptr<PublisherRouter>* out_publisher,
    std::unique_ptr<ShardingStrategy>* out_sharding);

}  // namespace rocketspeed
