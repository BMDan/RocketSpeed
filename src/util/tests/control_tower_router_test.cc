//  Copyright (c) 2014, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.

#include <algorithm>
#include <map>
#include <set>
#include <string>
#include <unordered_map>

#include "include/HostId.h"
#include "src/util/control_tower_router.h"
#include "src/util/testharness.h"
#include "src/util/testutil.h"

namespace rocketspeed {

class ConsistentHashTowerRouterTest : public ::testing::Test { };
class RendezvousHashTowerRouterTest : public ::testing::Test { };

std::unordered_map<ControlTowerId, HostId> MakeControlTowers(int num) {
  std::unordered_map<ControlTowerId, HostId> control_towers;
  for (int i = 0; i < num; ++i) {
    control_towers.emplace(i, HostId::CreateLocal(static_cast<uint16_t>(i)));
  }
  return control_towers;
}

struct MakeCHRouter {
  ConsistentHashTowerRouter
  operator()(std::unordered_map<ControlTowerId, HostId> hosts, size_t n) {
    return ConsistentHashTowerRouter(std::move(hosts), 100, n);
  }
};

struct MakeRHRouter {
  RendezvousHashTowerRouter
  operator()(std::unordered_map<ControlTowerId, HostId> hosts, size_t n) {
    return RendezvousHashTowerRouter(std::move(hosts), n);
  }
};

template <typename Func>
void ConsistencyTest(Func make_router) {
  // Test that log mapping changes minimally when increasing number of CTs.
  const int num_towers = 1000;
  const size_t num_copies = 3;
  auto router1 =
    make_router(MakeControlTowers(num_towers), num_copies);
  auto router2 =
    make_router(MakeControlTowers(num_towers * 105 / 100), num_copies);

  // Count number of changes for 100k logs.
  int num_relocations = 0;
  const int num_logs = 100000;
  for (int i = 0; i < num_logs; ++i) {
    std::vector<HostId const*> hosts1;
    std::vector<HostId const*> hosts2;
    ASSERT_TRUE(router1.GetControlTowers(i, &hosts1).ok());
    ASSERT_TRUE(router2.GetControlTowers(i, &hosts2).ok());
    ASSERT_EQ(hosts1.size(), num_copies);
    ASSERT_EQ(hosts2.size(), num_copies);

    auto host_less = [](HostId const* lhs, HostId const* rhs) {
      return *lhs < *rhs;
    };
    std::sort(hosts1.begin(), hosts1.end(), host_less);
    std::sort(hosts2.begin(), hosts2.end(), host_less);
    std::vector<HostId const*> intersection(num_copies);
    auto intersection_end = std::set_intersection(
      hosts1.begin(), hosts1.end(),
      hosts2.begin(), hosts2.end(),
      intersection.begin(),
      host_less);
    num_relocations += static_cast<int>(intersection.end() - intersection_end);
  }

  // Ideally ~5% should change, but allow for up to 2-8% margin of error.
  ASSERT_LT(num_relocations, (num_logs * num_copies) * 8 / 100);
  ASSERT_GT(num_relocations, (num_logs * num_copies) * 2 / 100);
}

TEST_F(ConsistentHashTowerRouterTest, ConsistencyTestCH) {
  ConsistencyTest(MakeCHRouter());
}

TEST_F(RendezvousHashTowerRouterTest, ConsistencyTestRH) {
  ConsistencyTest(MakeRHRouter());
}

template <typename Func>
void LogDistribution(Func make_router) {
  // Test that logs are well distributed among control towers
  int num_control_towers = 1000;
  auto control_towers = MakeControlTowers(num_control_towers);
  std::unordered_map<HostId, int> log_count;
  for (const auto& entry : control_towers) {
    log_count[entry.second] = 0;
  }
  auto router = make_router(std::move(control_towers), 1);

  // Count number of changed for 100k logs.
  int num_logs = 100000;
  for (int i = 0; i < num_logs; ++i) {
    std::vector<HostId const*> clients;
    ASSERT_TRUE(router.GetControlTowers(i, &clients).ok());
    log_count[*clients[0]]++;
  }

  // Find the minimum and maximum logs per control tower.
  auto minmax = std::minmax_element(log_count.begin(), log_count.end());
  int expected = num_logs / num_control_towers;  // perfect distribution
  ASSERT_GT(minmax.first->second, expected * 0.5);  // allow +/-50% error
  ASSERT_LT(minmax.second->second, expected * 1.6);
}

TEST_F(ConsistentHashTowerRouterTest, LogDistributionCH) {
  LogDistribution(MakeCHRouter());
}

TEST_F(RendezvousHashTowerRouterTest, LogDistributionRH) {
  LogDistribution(MakeRHRouter());
}

template <typename Func>
void ChangeHost(Func make_router) {
  // Test that we can swap an existing host with a new one, without changing
  // the mapping.
  std::unordered_map<ControlTowerId, HostId> control_towers {
    { 0, HostId::CreateLocal(0) },
    { 1, HostId::CreateLocal(1) },
    { 2, HostId::CreateLocal(2) },
  };
  const int num_logs = 10000;

  std::unordered_map<HostId, std::set<LogID>> host_logs_before;
  {  // Determine logs serviced by each host.
    auto router = make_router(control_towers, 1);
    for (int i = 0; i < num_logs; ++i) {
      std::vector<HostId const*> host_ids;
      ASSERT_OK(router.GetControlTowers(i, &host_ids));
      ASSERT_TRUE(!host_ids.empty());
      host_logs_before[*host_ids[0]].insert(i);
    }
  }

  // Swap out node 1 with a new host:
  control_towers[1] = HostId::CreateLocal(3);

  std::unordered_map<HostId, std::set<LogID>> host_logs_after;
  {  // Determine logs serviced by each host after the swap.
    auto router = make_router(control_towers, 1);
    for (int i = 0; i < num_logs; ++i) {
      std::vector<HostId const*> host_ids;
      ASSERT_OK(router.GetControlTowers(i, &host_ids));
      ASSERT_TRUE(!host_ids.empty());
      host_logs_after[*host_ids[0]].insert(i);
    }
  }

  // All logs for Host(1) should now be serviced by HostId(3).
  ASSERT_TRUE(host_logs_before[HostId::CreateLocal(1)] ==
              host_logs_after[HostId::CreateLocal(3)]);
}

TEST_F(ConsistentHashTowerRouterTest, ChangeHostCH) {
  ChangeHost(MakeCHRouter());
}

TEST_F(RendezvousHashTowerRouterTest, ChangeHostRH) {
  ChangeHost(MakeRHRouter());
}

}  // namespace rocketspeed

int main(int argc, char** argv) {
  return rocketspeed::test::RunAllTests(argc, argv);
}
