//  Copyright (c) 2014, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.
//
#include <unistd.h>
#include <chrono>
#include <set>
#include <string>
#include <thread>
#include <vector>

#include "src/test/test_cluster.h"
#include "src/pilot/pilot.h"
#include "src/util/testharness.h"

namespace rocketspeed {

class PilotTest {
 public:
  // Create a new instance of the pilot
  PilotTest() : env_(Env::Default()) {
    // Create Logger
    ASSERT_OK(test::CreateLogger(env_, "PilotTest", &info_log_));
  }

  virtual ~PilotTest() {
    env_->WaitForJoin();  // This is good hygine
  }

 protected:
  Env* env_;
  EnvOptions env_options_;
  std::shared_ptr<Logger> info_log_;
  std::set<MsgId> sent_msgs_;
  std::set<MsgId> acked_msgs_;

  void ProcessDataAck(std::unique_ptr<Message> msg) {
    const MessageDataAck* acks = static_cast<const MessageDataAck*>(msg.get());
    for (const auto& ack : acks->GetAcks()) {
      ASSERT_EQ(ack.status, MessageDataAck::AckStatus::Success);
      acked_msgs_.insert(ack.msgid);  // mark msgid as ack'd
    }
  }

  // A static method that is the entry point of a background MsgLoop
  static void MsgLoopStart(void* arg) {
    MsgLoop* loop = reinterpret_cast<MsgLoop*>(arg);
    loop->Run();
  }
};

TEST(PilotTest, Publish) {
  // Create cluster with pilot only.
  LocalTestCluster cluster(info_log_, false, false, true);
  ASSERT_OK(cluster.GetStatus());

  port::Semaphore checkpoint;

  // create a client to communicate with the Pilot
  std::map<MessageType, MsgCallbackType> client_callback;
  client_callback[MessageType::mDataAck] =
      [this, &checkpoint](std::unique_ptr<Message> msg) {
    ProcessDataAck(std::move(msg));
    if (sent_msgs_.size() == acked_msgs_.size()) {
      checkpoint.Post();
    }
  };

  MsgLoop loop(env_, env_options_, 58499, 1, info_log_, "test");
  loop.RegisterCallbacks(client_callback);
  env_->StartThread(MsgLoopStart, &loop, "client");
  while (!loop.IsRunning()) {
    std::this_thread::yield();
  }

  // send messages to pilot
  NamespaceID nsid = 101;
  const bool is_new_request = true;
  for (int i = 0; i < 100; ++i) {
    std::string payload = std::to_string(i);
    std::string topic = "test" + std::to_string(i);
    std::string serial;
    MessageData data(MessageType::mPublish,
                     Tenant::GuestTenant,
                     ClientID("client1"),
                     Slice(topic),
                     nsid,
                     Slice(payload));
    data.SerializeToString(&serial);
    sent_msgs_.insert(data.GetMessageId());
    std::unique_ptr<Command> cmd(new SerializedSendCommand(
        std::move(serial),
        cluster.GetPilotHostIds().front().ToClientId(),
        env_->NowMicros(),
        is_new_request));
    ASSERT_OK(loop.SendCommand(std::move(cmd)));
  }

  // Ensure all messages were ack'd
  ASSERT_TRUE(checkpoint.TimedWait(std::chrono::seconds(100)));
  ASSERT_TRUE(sent_msgs_ == acked_msgs_);

  const Statistics& stats = cluster.GetStatistics();
  std::string stats_report = stats.Report();
  ASSERT_NE(stats_report.find("rocketspeed.pilot.append_requests:        100"),
            std::string::npos);
  ASSERT_NE(stats_report.find("rocketspeed.pilot.failed_appends:         0"),
            std::string::npos);
  ASSERT_NE(stats_report.find("rocketspeed.pilot.append_latency_us"),
            std::string::npos);
}

}  // namespace rocketspeed

int main(int argc, char** argv) {
  return rocketspeed::test::RunAllTests();
}
