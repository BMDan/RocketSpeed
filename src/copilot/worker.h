// Copyright (c) 2014, Facebook, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.
//
#pragma once

#include <map>
#include <memory>
#include <unordered_map>
#include <vector>
#include "include/Types.h"
#include "src/copilot/options.h"
#include "src/messages/messages.h"
#include "src/messages/msg_client.h"
#include "src/messages/msg_loop.h"
#include "src/util/control_tower_router.h"

namespace rocketspeed {

/**
 * Copilot worker. The copilot will allocate several of these, ideally one
 * per hardware thread. The workers take load off of the main thread by handling
 * the log appends and ack sending, and allows us to scale to multiple cores.
 */
class CopilotWorker {
 public:
  // Constructs a new CopilotWorker (does not start a thread).
  CopilotWorker(const CopilotOptions& options,
                const ControlTowerRouter* control_tower_router,
                MsgClient* msg_client);

  // Forward a message to this worker for processing.
  // This will asynchronously append the message into the log storage,
  // and then send an ack back to the to the message origin.
  void Forward(LogID logid, std::unique_ptr<Message> msg);

  // Start the message loop on this thread.
  // Blocks until the message loop ends.
  void Run() {
    msg_loop_->Run();
  }

  // Stop the message loop.
  void Stop() {
    msg_loop_->Stop();
  }

  // Check if the message loop is running.
  bool IsRunning() const {
    return msg_loop_->IsRunning();
  }

  // Get the host id of this worker's message loop.
  HostId GetHostId() const {
    return HostId(options_.copilotname, options_.port_number);
  }

 private:
  // Callback for message loop commands.
  void CommandCallback(std::unique_ptr<Command> command);

  // Send an ack message to the host for the msgid.
  void SendAck(const HostId& host,
               const MsgId& msgid,
               MessageDataAck::AckStatus status);

  // Add a subscriber to a topic.
  void ProcessSubscribe(MessageMetadata* msg,
                        const TopicPair& request,
                        LogID logid);

  // Remove a subscriber from a topic.
  void ProcessUnsubscribe(MessageMetadata* msg,
                          const TopicPair& request,
                          LogID logid);

  // Process a metadata response from control tower.
  void ProcessMetadataResponse(MessageMetadata* msg,
                               const TopicPair& request);

  // Forward data to subscribers.
  void ProcessData(MessageData* msg);

  // Main message loop for this worker.
  std::unique_ptr<MsgLoop> msg_loop_;

  // Copilot specific options.
  const CopilotOptions& options_;

  // MsgLoop callback map.
  std::map<MessageType, MsgCallbackType> callbacks_;

  // Shared router for control towers.
  const ControlTowerRouter* control_tower_router_;

  // MsgClient from CoPilot, this is shared by all workers.
  MsgClient* msg_client_;

  // Subscription metadata
  struct Subscription {
    Subscription(HostId const& host_id, SequenceNumber seqno, bool awaiting_ack)
    : host_id(host_id)
    , seqno(seqno)
    , awaiting_ack(awaiting_ack) {}

    HostId host_id;        // The subscriber
    SequenceNumber seqno;  // Lowest seqno to accept
    bool awaiting_ack;     // Is the subscriber awaiting an subscribe response?
  };

  // Map of topics to active subscriptions.
  std::unordered_map<Topic, std::vector<Subscription>> subscriptions_;
};

}  // namespace rocketspeed
