// Copyright (c) 2014, Facebook, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.
//
#define __STDC_FORMAT_MACROS
#include "src/client/client.h"

#include <cassert>
#include <chrono>
#include <functional>
#include <memory>
#include <mutex>
#include <thread>
#include <unordered_map>
#include <unordered_set>

#include "external/folly/move_wrapper.h"

#include "include/Logger.h"
#include "include/RocketSpeed.h"
#include "include/Slice.h"
#include "include/Status.h"
#include "include/SubscriptionStorage.h"
#include "include/Types.h"
#include "include/WakeLock.h"
#include "src/client/message_received.h"
#include "src/client/smart_wake_lock.h"
#include "src/messages/msg_loop_base.h"
#include "src/port/port.h"
#include "src/util/common/hash.h"

#ifndef USE_MQTTMSGLOOP
#include "src/messages/msg_loop.h"
#else
#include "rocketspeed/mqttclient/configuration.h"
#include "rocketspeed/mqttclient/mqtt_msg_loop.h"
#include "rocketspeed/mqttclient/proxygen_mqtt.h"
#endif

namespace rocketspeed {

////////////////////////////////////////////////////////////////////////////////
/** Represents a state of a single subscription. */
class SubscriptionState {
 public:
  // Noncopyable
  SubscriptionState(const SubscriptionState&) = delete;
  SubscriptionState& operator=(const SubscriptionState&) = delete;
  // Movable
  SubscriptionState(SubscriptionState&&) = default;
  SubscriptionState& operator=(SubscriptionState&&) = default;

  SubscriptionState(SubscriptionParameters parameters,
                    SubscribeCallback subscription_callback,
                    MessageReceivedCallback deliver_callback)
      : tenant_id_(parameters.tenant_id)
      , namespace_id_(std::move(parameters.namespace_id))
      , topic_name_(std::move(parameters.topic_name))
      , subscription_callback_(std::move(subscription_callback))
      , deliver_callback_(std::move(deliver_callback))
      , expected_seqno_(parameters.start_seqno)
      // If we were to restore state from subscription storage before the
      // subscription advances, we would restore from the next sequence number,
      // that is why we persist the previous one.
      , last_acked_seqno_(
            parameters.start_seqno == 0 ? 0 : parameters.start_seqno - 1) {}

  TenantID GetTenant() const { return tenant_id_; }

  const NamespaceID& GetNamespace() const { return namespace_id_; }

  const Topic& GetTopicName() const { return topic_name_; }

  void AssignID(const std::shared_ptr<Logger>& info_log,
                SubscriptionID sub_id) {
    thread_check_.Check();
    LOG_INFO(info_log,
             "Subscription on Topic(%s, %s)@%" PRIu64
             " for tenant %u assigned ID (%u)",
             namespace_id_.c_str(),
             topic_name_.c_str(),
             expected_seqno_,
             tenant_id_,
             sub_id);
  }

  /**
   * Processes unsubscribe message, optionally announces its status and decides
   * on its fate.
   */
  enum class Action {
    kTerminate,
    kResubscribe,
  };
  Action ProcessMessage(const std::shared_ptr<Logger>& info_log,
                        const MessageUnsubscribe& unsubscribe);

  /** Processes gap message, gap messages are not passed to the application. */
  void ReceiveMessage(const std::shared_ptr<Logger>& info_log,
                      std::unique_ptr<MessageDeliverGap> gap);

  /** Processes data message, and delivers it to the application. */
  void ReceiveMessage(const std::shared_ptr<Logger>& info_log,
                      std::unique_ptr<MessageDeliverData> data);

  /** Returns a lower bound on the seqno of the next expected message. */
  SequenceNumber GetExpected() const {
    thread_check_.Check();
    return expected_seqno_;
  }

  /** Marks provided sequence number as acknowledged. */
  void Acknowledge(SequenceNumber seqno);

  /** Returns sequnce number of last acknowledged message. */
  SequenceNumber GetLastAcknowledged() const {
    thread_check_.Check();
    return last_acked_seqno_;
  }

 private:
  ThreadCheck thread_check_;

  const TenantID tenant_id_;
  const NamespaceID namespace_id_;
  const Topic topic_name_;
  const SubscribeCallback subscription_callback_;
  const MessageReceivedCallback deliver_callback_;

  /** Next expected sequence number on this subscription. */
  SequenceNumber expected_seqno_;
  /** Seqence number of the last acknowledged message. */
  SequenceNumber last_acked_seqno_;

  /** Returns true iff message arrived in order and not duplicated. */
  bool ProcessMessage(const std::shared_ptr<Logger>& info_log,
                      const MessageDeliver& deliver);

  /** Announces status of a subscription via user defined callback. */
  void AnnounceStatus(bool subscribed, Status status);
};

SubscriptionState::Action SubscriptionState::ProcessMessage(
    const std::shared_ptr<Logger>& info_log,
    const MessageUnsubscribe& unsubscribe) {
  thread_check_.Check();

  switch (unsubscribe.GetReason()) {
    case MessageUnsubscribe::Reason::kRequested:
      LOG_DEBUG(info_log,
                "Terminated subscription ID (%u) on Topic(%s, %s)@%" PRIu64,
                unsubscribe.GetSubID(),
                namespace_id_.c_str(),
                topic_name_.c_str(),
                expected_seqno_);
      AnnounceStatus(false, Status::OK());
      return Action::kTerminate;
    case MessageUnsubscribe::Reason::kBackOff:
      LOG_INFO(info_log, "Resubscribing with ID (%u) on Topic(%s, %s)@%" PRIu64,
               unsubscribe.GetSubID(),
               namespace_id_.c_str(),
               topic_name_.c_str(),
               expected_seqno_);
      // We will silently resubscribe, don't announce subscription status.
      return Action::kResubscribe;
    case MessageUnsubscribe::Reason::kInvalid:
      LOG_WARN(
          info_log,
          "Terminated invalid subscription ID (%u) on Topic(%s, %s)@%" PRIu64,
          unsubscribe.GetSubID(),
          namespace_id_.c_str(),
          topic_name_.c_str(),
          expected_seqno_);
      AnnounceStatus(false, Status::InvalidArgument("Invalid subscription"));
      return Action::kTerminate;
      // No default, we will be warned about unhandled code.
  }
  assert(false);
}

void SubscriptionState::ReceiveMessage(const std::shared_ptr<Logger>& info_log,
                                       std::unique_ptr<MessageDeliverGap> gap) {
  thread_check_.Check();

  ProcessMessage(info_log, *gap);
  // Do not deliver, this is internal message.
}

void SubscriptionState::ReceiveMessage(
    const std::shared_ptr<Logger>& info_log,
    std::unique_ptr<MessageDeliverData> data) {
  thread_check_.Check();

  if (ProcessMessage(info_log, *data)) {
    // Deliver message to the application.
    if (deliver_callback_) {
      deliver_callback_(
          std::unique_ptr<MessageReceivedClient>(new MessageReceivedClient(
              namespace_id_, topic_name_, std::move(data))));
    }
  }
}

bool SubscriptionState::ProcessMessage(const std::shared_ptr<Logger>& info_log,
                                       const MessageDeliver& deliver) {
  thread_check_.Check();

  const auto current = deliver.GetSequenceNumber(),
             previous = deliver.GetPrevSequenceNumber();
  assert(current >= previous);

  if (expected_seqno_ > current ||
      expected_seqno_ < previous ||
      (expected_seqno_ == 0 && previous != 0)) {
    LOG_INFO(info_log,
             "Duplicate message %" PRIu64 "-%" PRIu64
             " on Topic(%s, %s) expected %" PRIu64,
             previous,
             current,
             namespace_id_.c_str(),
             topic_name_.c_str(),
             expected_seqno_);
    return false;
  }

  const char* type_description =
      deliver.GetMessageType() == MessageType::mDeliverGap ? "gap" : "data";
  LOG_DEBUG(info_log,
            "Received %s %" PRIu64 "-%" PRIu64 " on Topic(%s, %s)@%" PRIu64,
            type_description,
            previous,
            current,
            namespace_id_.c_str(),
            topic_name_.c_str(),
            expected_seqno_);

  expected_seqno_ = current + 1;
  return true;
}

void SubscriptionState::Acknowledge(SequenceNumber seqno) {
  thread_check_.Check();

  if (last_acked_seqno_ < seqno) {
    last_acked_seqno_ = seqno;
  }
}

void SubscriptionState::AnnounceStatus(bool subscribed, Status status) {
  thread_check_.Check();

  if (subscription_callback_) {
    // TODO(stupaq) pass heavy stuff by const reference
    subscription_callback_(SubscriptionStatus(tenant_id_,
                                              namespace_id_,
                                              topic_name_,
                                              expected_seqno_,
                                              subscribed,
                                              std::move(status)));
  }
}

////////////////////////////////////////////////////////////////////////////////
/** State of a single subscriber worker, aligned to avoid false sharing. */
class alignas(CACHE_LINE_SIZE) ClientWorkerData {
 public:
  // Noncopyable
  ClientWorkerData(const ClientWorkerData&) = delete;
  ClientWorkerData& operator=(const ClientWorkerData&) = delete;
  // Nonmovable
  ClientWorkerData(ClientWorkerData&&) = delete;
  ClientWorkerData& operator=(ClientWorkerData&&) = delete;

  ClientWorkerData() : next_sub_id_(0) {}

  /** Stream socket used by this worker to talk to the copilot. */
  StreamSocket copilot_socket;
  /** Next subscription ID to be used for new subscription. */
  SubscriptionID next_sub_id_;
  /** All subscriptions served by this worker. */
  std::unordered_map<SubscriptionID, SubscriptionState> subscriptions_;
  /** A mapping from topics to subscription IDs. */
  std::unordered_map<TopicID, SubscriptionID> subscribed_topics_;
};

////////////////////////////////////////////////////////////////////////////////
/** An implementation of the Client API that represents a creation error. */
class ClientCreationError : public Client {
 public:
  explicit ClientCreationError(const Status& creationStatus)
    : creationStatus_(creationStatus)
  {}

  virtual Status Start(SubscribeCallback,
                       MessageReceivedCallback) {
    return creationStatus_;
  }

  virtual PublishStatus Publish(const TenantID tenant_id,
                                const Topic&,
                                const NamespaceID&,
                                const TopicOptions&,
                                const Slice&,
                                PublishCallback,
                                const MsgId message_id) {
    return PublishStatus(creationStatus_, message_id);
  }

  virtual void ListenTopics(const TenantID,
                            const std::vector<SubscriptionRequest>&) {}

  Status Subscribe(SubscriptionParameters,
                   SubscribeCallback,
                   MessageReceivedCallback) override {
    return Status::OK();
  }

  Status Unsubscribe(NamespaceID namespace_id, Topic topic_name) {
    return Status::OK();
  }

  Status Acknowledge(const MessageReceived&) override { return Status::OK(); }

  void SaveSubscriptions(SaveSubscriptionsCallback) override {}

  Status RestoreSubscriptions(std::vector<SubscriptionParameters>*) override {
    return Status::OK();
  }

 private:
  Status creationStatus_;
};

Status Client::Create(ClientOptions options, std::unique_ptr<Client>* client) {
  assert (client);
  std::unique_ptr<ClientImpl> clientImpl;
  auto st = ClientImpl::Create(std::move(options), &clientImpl);
  if (st.ok()) {
    client->reset(clientImpl.release());
  } else {
    client->reset(new ClientCreationError(st));
  }
  return st;
}

Status ClientImpl::Create(ClientOptions options,
                          std::unique_ptr<ClientImpl>* client,
                          bool is_internal) {
  assert (client);

  // Validate arguments.
  if (!options.info_log) {
    options.info_log = std::make_shared<NullLogger>();
  }

#ifndef USE_MQTTMSGLOOP
  std::unique_ptr<MsgLoopBase> msg_loop_(
    new MsgLoop(options.env,
                EnvOptions(),
                0,
                options.num_workers,
                options.info_log,
                "client"));
#else
  MQTTConfiguration* mqtt_config =
    static_cast<MQTTConfiguration*>(options.config.get());
  std::unique_ptr<MsgLoopBase> msg_loop_(
    new MQTTMsgLoop(
      options.env,
      mqtt_config->GetVIP(),
      mqtt_config->GetUsername(),
      mqtt_config->GetAccessToken(),
      mqtt_config->UseSSL(),
      options.info_log,
      &ProxygenMQTTClient::Create));
#endif

  Status st = msg_loop_->Initialize();
  if (!st.ok()) {
    return st;
  }

  client->reset(new ClientImpl(options.env,
                               options.config,
                               options.wake_lock,
                               std::move(msg_loop_),
                               std::move(options.storage),
                               options.info_log,
                               is_internal));
  return Status::OK();
}

ClientImpl::ClientImpl(BaseEnv* env,
                       std::shared_ptr<Configuration> config,
                       std::shared_ptr<WakeLock> wake_lock,
                       std::unique_ptr<MsgLoopBase> msg_loop,
                       std::unique_ptr<SubscriptionStorage> storage,
                       std::shared_ptr<Logger> info_log,
                       bool is_internal)
: env_(env)
, config_(std::move(config))
, wake_lock_(std::move(wake_lock))
, msg_loop_(std::move(msg_loop))
, msg_loop_thread_spawned_(false)
, storage_(std::move(storage))
, info_log_(info_log)
, is_internal_(is_internal)
, publisher_(env, config_, info_log, msg_loop_.get(), &wake_lock_) {
  using std::placeholders::_1;

  LOG_VITAL(info_log_, "Creating Client");

  // Setup callbacks.
  std::map<MessageType, MsgCallbackType> callbacks;
  callbacks[MessageType::mDeliverData] = [this] (std::unique_ptr<Message> msg,
                                             StreamID origin) {
    ProcessDeliverData(std::move(msg), origin);
  };
  callbacks[MessageType::mDeliverGap] = [this] (std::unique_ptr<Message> msg,
                                         StreamID origin) {
    ProcessDeliverGap(std::move(msg), origin);
  };
  callbacks[MessageType::mUnsubscribe] = [this] (std::unique_ptr<Message> msg,
                                              StreamID origin) {
    ProcessUnsubscribe(std::move(msg), origin);
  };
  callbacks[MessageType::mGoodbye] =
      [this](std::unique_ptr<Message> msg, StreamID origin) {
        ProcessGoodbye(std::move(msg), origin);
      };

  // Create sharded state.
  worker_data_.reset(new ClientWorkerData[msg_loop_->GetNumWorkers()]);

  // Initialise stream socket for each worker, each of them is independent.
  HostId copilot;
  Status st = config_->GetCopilot(&copilot);
  assert(st.ok());  // TODO(pja) : handle failures
  for (int i = 0; i < msg_loop_->GetNumWorkers(); ++i) {
    worker_data_[i].copilot_socket = StreamSocket(
        msg_loop_->CreateOutboundStream(copilot.ToClientId(), i));
  }

  msg_loop_->RegisterCallbacks(callbacks);
}

Status ClientImpl::Start(SubscribeCallback subscribe_callback,
                         MessageReceivedCallback receive_callback) {
  subscription_callback_ = std::move(subscribe_callback);
  receive_callback_ = std::move(receive_callback);

  msg_loop_thread_ = env_->StartThread([this]() {
    msg_loop_->Run();
  }, "client");
  msg_loop_thread_spawned_ = true;

  Status st = msg_loop_->WaitUntilRunning();
  if (!st.ok()) {
    return st;
  }
  return Status::OK();
}

ClientImpl::~ClientImpl() {
  // Stop the event loop. May block.
  msg_loop_->Stop();

  if (msg_loop_thread_spawned_) {
    // Wait for thread to join.
    env_->WaitForJoin(msg_loop_thread_);
  }
}

PublishStatus ClientImpl::Publish(const TenantID tenant_id,
                                  const Topic& name,
                                  const NamespaceID& namespace_id,
                                  const TopicOptions& options,
                                  const Slice& data,
                                  PublishCallback callback,
                                  const MsgId message_id) {
  if (!is_internal_) {
    if (tenant_id <= 100 && tenant_id != GuestTenant) {
      return PublishStatus(
        Status::InvalidArgument("TenantID must be greater than 100."),
        message_id);
    }

    if (IsReserved(namespace_id)) {
      return PublishStatus(
        Status::InvalidArgument("NamespaceID is reserved for internal usage."),
        message_id);
    }
  }
  return publisher_.Publish(tenant_id,
                            namespace_id,
                            name,
                            options,
                            data,
                            std::move(callback),
                            message_id);
}

void ClientImpl::ListenTopics(
    const TenantID tenant_id,
    const std::vector<SubscriptionRequest>& requests) {
  for (const auto& req : requests) {
    Status st;
    if (req.subscribe) {
      st = Client::Subscribe(tenant_id,
                             std::move(req.namespace_id),
                             std::move(req.topic_name),
                             req.start,
                             subscription_callback_,
                             receive_callback_);
    } else {
      st = Unsubscribe(std::move(req.namespace_id), std::move(req.topic_name));
    }
    if (!st.ok() && subscription_callback_) {
      SubscriptionStatus error_msg;
      error_msg.tenant_id = tenant_id;
      error_msg.namespace_id = std::move(req.namespace_id);
      error_msg.topic_name = std::move(req.topic_name);
      error_msg.status = std::move(st);
      subscription_callback_(std::move(error_msg));
    }
  }
}

Status ClientImpl::Subscribe(SubscriptionParameters parameters,
                             SubscribeCallback subscription_callback,
                             MessageReceivedCallback deliver_callback) {
  const auto worker_id = GetWorkerForTopic(parameters.topic_name);
  // Create an object that manages state of the subscription.
  auto moved_sub_state = folly::makeMoveWrapper(SubscriptionState(
      std::move(parameters), subscription_callback, deliver_callback));
  // Send command to responsible worker.
  auto action = [this, moved_sub_state]() mutable {
    StartSubscription(moved_sub_state.move());
  };
  return msg_loop_->SendCommand(
      std::unique_ptr<Command>(new ExecuteCommand(std::move(action))),
      worker_id);
}

Status ClientImpl::Unsubscribe(NamespaceID namespace_id, Topic topic_name) {
  const auto worker_id = GetWorkerForTopic(topic_name);
  // Send command to responsible worker.
  auto moved_namespace_id = folly::makeMoveWrapper(std::move(namespace_id));
  auto moved_topic_name = folly::makeMoveWrapper(std::move(topic_name));
  auto action = [this, moved_namespace_id, moved_topic_name]() mutable {
    TerminateSubscription(moved_namespace_id.move(), moved_topic_name.move());
  };
  return msg_loop_->SendCommand(
      std::unique_ptr<Command>(new ExecuteCommand(std::move(action))),
      worker_id);
}

Status ClientImpl::Acknowledge(const MessageReceived& message) {
  // Find the right worker to send command to.
  auto moved_topic_id = folly::makeMoveWrapper(TopicID(
      message.GetNamespaceId().ToString(), message.GetTopicName().ToString()));
  int worker_id = GetWorkerForTopic(moved_topic_id->topic_name);
  // Acknowledge the message in subscription state.
  SequenceNumber acked_seqno = message.GetSequenceNumber();
  auto action = [this, worker_id, moved_topic_id, acked_seqno]() mutable {
    auto& worker_data = worker_data_[worker_id];

    SubscriptionID sub_id;
    TopicID topic_id(moved_topic_id.move());
    {  // Find corresponding subscription ID.
      auto it = worker_data.subscribed_topics_.find(topic_id);
      if (it == worker_data.subscribed_topics_.end()) {
        LOG_WARN(info_log_,
                 "Cannot acknowledge missing subscription on Topic(%s, %s)",
                 topic_id.namespace_id.c_str(),
                 topic_id.topic_name.c_str());
        return;
      }
      sub_id = it->second;
    }

    // Find corresponding subscription object.
    auto it = worker_data.subscriptions_.find(sub_id);
    if (it == worker_data.subscriptions_.end()) {
      LOG_ERROR(info_log_,
                "Cannot acknowledge missing subscription ID (%u)",
                sub_id);
      assert(false);
      return;
    }

    // Record acknowledgement in the state.
    it->second.Acknowledge(acked_seqno);
  };
  return msg_loop_->SendCommand(
      std::unique_ptr<Command>(new ExecuteCommand(std::move(action))),
      worker_id);
}

void ClientImpl::SaveSubscriptions(SaveSubscriptionsCallback save_callback) {
  if (!storage_) {
    save_callback(Status::NotInitialized());
    return;
  }

  std::shared_ptr<SubscriptionStorage::Snapshot> snapshot;
  Status st = storage_->CreateSnapshot(msg_loop_->GetNumWorkers(), &snapshot);
  if (!st.ok()) {
    LOG_ERROR(info_log_,
              "Failed to create snapshot to save subscriptions: %s",
              st.ToString().c_str());
    save_callback(std::move(st));
    return;
  }

  // For each worker we attemp to append entries for all subscriptions.
  auto map = [this, snapshot](int worker_id) {
    const auto& worker_data = worker_data_[worker_id];

    for (const auto& entry : worker_data.subscriptions_) {
      const auto sub_state = &entry.second;
      SequenceNumber start_seqno = sub_state->GetLastAcknowledged();
      // Subscription storage stores parameters of subscribe requests that shall
      // be reissued, therefore we must persiste the next sequence number.
      if (start_seqno > 0) {
        ++start_seqno;
      }
      Status status = snapshot->Append(worker_id,
                                       sub_state->GetTenant(),
                                       sub_state->GetNamespace(),
                                       sub_state->GetTopicName(),
                                       start_seqno);
      if (!status.ok()) {
        return status;
      }
    }
    return Status::OK();
  };

  // Once all workers are done, we commit the snapshot and call the callback if
  // necessary.
  auto reduce = [save_callback, snapshot](std::vector<Status> statuses) {
    for (auto& status : statuses) {
      if (!status.ok()) {
        save_callback(std::move(status));
        return;
      }
    }
    Status status = snapshot->Commit();
    save_callback(std::move(status));
  };

  // Fan out commands to all workers.
  st = msg_loop_->Gather(std::move(map), std::move(reduce));
  if (!st.ok()) {
    LOG_ERROR(info_log_,
              "Failed to send snapshot command to all workers: %s",
              st.ToString().c_str());
    save_callback(std::move(st));
    return;
  }
}

Status ClientImpl::RestoreSubscriptions(
    std::vector<SubscriptionParameters>* subscriptions) {
  if (!storage_) {
    return Status::NotInitialized();
  }

  return storage_->RestoreSubscriptions(subscriptions);
}

Statistics ClientImpl::GetStatisticsSync() const {
  return msg_loop_->GetStatisticsSync();
}

int ClientImpl::GetWorkerForTopic(const Topic& name) const {
  return static_cast<int>(MurmurHash2<std::string>()(name) %
                          msg_loop_->GetNumWorkers());
}

void ClientImpl::StartSubscription(SubscriptionState sub_state_val) {
  const auto worker_id = msg_loop_->GetThreadWorkerIndex();
  auto& worker_data = worker_data_[worker_id];

  TopicID topic_id(sub_state_val.GetNamespace(), sub_state_val.GetTopicName());
  {  // Kill any existing subscription on the topic.
    if (worker_data.subscribed_topics_.count(topic_id) > 0) {
      TerminateSubscription(sub_state_val.GetNamespace(),
                            sub_state_val.GetTopicName());
    }
  }

  SubscriptionID sub_id;
  SubscriptionState* sub_state;
  {  // Assign subscription ID and store the subscription.
    const SubscriptionID started_search_at = worker_data.next_sub_id_;
    do {
      // This loop will do up to number of existing subscriptions iterations.
      sub_id = worker_data.next_sub_id_++;
      auto it = worker_data.subscriptions_.find(sub_id);
      if (it == worker_data.subscriptions_.end()) {
        auto emplace_result = worker_data.subscriptions_.emplace(
            sub_id, std::move(sub_state_val));
        assert(emplace_result.second);
        sub_state = &emplace_result.first->second;
        break;
      }
    } while (worker_data.next_sub_id_ != started_search_at);
    // If we've made a full cycle, it means we've run out of subscription IDs.
    if (worker_data.next_sub_id_ == started_search_at) {
      // Apparently we have about 4 TB of ram or we're leaking IDs.
      LOG_FATAL(info_log_,
                "Failed to allocate ID for new subscription on Topic(%s, %s)",
                sub_state_val.GetNamespace().c_str(),
                sub_state_val.GetTopicName().c_str());
      assert(false);
      return;
    }
  }
  sub_state->AssignID(info_log_, sub_id);

  {  // Replace existing subscription on the topic.
    // Insert entry for new subscription.
    auto emplace_result =
        worker_data.subscribed_topics_.emplace(std::move(topic_id), sub_id);
    assert(emplace_result.second);
  }

  // Prepare first subscription request.
  MessageSubscribe message(sub_state->GetTenant(),
                           sub_state->GetNamespace(),
                           sub_state->GetTopicName(),
                           sub_state->GetExpected(),
                           sub_id);

  // Send message.
  wake_lock_.AcquireForSending();
  Status st =
      msg_loop_->SendRequest(message, &worker_data.copilot_socket, worker_id);
  // TODO(stupaq) handle failures
  assert(st.ok());
}

void ClientImpl::TerminateSubscription(NamespaceID namespace_id,
                                       Topic topic_name) {
  const auto worker_id = msg_loop_->GetThreadWorkerIndex();
  auto& worker_data = worker_data_[worker_id];

  const TopicID topic_id(std::move(namespace_id), std::move(topic_name));
  SubscriptionID sub_id;
  {  // Remove entry for the topic.
    auto it = worker_data.subscribed_topics_.find(topic_id);
    if (it == worker_data.subscribed_topics_.end()) {
      LOG_WARN(info_log_,
               "Cannot remove missing subscription on Topic(%s, %s)",
               topic_id.namespace_id.c_str(),
               topic_id.topic_name.c_str());
      return;
    }
    sub_id = it->second;
    worker_data.subscribed_topics_.erase(it);
  }

  // Remove subscription state and prepare unsubscribe message.
  auto it = worker_data.subscriptions_.find(sub_id);
  if (it == worker_data.subscriptions_.end()) {
    LOG_ERROR(info_log_, "Cannot remove missing subscription ID (%u)", sub_id);
    assert(false);
    return;
  }
  SubscriptionState sub_state(std::move(it->second));
  worker_data.subscriptions_.erase(it);

  // Prepare unsubscription request.
  MessageUnsubscribe message(sub_state.GetTenant(),
                             sub_id,
                             MessageUnsubscribe::Reason::kRequested);

  // Update subscription state, which will announce subscription status to the
  // application.
  sub_state.ProcessMessage(info_log_, message);

  // Send message.
  wake_lock_.AcquireForSending();
  Status st =
      msg_loop_->SendRequest(message, &worker_data.copilot_socket, worker_id);
  if (!st.ok()) {
    // No harm done if we fail to send unsubscribe request, since we've marked
    // subscription as removed, we will respond with appropriate unsubscribe
    // request to every message on the terminated subscription.
    LOG_WARN(info_log_,
             "Failed to send unsubscribe response for ID (%u)",
             sub_id);
  }
}

bool ClientImpl::IsNotCopilot(const ClientWorkerData& worker_data,
                              StreamID origin) {
  if (worker_data.copilot_socket.GetStreamID() != origin) {
    LOG_ERROR(info_log_,
              "Incorrect message stream: (%llu) expected: (%llu)",
              origin,
              worker_data.copilot_socket.GetStreamID());
    assert(false);
    return true;
  }
  return false;
}

SubscriptionState* ClientImpl::FindOrSendUnsubscribe(TenantID tenant_id,
                                                     SubscriptionID sub_id) {
  // Get worker data that all topics in the message are assigned to.
  const auto worker_id = msg_loop_->GetThreadWorkerIndex();
  auto& worker_data = worker_data_[worker_id];

  {  // Attemp to find corresponding subscription.
    auto it = worker_data.subscriptions_.find(sub_id);
    if (it != worker_data.subscriptions_.end()) {
      return &it->second;
    }
  }

  LOG_WARN(info_log_,
           "Cannot find subscription ID (%u), sending unsubscribe",
           sub_id);

  // Prepare unsubscription request.
  MessageUnsubscribe message(tenant_id,
                             sub_id,
                             MessageUnsubscribe::Reason::kRequested);

  // Send message.
  wake_lock_.AcquireForSending();
  Status st =
      msg_loop_->SendRequest(message, &worker_data.copilot_socket, worker_id);
  if (!st.ok()) {
    // No harm done if we fail to send unsubscribe request, the subscription
    // does not really exist.
    LOG_WARN(info_log_,
             "Failed to send unsubscribe response ID (%u)",
             sub_id);
  }

  return nullptr;
}

void ClientImpl::ProcessDeliverData(std::unique_ptr<Message> msg,
                                    StreamID origin) {
  std::unique_ptr<MessageDeliverData> data(
      static_cast<MessageDeliverData*>(msg.release()));

  wake_lock_.AcquireForReceiving();
  // Get worker data that this topic is assigned to.
  const auto worker_id = msg_loop_->GetThreadWorkerIndex();
  auto& worker_data = worker_data_[worker_id];

  // Check that message arrived on correct stream.
  if (IsNotCopilot(worker_data, origin)) {
    return;
  }

  // Find the right subscription and deliver the message to it.
  auto sub_state = FindOrSendUnsubscribe(data->GetTenantID(), data->GetSubID());
  if (sub_state) {
    sub_state->ReceiveMessage(info_log_, std::move(data));
  }
}

void ClientImpl::ProcessDeliverGap(std::unique_ptr<Message> msg,
                                   StreamID origin) {
  std::unique_ptr<MessageDeliverGap> gap(
      static_cast<MessageDeliverGap*>(msg.release()));

  wake_lock_.AcquireForReceiving();
  // Get worker data that this topic is assigned to.
  const auto worker_id = msg_loop_->GetThreadWorkerIndex();
  auto& worker_data = worker_data_[worker_id];

  // Check that message arrived on correct stream.
  if (IsNotCopilot(worker_data, origin)) {
    return;
  }

  // Find the right subscription and deliver the message to it.
  auto sub_state = FindOrSendUnsubscribe(gap->GetTenantID(), gap->GetSubID());
  if (sub_state) {
    sub_state->ReceiveMessage(info_log_, std::move(gap));
  }
}

void ClientImpl::ProcessUnsubscribe(std::unique_ptr<Message> msg,
                                    StreamID origin) {
  std::unique_ptr<MessageUnsubscribe> unsubscribe(
      static_cast<MessageUnsubscribe*>(msg.release()));

  wake_lock_.AcquireForReceiving();
  // Get worker data that all topics in the message are assigned to.
  const auto worker_id = msg_loop_->GetThreadWorkerIndex();
  auto& worker_data = worker_data_[worker_id];

  // Check that message arrived on correct stream.
  if (IsNotCopilot(worker_data, origin)) {
    return;
  }

  const SubscriptionID sub_id = unsubscribe->GetSubID();
  // Find the right subscription and deliver the message to it.
  auto it = worker_data.subscriptions_.find(sub_id);
  if (it == worker_data.subscriptions_.end()) {
    LOG_WARN(info_log_,
             "Received unsibscribe with unrecognised ID(%u)",
             sub_id);
    return;
  }
  SubscriptionState* sub_state = &it->second;

  auto action = sub_state->ProcessMessage(info_log_, *unsubscribe);
  switch (action) {
    case SubscriptionState::Action::kTerminate:
      worker_data.subscriptions_.erase(it);
      break;
    case SubscriptionState::Action::kResubscribe: {
      MessageSubscribe message(sub_state->GetTenant(),
                               sub_state->GetNamespace(),
                               sub_state->GetTopicName(),
                               sub_state->GetExpected(),
                               sub_id);
      Status st = msg_loop_->SendRequest(message,
                                         &worker_data.copilot_socket,
                                         worker_id);
      // TODO(stupaq) handle failure
      assert(st.ok());
    } break;
      // No default, we will be warned about unhandled code.
  }
}

void ClientImpl::ProcessGoodbye(std::unique_ptr<Message> msg, StreamID origin) {
  const auto worker_id = msg_loop_->GetThreadWorkerIndex();
  auto& worker_data = worker_data_[worker_id];

  // Check that message arrived on correct stream.
  if (worker_data.copilot_socket.GetStreamID() != origin) {
    // It might still be addressed to the publisher.
    publisher_.ProcessGoodbye(std::move(msg), origin);
    return;
  }

  // Get the copilot's address.
  HostId copilot;
  Status st = config_->GetCopilot(&copilot);
  assert(st.ok());  // TODO(pja) : handle failures

  // And create socket to it.
  worker_data.copilot_socket =
      msg_loop_->CreateOutboundStream(copilot.ToClientId(), worker_id);

  LOG_INFO(info_log_,
           "Reconnected to %s on stream %llu",
           copilot.ToString().c_str(),
           worker_data.copilot_socket.GetStreamID());

  // Reissue all subscriptions.
  for (auto& entry : worker_data.subscriptions_) {
    SubscriptionID sub_id = entry.first;
    SubscriptionState* sub_state = &entry.second;

    LOG_INFO(info_log_,
             "Reissued subscription ID (%u) on Topic(%s, %s)@%" PRIu64,
             sub_id,
             sub_state->GetNamespace().c_str(),
             sub_state->GetTopicName().c_str(),
             sub_state->GetExpected());

    // Prepare subscription request.
    MessageSubscribe message(sub_state->GetTenant(),
                             sub_state->GetNamespace(),
                             sub_state->GetTopicName(),
                             sub_state->GetExpected(),
                             sub_id);

    // Send message.
    st =
        msg_loop_->SendRequest(message, &worker_data.copilot_socket, worker_id);
    // TODO(stupaq) handle failure
    assert(st.ok());
  }
}

}  // namespace rocketspeed
