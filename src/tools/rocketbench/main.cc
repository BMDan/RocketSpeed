// Copyright (c) 2014, Facebook, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.
//
#include <assert.h>
#include <gflags/gflags.h>
#include <signal.h>
#include <unistd.h>
#include <future>
#include <random>
#include <string>
#include <thread>
#include <vector>
#include "include/RocketSpeed.h"
#include "include/Types.h"
#include "src/port/port_posix.h"
#include "src/messages/messages.h"
#include "src/util/auto_roll_logger.h"
#include "src/tools/rocketbench/random_distribution.h"

// This tool can behave as a standalone producer, a standalone
// consumer or both a producer and a consumer.
DEFINE_bool(start_producer, true, "starts the producer");
DEFINE_bool(start_consumer, false, "starts the consumer");

DEFINE_int32(num_threads, 4, "number of threads");
DEFINE_string(pilot_hostname, "localhost", "hostname of pilot");
DEFINE_string(copilot_hostname, "localhost", "hostname of copilot");
DEFINE_int32(pilot_port, 58600, "port number of pilot");
DEFINE_int32(copilot_port, 58700, "port number of copilot");
DEFINE_int32(producer_port, 58800, "port number of producer");
DEFINE_int32(consumer_port, 58800, "port number of consumer");
DEFINE_int32(message_size, 100, "message size (bytes)");
DEFINE_uint64(num_topics, 1000, "number of topics");
DEFINE_int64(num_messages, 10000, "number of messages to send");
DEFINE_int64(message_rate, 100000, "messages per second (0 = unlimited)");
DEFINE_bool(await_ack, true, "wait for and include acks in times");
DEFINE_bool(logging, true, "enable/disable logging");
DEFINE_bool(report, true, "report results to stdout");
DEFINE_string(topics_distribution, "uniform",
"uniform, normal, poisson, fixed");
DEFINE_int64(topics_mean, 0,
"Mean for Normal and Poisson topic distributions (rounded to nearest int64)");
DEFINE_int64(topics_stddev, 0,
"Standard Deviation for Normal topic distribution (rounded to nearest int64)");

std::shared_ptr<rocketspeed::Logger> info_log;

struct Result {
  bool succeeded;
};

static const Result failed = { false, };

namespace rocketspeed {

Result ProducerWorker(int64_t num_messages, Client* producer) {
  Env* env = Env::Default();

  // Random number generator.
  std::mt19937_64 rng;
  std::unique_ptr<rocketspeed::RandomDistributionBase>
    distr(GetDistributionByName(FLAGS_topics_distribution,
    0, FLAGS_num_topics - 1, FLAGS_topics_mean, FLAGS_topics_stddev));

  // Generate some dummy data.
  std::vector<char> data(FLAGS_message_size);
  const char* data_message = "RocketSpeed ";
  for (int i = 0; i < FLAGS_message_size; ++i) {
    data[i] = data_message[i % strlen(data_message)];
  }
  Slice payload(data.data(), data.size());

  Log(InfoLogLevel::INFO_LEVEL, info_log, "Starting message loop");
  info_log->Flush();

  // Calculate message rate for this worker.
  int64_t rate = FLAGS_message_rate / FLAGS_num_threads;

  // Number of messages we should have sent in 10ms
  int64_t rate_check = std::max<int64_t>(1, rate / 100);

  auto start = std::chrono::steady_clock::now();
  for (int64_t i = 0; i < num_messages; ++i) {
    // Create random topic name
    char topic_name[64];
    if (distr.get() != nullptr) {
      snprintf(topic_name, sizeof(topic_name),
               "benchmark.%lu",
               distr->generateRandomInt());
    } else {               // distribution is "fixed"
      snprintf(topic_name, sizeof(topic_name),
               "benchmark.%lu",
               i % 100);   // 100 messages per topic
    }

    NamespaceID namespace_id = 101;

    // Random topic options
    TopicOptions topic_options;
    switch (rng() % Retention::Total) {
      case 0:
        topic_options.retention = Retention::OneHour;
        break;
      case 1:
        topic_options.retention = Retention::OneDay;
        break;
      case 2:
        topic_options.retention = Retention::OneWeek;
        break;
    }

    // Add ID and timestamp to message ID.
    static std::atomic<uint64_t> message_index;
    snprintf(data.data(), data.size(),
             "%lu %lu", message_index++, env->NowMicros());

    // Send the message
    PublishStatus ps = producer->Publish(topic_name,
                                         namespace_id,
                                         topic_options,
                                         payload);

    if (!ps.status.ok()) {
      Log(InfoLogLevel::WARN_LEVEL, info_log,
        "Failed to send message number %lu (%s)",
        i, ps.status.ToString().c_str());
      info_log->Flush();
      return failed;
    }

    if (FLAGS_message_rate != 0 &&
        (i + 1) % rate_check == 0) {
      // Check if we are sending messages too fast, and if so, sleep.
      int64_t have_sent = i;

      // Time since we started sending messages.
      auto expired = std::chrono::steady_clock::now() - start;

      // Time that should have expired if we were sending at the desired rate.
      auto expected = std::chrono::microseconds(1000000 * have_sent / rate);

      // If we are ahead of schedule then sleep for the difference.
      if (expected > expired) {
        std::this_thread::sleep_for(expected - expired);
      }
    }
  }
  return Result { true };
}

/*
 ** Produce messages
 */
int DoProduce(Client* producer,
              rocketspeed::port::Semaphore* all_ack_messages_received,
              int64_t* ack_messages_received,
              std::chrono::time_point<std::chrono::steady_clock>*
                                                     last_ack_message){
  // Distribute total number of messages among them.
  std::vector<std::future<Result>> futures;
  int64_t total_messages = FLAGS_num_messages;
  for (int32_t remaining = FLAGS_num_threads; remaining; --remaining) {
    int64_t num_messages = total_messages / remaining;
    futures.emplace_back(std::async(std::launch::async,
                                    &rocketspeed::ProducerWorker,
                                    num_messages,
                                    producer));
    total_messages -= num_messages;
  }
  assert(total_messages == 0);

  // Join all the threads to finish production.
  int ret = 0;
  for (int i = 0; i < static_cast<int>(futures.size()); ++i) {
    Result result = futures[i].get();
    if (!result.succeeded) {
      if (FLAGS_report) {
        printf("Thread %d failed to send all messages\n", i);
      }
      ret = 1;
    }
  }

  if (FLAGS_await_ack) {
    // Wait for the all_ack_messages_received semaphore to be posted.
    // Keep waiting as long as a message was received in the last 5 seconds.
    auto timeout = std::chrono::seconds(5);
    do {
      all_ack_messages_received->TimedWait(timeout);
    } while (*ack_messages_received != FLAGS_num_messages &&
             std::chrono::steady_clock::now() - *last_ack_message < timeout);
  }
  return ret;
}

/*
 ** Receive messages
 */
int DoConsume(Client* consumer,
              rocketspeed::port::Semaphore* all_messages_received,
              int64_t* messages_received,
              std::chrono::time_point<std::chrono::steady_clock>*
                                                     last_data_message) {

  SequenceNumber start = 1;   // start sequence number
  NamespaceID nsid = 101;

  // subscribe 10K topics at a time
  unsigned int batch_size = 10240;

  std::vector<SubscriptionPair> topics;
  topics.reserve(batch_size);

  // create all subscriptions from seqno 1
  SubscriptionPair pair(start, "", nsid);
  for (uint64_t i = 0; i < FLAGS_num_topics; i++) {
    pair.topic_name.assign("benchmark." + std::to_string(i));
    topics.push_back(pair);
    if (i % batch_size == batch_size - 1) {
      // send a subscription request
      consumer->ListenTopics(topics, TopicOptions());
      topics.clear();
    }
  }

  // subscribe to all remaining topics
  if (topics.size() != 0) {
    consumer->ListenTopics(topics, TopicOptions());
  }

  // Wait for the all_messages_received semaphore to be posted.
  // Keep waiting as long as a message was received in the last 5 seconds.
  auto timeout = std::chrono::seconds(5);
  do {
    all_messages_received->TimedWait(timeout);
  } while (*messages_received != FLAGS_num_messages &&
           std::chrono::steady_clock::now() - *last_data_message < timeout);

  return 0;
}

}  // namespace rocketspeed

int main(int argc, char** argv) {
  GFLAGS::ParseCommandLineFlags(&argc, &argv, true);

  // Ignore SIGPIPE, we'll just handle the EPIPE returned by write.
  signal(SIGPIPE, SIG_IGN);

  // Validate flags
  if (FLAGS_num_threads <= 0) {
    fprintf(stderr, "num_threads must be greater than 0.\n");
    return 1;
  }

  if (FLAGS_pilot_port > 65535 || FLAGS_pilot_port < 0) {
    fprintf(stderr, "pilot_port must be 0-65535.\n");
    return 1;
  }

  if (FLAGS_copilot_port > 65535 || FLAGS_copilot_port < 0) {
    fprintf(stderr, "copilot_port must be 0-65535.\n");
    return 1;
  }

  if (FLAGS_producer_port > 65535 || FLAGS_producer_port < 0) {
    fprintf(stderr, "producer_port must be 0-65535.\n");
    return 1;
  }

  if (FLAGS_consumer_port > 65535 || FLAGS_consumer_port < 0) {
    fprintf(stderr, "consumer_port must be 0-65535.\n");
    return 1;
  }

  if (FLAGS_message_size <= 0 || FLAGS_message_size > 1024*1024) {
    fprintf(stderr, "message_size must be 0-1MB.\n");
    return 1;
  }

  if (FLAGS_num_topics <= 0) {
    fprintf(stderr, "num_topics must be greater than 0.\n");
    return 1;
  }

  if (FLAGS_num_messages <= 0) {
    fprintf(stderr, "num_messages must be greater than 0.\n");
    return 1;
  }
  if (!FLAGS_start_consumer && !FLAGS_start_producer) {
    fprintf(stderr, "You must specify at least one --start_producer "
            "or --start_consumer\n");
    return 1;
  }

  // Create logger
  if (FLAGS_logging) {
    if (!rocketspeed::CreateLoggerFromOptions(rocketspeed::Env::Default(),
                                              "",
                                              "LOG.rocketbench",
                                              0,
                                              0,
                                              rocketspeed::INFO_LEVEL,
                                              &info_log).ok()) {
      fprintf(stderr, "Error creating logger, aborting.\n");
      return 1;
    }
  } else {
    info_log = std::make_shared<rocketspeed::NullLogger>();
  }

  // Configuration for RocketSpeed.
  rocketspeed::HostId pilot(FLAGS_pilot_hostname, FLAGS_pilot_port);
  rocketspeed::HostId copilot(FLAGS_copilot_hostname, FLAGS_copilot_port);

  // Create two configs: one for the prpducer and one for the consumer
  std::unique_ptr<rocketspeed::Configuration> pconfig(
    rocketspeed::Configuration::Create(
      std::vector<rocketspeed::HostId>{ pilot },
      std::vector<rocketspeed::HostId>{ copilot },
      rocketspeed::Tenant(102),
      FLAGS_producer_port));
  std::unique_ptr<rocketspeed::Configuration> cconfig(
    rocketspeed::Configuration::Create(
      std::vector<rocketspeed::HostId>{ pilot },
      std::vector<rocketspeed::HostId>{ copilot },
      rocketspeed::Tenant(102),
      FLAGS_consumer_port));

  // Start/end time for benchmark.
  std::chrono::time_point<std::chrono::steady_clock> start, end;

  // Semaphore to signal when all messages have been ack'd
  rocketspeed::port::Semaphore all_ack_messages_received;

  // Time last ack/data message was received.
  std::chrono::time_point<std::chrono::steady_clock> last_ack_message;
  std::chrono::time_point<std::chrono::steady_clock> last_data_message;

  // Create callback for publish acks.
  int64_t ack_messages_received = 0;
  auto publish_callback = [&] (rocketspeed::ResultStatus rs) {
    ++ack_messages_received;

    if (FLAGS_await_ack) {
      // This may be the last ack we receive, so set end to the time now.
      end = std::chrono::steady_clock::now();
      last_ack_message = std::chrono::steady_clock::now();

      // If we've received all messages, let the main thread know to finish up.
      if (ack_messages_received == FLAGS_num_messages) {
        all_ack_messages_received.Post();
      }
    }
  };

  // Semaphore to signal when all data messages have been received
  rocketspeed::port::Semaphore all_messages_received;

  // Data structure to track received message indices and latencies (microsecs).
  uint64_t latency_max = static_cast<uint64_t>(-1);
  std::vector<uint64_t> latencies(FLAGS_num_messages, latency_max);

  // Create callback for processing messages received
  rocketspeed::Env* env = rocketspeed::Env::Default();
  int64_t messages_received = 0;
  auto receive_callback = [&]
    (std::unique_ptr<rocketspeed::MessageReceived> rs) {
    uint64_t now = env->NowMicros();
    ++messages_received;
    last_data_message = std::chrono::steady_clock::now();

    // Parse message data to get received index.
    rocketspeed::Slice data = rs->GetContents();
    uint64_t message_index, send_time;
    std::sscanf(data.data(), "%lu %lu", &message_index, &send_time);
    if (message_index < latencies.size()) {
      if (latencies[message_index] == latency_max) {
        latencies[message_index] = now - send_time;
      } else {
        Log(rocketspeed::InfoLogLevel::WARN_LEVEL, info_log,
            "Received duplicate message index (%lu)",
            message_index);
      }
    } else {
      Log(rocketspeed::InfoLogLevel::WARN_LEVEL, info_log,
          "Received out of bounds message index (%lu)",
          message_index);
    }

    // If we've received all messages, let the main thread know to finish up.
    if (messages_received == FLAGS_num_messages) {
      all_messages_received.Post();
    }
  };

  rocketspeed::Client* producer = nullptr;
  rocketspeed::Client* consumer = nullptr;

  // If the producer port and the consumer port are the same, then we
  // use a single client object. This allows the producer and the
  // consumer to share the same connections to the Cloud.
  if (FLAGS_start_producer && FLAGS_start_consumer &&
      FLAGS_producer_port == FLAGS_consumer_port) {
    if (!rocketspeed::Client::Open(pconfig.get(), publish_callback,
                                   nullptr, receive_callback,
                                   &producer).ok()) {
      Log(rocketspeed::InfoLogLevel::WARN_LEVEL, info_log,
          "Failed to connect to RocketSpeed");
      info_log->Flush();
      return 1;
    }
    consumer = producer;
  } else {
    if (FLAGS_start_producer &&
        !rocketspeed::Client::Open(pconfig.get(), publish_callback,
                                   nullptr, nullptr,
                                   &producer).ok()) {
      Log(rocketspeed::InfoLogLevel::WARN_LEVEL, info_log,
          "Failed to connect to RocketSpeed: producer ");
      info_log->Flush();
      return 1;
    }
    if (FLAGS_start_consumer &&
        !rocketspeed::Client::Open(cconfig.get(), nullptr,
                                   nullptr, receive_callback,
                                   &consumer).ok()) {
      Log(rocketspeed::InfoLogLevel::WARN_LEVEL, info_log,
          "Failed to connect to RocketSpeed: consumer ");
      info_log->Flush();
      return 1;
    }
  }
  // Start the clock.
  start = std::chrono::steady_clock::now();

  std::future<int> producer_ret;
  std::future<int> consumer_ret;
  if (FLAGS_start_producer) {
    producer_ret = std::async(std::launch::async,
                              &rocketspeed::DoProduce,
                              producer,
                              &all_ack_messages_received,
                              &ack_messages_received,
                              &last_ack_message);
  }
  if (FLAGS_start_consumer) {
    consumer_ret = std::async(std::launch::async,
                              &rocketspeed::DoConsume,
                              consumer,
                              &all_messages_received,
                              &messages_received,
                              &last_data_message);
  }

  int ret = 0;
  if (FLAGS_start_producer) {
    printf("Messages sent, awaiting acks...");
    fflush(stdout);

    ret = producer_ret.get();

    if (ack_messages_received == FLAGS_num_messages) {
      printf(" done\n");
    } else {
      printf(" time out\n");
    }
  }
  if (FLAGS_start_consumer) {
    printf("Waiting for Messages to be received...");
    fflush(stdout);

    ret = consumer_ret.get();

    if (messages_received == FLAGS_num_messages) {
      printf(" done\n");
    } else {
      printf(" time out\n");
    }
  }

  end = std::chrono::steady_clock::now();

  // Calculate total time.
  auto total_time = std::chrono::duration_cast<std::chrono::milliseconds>(
    end - start);

  if (ret == 0 && FLAGS_report) {
    uint32_t total_ms = static_cast<uint32_t>(total_time.count());
    if (total_ms == 0) {
      // To avoid divide-by-zero on near-instant benchmarks.
      total_ms = 1;
    }

    uint32_t msg_per_sec = 1000 *
                           FLAGS_num_messages /
                           total_ms;
    uint32_t bytes_per_sec = 1000 *
                             FLAGS_num_messages *
                             FLAGS_message_size /
                             total_ms;

    printf("\n");
    printf("Results\n");
    printf("%ld messages sent\n", FLAGS_num_messages);
    printf("%ld messages sends acked\n", ack_messages_received);
    printf("%ld messages received\n", messages_received);

    printf("\n");
    printf("Throughput\n");
    printf("%u messages/s\n", msg_per_sec);
    printf("%.2lf MB/s\n", bytes_per_sec * 1e-6);

    std::sort(latencies.begin(), latencies.end());
    printf("\n");
    printf("Latency (microseconds)\n");
    printf("p50 = %lu\n", latencies[FLAGS_num_messages * 0.50]);
    printf("p75 = %lu\n", latencies[FLAGS_num_messages * 0.75]);
    printf("p90 = %lu\n", latencies[FLAGS_num_messages * 0.90]);
    printf("p95 = %lu\n", latencies[FLAGS_num_messages * 0.95]);
    printf("p99 = %lu\n", latencies[FLAGS_num_messages * 0.99]);
  }

  delete producer;
  if (producer != consumer) {
    delete consumer;
  }
  return ret;
}
