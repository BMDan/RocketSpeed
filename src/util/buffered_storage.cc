// Copyright (c) 2015, Facebook, Inc.  All rights reserved.
// This source code is licensed under the BSD-style license found in the
// LICENSE file in the root directory of this source tree. An additional grant
// of patent rights can be found in the PATENTS file in the same directory.
//
#include "src/util/buffered_storage.h"

namespace rocketspeed {

class BufferedAsyncLogReader : public AsyncLogReader {
 public:
  BufferedAsyncLogReader(
    BufferedLogStorage* storage,
    std::function<bool(LogRecord&)> record_cb,
    std::function<bool(const GapRecord&)> gap_cb,
    std::unique_ptr<AsyncLogReader> reader);

  ~BufferedAsyncLogReader() final;

  Status Open(LogID id,
              SequenceNumber startPoint,
              SequenceNumber endPoint) final;

  Status Close(LogID id) final;

 private:
  std::unique_ptr<AsyncLogReader> reader_;
  BufferedLogStorage* storage_;
};

Status BufferedLogStorage::Create(Env* env,
                                  std::shared_ptr<Logger> info_log,
                                  std::unique_ptr<LogStorage> wrapped_storage,
                                  MsgLoop* msg_loop,
                                  size_t max_batch_entries,
                                  size_t max_batch_bytes,
                                  std::chrono::microseconds max_batch_latency,
                                  LogStorage** storage) {
  *storage = new BufferedLogStorage(env,
                                    std::move(info_log),
                                    std::move(wrapped_storage),
                                    msg_loop,
                                    max_batch_entries,
                                    max_batch_bytes,
                                    max_batch_latency);
  return Status::OK();
}

BufferedLogStorage::~BufferedLogStorage() {

}

Status BufferedLogStorage::AppendAsync(LogID id,
                                       const Slice& data,
                                       AppendCallback callback) {
  return Status::OK();
}

Status BufferedLogStorage::FindTimeAsync(
  LogID id,
  std::chrono::milliseconds timestamp,
  std::function<void(Status, SequenceNumber)> callback) {
  return Status::OK();
}

Status BufferedLogStorage::CreateAsyncReaders(
  unsigned int parallelism,
  std::function<bool(LogRecord&)> record_cb,
  std::function<bool(const GapRecord&)> gap_cb,
  std::vector<AsyncLogReader*>* readers) {
  return Status::OK();
}

BufferedLogStorage::BufferedLogStorage(
  Env* env,
  std::shared_ptr<Logger> info_log,
  std::unique_ptr<LogStorage> wrapped_storage,
  MsgLoop* msg_loop,
  size_t max_batch_entries,
  size_t max_batch_bytes,
  std::chrono::microseconds max_batch_latency)
: env_(env)
, info_log_(std::move(info_log))
, storage_(std::move(wrapped_storage))
, msg_loop_(msg_loop)
, max_batch_entries_(max_batch_entries)
, max_batch_bytes_(max_batch_bytes)
, max_batch_latency_(max_batch_latency) {
  // Calculate bits for batch.
  batch_bits_ = 0;
  while ((1u << batch_bits_) < max_batch_entries_) {
    ++batch_bits_;
  }
  assert(batch_bits_ <= 8);  // up to 8 bits supported
}

BufferedAsyncLogReader::BufferedAsyncLogReader(
  BufferedLogStorage* storage,
  std::function<bool(LogRecord&)> record_cb,
  std::function<bool(const GapRecord&)> gap_cb,
  std::unique_ptr<AsyncLogReader> reader)
: reader_(std::move(reader))
, storage_(storage) {

}

BufferedAsyncLogReader::~BufferedAsyncLogReader() {

}

Status BufferedAsyncLogReader::Open(LogID id,
            SequenceNumber startPoint,
            SequenceNumber endPoint) {
  return Status::OK();
}

Status BufferedAsyncLogReader::Close(LogID id) {
  return Status::OK();
}

}  // namespace rocketspeed
