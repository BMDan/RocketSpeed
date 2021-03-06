//  Copyright (c) 2014, Facebook, Inc.  All rights reserved.
//  This source code is licensed under the BSD-style license found in the
//  LICENSE file in the root directory of this source tree. An additional grant
//  of patent rights can be found in the PATENTS file in the same directory.
//

#include <sys/time.h>
#include <signal.h>

#include "include/Slice.h"
#include "include/Env.h"
#include "src/port/stack_trace.h"
#include "src/util/arena.h"

namespace rocketspeed {

void Env::InstallSignalHandlers() {
  // Ignore SIGPIPE, we'll just handle the EPIPE returned by write.
  signal(SIGPIPE, SIG_IGN);
  // Print backtrace on these.
  port::InstallStackTraceHandler();
}

Env::~Env() {
}

RandomAccessFile::~RandomAccessFile() {
}

WritableFile::~WritableFile() {
}

FileLock::~FileLock() {
}

Status WriteStringToFile(Env* env, const Slice& data, const std::string& fname,
                         bool should_sync) {
  unique_ptr<WritableFile> file;
  EnvOptions soptions;
  Status s = env->NewWritableFile(fname, &file, soptions);
  if (!s.ok()) {
    return s;
  }
  s = file->Append(data);
  if (s.ok() && should_sync) {
    s = file->Sync();
  }
  if (!s.ok()) {
    env->DeleteFile(fname);
  }
  return s;
}

Status ReadFileToString(Env* env, const std::string& fname, std::string* data) {
  EnvOptions soptions;
  data->clear();
  unique_ptr<SequentialFile> file;
  Status s = env->NewSequentialFile(fname, &file, soptions);
  if (!s.ok()) {
    return s;
  }
  static const int kBufferSize = 8192;
  char* space = new char[kBufferSize];
  while (true) {
    Slice fragment;
    s = file->Read(kBufferSize, &fragment, space);
    if (!s.ok()) {
      break;
    }
    data->append(fragment.data(), fragment.size());
    if (fragment.empty()) {
      break;
    }
  }
  delete[] space;
  return s;
}

EnvWrapper::~EnvWrapper() {
}

}  // namespace rocketspeed
