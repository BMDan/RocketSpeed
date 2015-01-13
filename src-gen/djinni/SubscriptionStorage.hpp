// AUTOGENERATED FILE - DO NOT MODIFY!
// This file generated by Djinni from rocketspeed.djinni

#pragma once

#include "StorageType.hpp"
#include <experimental/optional>
#include <string>
#include <utility>

namespace rocketspeed { namespace djinni {

struct SubscriptionStorage final {

    StorageType type;

    /** Used only when this is a file storage */
    std::experimental::optional<std::string> file_path;


    SubscriptionStorage(
            StorageType type,
            std::experimental::optional<std::string> file_path) :
                type(std::move(type)),
                file_path(std::move(file_path)) {
    }
};

} }  // namespace rocketspeed::djinni
