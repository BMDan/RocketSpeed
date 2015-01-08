// AUTOGENERATED FILE - DO NOT MODIFY!
// This file generated by Djinni from rocketspeed.djinni

#pragma once

#include "SubscriptionRequestImpl.hpp"
#include "djinni_support.hpp"

namespace djinni_generated {

class NativeSubscriptionRequestImpl final {
public:
    using CppType = ::rocketspeed::djinni::SubscriptionRequestImpl;
    using JniType = jobject;

    static jobject toJava(JNIEnv*, ::rocketspeed::djinni::SubscriptionRequestImpl);
    static ::rocketspeed::djinni::SubscriptionRequestImpl fromJava(JNIEnv*, jobject);

    const djinni::GlobalRef<jclass> clazz { djinni::jniFindClass("org/rocketspeed/SubscriptionRequestImpl") };
    const jmethodID jconstructor { djinni::jniGetMethodID(clazz.get(), "<init>", "(SLjava/lang/String;ZLorg/rocketspeed/SubscriptionStartImpl;)V") };
    const jfieldID field_namespaceId { djinni::jniGetFieldID(clazz.get(), "namespaceId", "S") };
    const jfieldID field_topicName { djinni::jniGetFieldID(clazz.get(), "topicName", "Ljava/lang/String;") };
    const jfieldID field_subscribe { djinni::jniGetFieldID(clazz.get(), "subscribe", "Z") };
    const jfieldID field_sequenceNumber { djinni::jniGetFieldID(clazz.get(), "sequenceNumber", "Lorg/rocketspeed/SubscriptionStartImpl;") };

private:
    NativeSubscriptionRequestImpl() {}
    friend class djinni::JniClass<::djinni_generated::NativeSubscriptionRequestImpl>;
};

}  // namespace djinni_generated
