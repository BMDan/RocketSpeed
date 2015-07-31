// AUTOGENERATED FILE - DO NOT MODIFY!
// This file generated by Djinni from rocketspeed.djinni

#pragma once

#include "djinni_support.hpp"
#include "src-gen/djinni/cpp/InboundID.hpp"

namespace djinni_generated {

class NativeInboundID final {
public:
    using CppType = ::rocketspeed::djinni::InboundID;
    using JniType = jobject;

    using Boxed = NativeInboundID;

    ~NativeInboundID();

    static CppType toCpp(JNIEnv* jniEnv, JniType j);
    static ::djinni::LocalRef<JniType> fromCpp(JNIEnv* jniEnv, const CppType& c);

private:
    NativeInboundID();
    friend ::djinni::JniClass<NativeInboundID>;

    const ::djinni::GlobalRef<jclass> clazz { ::djinni::jniFindClass("org/rocketspeed/InboundID") };
    const jmethodID jconstructor { ::djinni::jniGetMethodID(clazz.get(), "<init>", "([B)V") };
    const jfieldID field_serialised { ::djinni::jniGetFieldID(clazz.get(), "serialised", "[B") };
};

}  // namespace djinni_generated
