// AUTOGENERATED FILE - DO NOT MODIFY!
// This file generated by Djinni from rocketspeed.djinni

#pragma once

#include "MsgIdImpl.hpp"
#include "djinni_support.hpp"

namespace djinni_generated {

class NativeMsgIdImpl final {
public:
    using CppType = ::rocketspeed::djinni::MsgIdImpl;
    using JniType = jobject;

    static jobject toJava(JNIEnv*, ::rocketspeed::djinni::MsgIdImpl);
    static ::rocketspeed::djinni::MsgIdImpl fromJava(JNIEnv*, jobject);

    const djinni::GlobalRef<jclass> clazz { djinni::jniFindClass("org/rocketspeed/MsgIdImpl") };
    const jmethodID jconstructor { djinni::jniGetMethodID(clazz.get(), "<init>", "([B)V") };
    const jfieldID field_guid { djinni::jniGetFieldID(clazz.get(), "guid", "[B") };

private:
    NativeMsgIdImpl() {}
    friend class djinni::JniClass<::djinni_generated::NativeMsgIdImpl>;
};

}  // namespace djinni_generated
