// AUTOGENERATED FILE - DO NOT MODIFY!
// This file generated by Djinni from rocketspeed.djinni

#pragma once

#include "Status.hpp"
#include "djinni_support.hpp"

namespace djinni_generated {

class NativeStatus final {
public:
    using CppType = ::rocketspeed::djinni::Status;
    using JniType = jobject;

    static jobject toJava(JNIEnv*, ::rocketspeed::djinni::Status);
    static ::rocketspeed::djinni::Status fromJava(JNIEnv*, jobject);

    const djinni::GlobalRef<jclass> clazz { djinni::jniFindClass("org/rocketspeed/Status") };
    const jmethodID jconstructor { djinni::jniGetMethodID(clazz.get(), "<init>", "(Lorg/rocketspeed/StatusCode;Ljava/lang/String;)V") };
    const jfieldID field_code { djinni::jniGetFieldID(clazz.get(), "code", "Lorg/rocketspeed/StatusCode;") };
    const jfieldID field_state { djinni::jniGetFieldID(clazz.get(), "state", "Ljava/lang/String;") };

private:
    NativeStatus() {}
    friend class djinni::JniClass<::djinni_generated::NativeStatus>;
};

}  // namespace djinni_generated
