// AUTOGENERATED FILE - DO NOT MODIFY!
// This file generated by Djinni from rocketspeed.djinni

#include "NativeSubscribeCallbackImpl.hpp"  // my header
#include "HBool.hpp"
#include "HI32.hpp"
#include "HI64.hpp"
#include "HString.hpp"
#include "NativeStatus.hpp"

namespace djinni_generated {

NativeSubscribeCallbackImpl::NativeSubscribeCallbackImpl() : djinni::JniInterface<::rocketspeed::djinni::SubscribeCallbackImpl, NativeSubscribeCallbackImpl>() {}

NativeSubscribeCallbackImpl::JavaProxy::JavaProxy(jobject obj) : JavaProxyCacheEntry(obj) {}

void NativeSubscribeCallbackImpl::JavaProxy::JavaProxy::Call(::rocketspeed::djinni::Status c_status, int32_t c_namespace_id, std::string c_topic_name, int64_t c_sequence_number, bool c_subscribed) {
    JNIEnv * const jniEnv = djinni::jniGetThreadEnv();
    djinni::JniLocalScope jscope(jniEnv, 10);
    djinni::LocalRef<jobject> j_status(jniEnv, NativeStatus::toJava(jniEnv, c_status));
    jint j_namespace_id = ::djinni::HI32::Unboxed::toJava(jniEnv, c_namespace_id);
    djinni::LocalRef<jstring> j_topic_name(jniEnv, ::djinni::HString::toJava(jniEnv, c_topic_name));
    jlong j_sequence_number = ::djinni::HI64::Unboxed::toJava(jniEnv, c_sequence_number);
    jboolean j_subscribed = ::djinni::HBool::Unboxed::toJava(jniEnv, c_subscribed);
    const auto & data = djinni::JniClass<::djinni_generated::NativeSubscribeCallbackImpl>::get();
    jniEnv->CallVoidMethod(getGlobalRef(), data.method_Call, j_status.get(), j_namespace_id, j_topic_name.get(), j_sequence_number, j_subscribed);
    djinni::jniExceptionCheck(jniEnv);
};

}  // namespace djinni_generated
