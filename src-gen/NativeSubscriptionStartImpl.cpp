// AUTOGENERATED FILE - DO NOT MODIFY!
// This file generated by Djinni from rocketspeed.djinni

#include "NativeSubscriptionStartImpl.hpp"  // my header
#include "HBool.hpp"
#include "HI64.hpp"

namespace djinni_generated {

jobject NativeSubscriptionStartImpl::toJava(JNIEnv* jniEnv, ::rocketspeed::djinni::SubscriptionStartImpl c) {
    jlong j_sequence_number = ::djinni::HI64::Unboxed::toJava(jniEnv, c.sequence_number);
    jboolean j_present = ::djinni::HBool::Unboxed::toJava(jniEnv, c.present);
    const auto & data = djinni::JniClass<::djinni_generated::NativeSubscriptionStartImpl>::get();
    jobject r = jniEnv->NewObject(data.clazz.get(), data.jconstructor, j_sequence_number, j_present);
    djinni::jniExceptionCheck(jniEnv);
    return r;
}

::rocketspeed::djinni::SubscriptionStartImpl NativeSubscriptionStartImpl::fromJava(JNIEnv* jniEnv, jobject j) {
    assert(j != nullptr);
    const auto & data = djinni::JniClass<::djinni_generated::NativeSubscriptionStartImpl>::get();
    return ::rocketspeed::djinni::SubscriptionStartImpl(
        ::djinni::HI64::Unboxed::fromJava(jniEnv, jniEnv->GetLongField(j, data.field_sequenceNumber)),
        ::djinni::HBool::Unboxed::fromJava(jniEnv, jniEnv->GetBooleanField(j, data.field_present)));
}

}  // namespace djinni_generated
