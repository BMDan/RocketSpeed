// AUTOGENERATED FILE - DO NOT MODIFY!
// This file generated by Djinni from rocketspeed.djinni

#import "RSClientCppProxy+Private.h"
#import "DJIError.h"
#import "RSClient.h"
#import "RSClientCppProxy+Private.h"
#import "RSClientID+Private.h"
#import "RSConfigurationCppProxy+Private.h"
#import "RSMessageReceivedCallbackObjcProxy+Private.h"
#import "RSMsgId+Private.h"
#import "RSNamespaceID+Private.h"
#import "RSPublishCallbackObjcProxy+Private.h"
#import "RSSubscribeCallbackObjcProxy+Private.h"
#import "RSSubscriptionPair+Private.h"
#import "RSTopic+Private.h"
#import "RSTopicOptions+Private.h"
#include <exception>
#include <utility>

static_assert(__has_feature(objc_arc), "Djinni requires ARC to be enabled for this file");

@implementation RSClientCppProxy

- (id)initWithCpp:(const std::shared_ptr<::rocketglue::Client> &)cppRef
{
    if (self = [super init]) {
        _cppRef = cppRef;
    }
    return self;
}

- (void)dealloc
{
    djinni::DbxCppWrapperCache<::rocketglue::Client> & cache = djinni::DbxCppWrapperCache<::rocketglue::Client>::getInstance();
    cache.remove(_cppRef);
}

+ (id)ClientWithCpp:(const std::shared_ptr<::rocketglue::Client> &)cppRef
{
    djinni::DbxCppWrapperCache<::rocketglue::Client> & cache = djinni::DbxCppWrapperCache<::rocketglue::Client>::getInstance();
    return cache.get(cppRef, [] (const std::shared_ptr<::rocketglue::Client> & p) { return [[RSClientCppProxy alloc] initWithCpp:p]; });
}

+ (id <RSClient>)Open:(RSClientID *)clientId config:(id <RSConfiguration>)config publishCallback:(id <RSPublishCallback>)publishCallback subscribeCallback:(id <RSSubscribeCallback>)subscribeCallback receiveCallback:(id <RSMessageReceivedCallback>)receiveCallback {
    try {
        ::rocketglue::ClientID cppClientId = std::move([clientId cppClientID]);
        std::shared_ptr<::rocketglue::Configuration> cppConfig = [(RSConfigurationCppProxy *)config cppRef];
        std::shared_ptr<::rocketglue::PublishCallback> cppPublishCallback = ::rocketglue::PublishCallbackObjcProxy::PublishCallback_with_objc(publishCallback);
        std::shared_ptr<::rocketglue::SubscribeCallback> cppSubscribeCallback = ::rocketglue::SubscribeCallbackObjcProxy::SubscribeCallback_with_objc(subscribeCallback);
        std::shared_ptr<::rocketglue::MessageReceivedCallback> cppReceiveCallback = ::rocketglue::MessageReceivedCallbackObjcProxy::MessageReceivedCallback_with_objc(receiveCallback);
        std::shared_ptr<::rocketglue::Client> cppRet = ::rocketglue::Client::Open(std::move(cppClientId), std::move(cppConfig), std::move(cppPublishCallback), std::move(cppSubscribeCallback), std::move(cppReceiveCallback));
        id <RSClient> objcRet = [RSClientCppProxy ClientWithCpp:cppRet];
        return objcRet;
    } DJINNI_TRANSLATE_EXCEPTIONS()
}

- (void)Publish:(RSTopic *)topicName namespaceId:(RSNamespaceID *)namespaceId options:(RSTopicOptions *)options data:(NSData *)data msgid:(RSMsgId *)msgid {
    try {
        ::rocketglue::Topic cppTopicName = std::move([topicName cppTopic]);
        ::rocketglue::NamespaceID cppNamespaceId = std::move([namespaceId cppNamespaceID]);
        ::rocketglue::TopicOptions cppOptions = std::move([options cppTopicOptions]);
        std::vector<uint8_t> cppData([data length]);
        [data getBytes:(static_cast<void *>(cppData.data())) length:[data length]];
        ::rocketglue::MsgId cppMsgid = std::move([msgid cppMsgId]);
        _cppRef->Publish(std::move(cppTopicName), std::move(cppNamespaceId), std::move(cppOptions), std::move(cppData), std::move(cppMsgid));
    } DJINNI_TRANSLATE_EXCEPTIONS()
}

- (void)ListenTopics:(NSMutableArray *)names options:(RSTopicOptions *)options {
    try {
        std::vector<::rocketglue::SubscriptionPair> cppNames;
        cppNames.reserve([names count]);
        for (RSSubscriptionPair *objcValue_0 in names) {
            ::rocketglue::SubscriptionPair cppValue_0 = std::move([objcValue_0 cppSubscriptionPair]);
            cppNames.push_back(std::move(cppValue_0));
        }
        ::rocketglue::TopicOptions cppOptions = std::move([options cppTopicOptions]);
        _cppRef->ListenTopics(std::move(cppNames), std::move(cppOptions));
    } DJINNI_TRANSLATE_EXCEPTIONS()
}

@end
