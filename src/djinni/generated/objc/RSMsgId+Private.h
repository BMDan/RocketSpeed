// AUTOGENERATED FILE - DO NOT MODIFY!
// This file generated by Djinni from rocketspeed.djinni

#import "RSMsgId.h"
#include "MsgId.hpp"
#import <Foundation/Foundation.h>

@interface RSMsgId ()

- (id)initWithMsgId:(RSMsgId *)MsgId;
- (id)initWithGuid:(NSData *)guid;
- (id)initWithCppMsgId:(const ::rocketglue::MsgId &)MsgId;
- (::rocketglue::MsgId)cppMsgId;

@end