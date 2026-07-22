#import "whisper-compat.h"
#import <Foundation/Foundation.h>

@implementation MLModel (Compat)

#if !defined(MAC_OS_X_VERSION_14_00) || MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_14_00

- (void) predictionFromFeatures:(id<MLFeatureProvider>) input
              completionHandler:(void (^)(id<MLFeatureProvider> output, NSError * error)) completionHandler {
    [NSOperationQueue.new addOperationWithBlock:^{
        NSError *error = nil;
        id<MLFeatureProvider> prediction = [self predictionFromFeatures:input error:&error];

        [NSOperationQueue.mainQueue addOperationWithBlock:^{
            completionHandler(prediction, error);
        }];
    }];
}

- (void) predictionFromFeatures:(id<MLFeatureProvider>) input
                        options:(MLPredictionOptions *) options
              completionHandler:(void (^)(id<MLFeatureProvider> output, NSError * error)) completionHandler {
    [NSOperationQueue.new addOperationWithBlock:^{
        NSError *error = nil;
        id<MLFeatureProvider> prediction = [self predictionFromFeatures:input options:options error:&error];

        [NSOperationQueue.mainQueue addOperationWithBlock:^{
            completionHandler(prediction, error);
        }];
    }];
}

#endif

@end
