#import "MLModel+Compat.h"
#import <Foundation/Foundation.h>

@implementation MLModel (Compat)

#if !defined(MAC_OS_X_VERSION_14_00) || MAC_OS_X_VERSION_MAX_ALLOWED < MAC_OS_X_VERSION_14_00

- (void) predictionFromFeatures:(id<MLFeatureProvider>) input
              completionHandler:(void (^)(id<MLFeatureProvider> output, NSError * error)) completionHandler {
    [NSOperationQueue.mainQueue addOperationWithBlock:^{
        NSError *error = nil;
        id<MLFeatureProvider> prediction = [self predictionFromFeatures:input error:&error];
        completionHandler(prediction, error);
    }];
}

- (void) predictionFromFeatures:(id<MLFeatureProvider>) input
                        options:(MLPredictionOptions *) options
              completionHandler:(void (^)(id<MLFeatureProvider> output, NSError * error)) completionHandler {
    [NSOperationQueue.mainQueue addOperationWithBlock:^{
        NSError *error = nil;
        id<MLFeatureProvider> prediction = [self predictionFromFeatures:input options:options error:&error];
        completionHandler(prediction, error);
    }];
}

#endif

@end
