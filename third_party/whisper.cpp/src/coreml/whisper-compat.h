#import <CoreML/CoreML.h>

@interface MLModel (Compat)
- (void) predictionFromFeatures:(id<MLFeatureProvider>) input
              completionHandler:(void (^)(id<MLFeatureProvider> output, NSError * error)) completionHandler;

- (void) predictionFromFeatures:(id<MLFeatureProvider>) input
                        options:(MLPredictionOptions *) options
              completionHandler:(void (^)(id<MLFeatureProvider> output, NSError * error)) completionHandler;
@end
