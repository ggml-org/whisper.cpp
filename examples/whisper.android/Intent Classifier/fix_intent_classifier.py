#!/usr/bin/env python3
"""
Fixed TensorFlow Lite Intent Classifier Converter
Creates a simple neural network equivalent to the LogisticRegression classifier
"""

import json
import os
import pickle
import numpy as np
import tensorflow as tf
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder
from sentence_transformers import SentenceTransformer

def create_intent_classifier_tflite():
    """Create and convert intent classifier to TFLite"""
    print("🎯 Creating Intent Classifier TFLite model...")
    
    # Load training data
    with open('noise_ai_training_dataset_cleaned_1.json', 'r') as f:
        training_data = json.load(f)
    
    # Load sentence transformer and encode texts
    sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
    texts = [item['text'] for item in training_data]
    intents = [item['intent'] for item in training_data]
    
    print("Encoding texts for intent classification...")
    X = sentence_model.encode(texts, show_progress_bar=True)
    
    # Encode labels
    intent_encoder = LabelEncoder()
    y = intent_encoder.fit_transform(intents)
    
    # Train sklearn model first
    sklearn_model = LogisticRegression(max_iter=1000, random_state=42)
    sklearn_model.fit(X, y)
    
    print(f"Sklearn model accuracy: {sklearn_model.score(X, y):.3f}")
    
    # Create equivalent TensorFlow model
    input_dim = X.shape[1]
    n_classes = len(intent_encoder.classes_)
    
    print(f"Creating TF model: {input_dim} inputs → {n_classes} classes")
    
    # Build the model
    tf_model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(input_dim,), name='embedding_input'),
        tf.keras.layers.Dense(n_classes, activation='softmax', name='intent_output')
    ])
    
    # Copy weights from sklearn model
    weights = sklearn_model.coef_.T  # Transpose for Keras format
    bias = sklearn_model.intercept_
    
    tf_model.layers[0].set_weights([weights, bias])
    
    # Compile (for validation)
    tf_model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    # Validate the model
    tf_predictions = tf_model.predict(X, verbose=0)
    sklearn_predictions = sklearn_model.predict_proba(X)
    
    # Check if predictions are similar
    tf_classes = np.argmax(tf_predictions, axis=1)
    sklearn_classes = sklearn_model.predict(X)
    accuracy = np.mean(tf_classes == sklearn_classes)
    
    print(f"TF model validation accuracy: {accuracy:.3f}")
    
    # Convert to TFLite with correct API
    converter = tf.lite.TFLiteConverter.from_keras_model(tf_model)
    
    # Set optimization flags
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    
    # Use correct constant reference for TF 2.x
    try:
        converter.target_spec.supported_types = [tf.float16]
    except:
        print("Float16 optimization not available, using default")
    
    # Convert
    print("Converting to TFLite...")
    tflite_model = converter.convert()
    
    # Save the model
    os.makedirs('tflite_models', exist_ok=True)
    tflite_path = os.path.join('tflite_models', 'intent_classifier.tflite')
    
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"✅ Intent classifier saved: {tflite_path}")
    print(f"📊 Model size: {len(tflite_model) / 1024:.2f} KB")
    
    # Test the TFLite model
    print("🧪 Testing TFLite model...")
    interpreter = tf.lite.Interpreter(model_path=tflite_path)
    interpreter.allocate_tensors()
    
    input_details = interpreter.get_input_details()
    output_details = interpreter.get_output_details()
    
    print(f"Input shape: {input_details[0]['shape']}")
    print(f"Output shape: {output_details[0]['shape']}")
    
    # Test with sample data
    test_samples = X[:5]
    
    print("\n🔍 Testing sample predictions:")
    for i, sample in enumerate(test_samples):
        # TFLite prediction
        interpreter.set_tensor(input_details[0]['index'], sample.reshape(1, -1).astype(np.float32))
        interpreter.invoke()
        tflite_output = interpreter.get_tensor(output_details[0]['index'])
        tflite_pred = np.argmax(tflite_output[0])
        tflite_confidence = np.max(tflite_output[0])
        
        # Original prediction
        orig_pred = sklearn_model.predict([sample])[0]
        
        # Convert to intent names
        tflite_intent = intent_encoder.inverse_transform([tflite_pred])[0]
        orig_intent = intent_encoder.inverse_transform([orig_pred])[0]
        
        print(f"  Sample {i+1}: '{texts[i][:50]}...'")
        print(f"    Original: {orig_intent}")
        print(f"    TFLite:   {tflite_intent} (confidence: {tflite_confidence:.3f})")
        
        if tflite_intent == orig_intent:
            print(f"    ✅ Match!")
        else:
            print(f"    ❌ Mismatch")
    
    # Update metadata
    metadata = {
        "model_info": {
            "created_date": "2025-09-29",
            "framework": "tensorflow_lite",
            "input_dimension": input_dim,
            "num_intents": n_classes,
            "model_type": "intent_classification"
        },
        "files": {
            "sentence_encoder": "lightweight_sentence_encoder.tflite",
            "intent_classifier": "intent_classifier.tflite",
            "vocabulary": "vocabulary.json",
            "intent_encoder": "intent_encoder.pkl"
        },
        "intent_mapping": {int(i): intent for i, intent in enumerate(intent_encoder.classes_)},
        "intents": list(intent_encoder.classes_),
        "usage": {
            "preprocessing": "lowercase, tokenize",
            "max_sequence_length": 32,
            "input_format": "text_string"
        }
    }
    
    # Save updated metadata
    metadata_path = os.path.join('tflite_models', 'model_metadata.json')
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save intent encoder
    encoder_path = os.path.join('tflite_models', 'intent_encoder.pkl')
    with open(encoder_path, 'wb') as f:
        pickle.dump(intent_encoder, f)
    
    print(f"✅ Metadata updated: {metadata_path}")
    print(f"✅ Intent encoder saved: {encoder_path}")
    
    return tflite_path, metadata_path

if __name__ == "__main__":
    print("🔧 Fixed TensorFlow Lite Intent Classifier Conversion")
    print("=" * 60)
    
    tflite_path, metadata_path = create_intent_classifier_tflite()
    
    print(f"\n🎉 Intent classifier successfully converted!")
    print(f"📱 Ready for Android integration!")
    
    # Show final file structure
    print(f"\n📋 Final TFLite models:")
    tflite_dir = "tflite_models"
    for file in os.listdir(tflite_dir):
        file_path = os.path.join(tflite_dir, file)
        size = os.path.getsize(file_path)
        if size > 1024:
            print(f"  {file}: {size/1024:.2f} KB")
        else:
            print(f"  {file}: {size} bytes")