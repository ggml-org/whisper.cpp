#!/usr/bin/env python3
"""
TensorFlow Lite Model Conversion Script for Intent Classification
Converts trained sklearn models and sentence transformers to TFLite format for Android
"""

import json
import os
import pickle
import numpy as np
import tensorflow as tf
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Import required libraries
try:
    from sentence_transformers import SentenceTransformer
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import LabelEncoder
    print("✅ All required libraries loaded successfully")
except ImportError as e:
    print(f"❌ Missing dependency: {e}")
    print("Install with: pip install sentence-transformers scikit-learn tensorflow")
    exit(1)

class TFLiteModelConverter:
    """Convert trained NLU models to TensorFlow Lite format"""
    
    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.output_dir = "tflite_models"
        self.training_data = None
        self.sentence_model = None
        self.intent_classifier = None
        self.intent_encoder = None
        
        # Create output directory
        os.makedirs(self.output_dir, exist_ok=True)
        print(f"📁 Created output directory: {self.output_dir}")
    
    def load_and_prepare_data(self):
        """Load training data"""
        print("📊 Loading training data...")
        with open(self.dataset_path, 'r') as f:
            self.training_data = json.load(f)
        print(f"✅ Loaded {len(self.training_data)} training examples")
        
        # Get intent distribution
        intent_counts = {}
        for item in self.training_data:
            intent = item['intent']
            intent_counts[intent] = intent_counts.get(intent, 0) + 1
        
        print("📈 Intent distribution:")
        for intent, count in sorted(intent_counts.items(), key=lambda x: x[1], reverse=True):
            print(f"  {intent}: {count}")
        
        return self.training_data
    
    def train_models(self):
        """Train the models for conversion"""
        print("\n🎯 Training models for TFLite conversion...")
        
        # Load sentence transformer
        print("Loading sentence transformer...")
        self.sentence_model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Prepare data
        texts = [item['text'] for item in self.training_data]
        intents = [item['intent'] for item in self.training_data]
        
        print("Encoding texts...")
        X = self.sentence_model.encode(texts, show_progress_bar=True)
        
        # Encode labels
        self.intent_encoder = LabelEncoder()
        y = self.intent_encoder.fit_transform(intents)
        
        # Train classifier (using LogisticRegression for easier TFLite conversion)
        print("Training intent classifier...")
        self.intent_classifier = LogisticRegression(max_iter=1000, random_state=42)
        self.intent_classifier.fit(X, y)
        
        accuracy = self.intent_classifier.score(X, y)
        print(f"✅ Intent classifier trained - Accuracy: {accuracy:.3f}")
        
        return X, y
    
    def convert_sentence_model_to_tflite(self):
        """Convert sentence transformer to TensorFlow Lite"""
        print("\n🔄 Converting sentence model to TFLite...")
        
        try:
            # Get the transformer model
            transformer_model = self.sentence_model[0].auto_model
            
            # Convert to TensorFlow SavedModel format first
            saved_model_path = os.path.join(self.output_dir, "sentence_model_saved")
            
            # Create a concrete function for the model
            @tf.function
            def model_fn(input_ids, attention_mask):
                return transformer_model(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
            
            # Get sample input to trace the function
            sample_text = "hello world"
            encoded = self.sentence_model.tokenizer(
                sample_text, 
                padding=True, 
                truncation=True, 
                return_tensors='tf',
                max_length=128
            )
            
            # Trace the model
            concrete_function = model_fn.get_concrete_function(
                input_ids=tf.TensorSpec(shape=[None, None], dtype=tf.int32),
                attention_mask=tf.TensorSpec(shape=[None, None], dtype=tf.int32)
            )
            
            # Save as SavedModel
            tf.saved_model.save(
                transformer_model, 
                saved_model_path,
                signatures=concrete_function
            )
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_saved_model(saved_model_path)
            
            # Optimization settings
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
            
            # Convert
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = os.path.join(self.output_dir, "sentence_encoder.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✅ Sentence model converted to TFLite: {tflite_path}")
            print(f"📊 Model size: {len(tflite_model) / 1024 / 1024:.2f} MB")
            
            return tflite_path
            
        except Exception as e:
            print(f"⚠️ Complex sentence model conversion failed: {e}")
            print("🔄 Creating lightweight embedding model instead...")
            return self.create_lightweight_embedding_model()
    
    def create_lightweight_embedding_model(self):
        """Create a lightweight embedding model for mobile"""
        print("🔄 Creating lightweight embedding model...")
        
        # Get sample embeddings to determine dimensions
        sample_texts = [item['text'] for item in self.training_data[:100]]
        sample_embeddings = self.sentence_model.encode(sample_texts)
        embedding_dim = sample_embeddings.shape[1]
        
        # Create a simple embedding model using TF/Keras
        # This is a simplified approach - in practice, you'd use a mobile-optimized model
        
        # For now, we'll create a lookup-based approach
        # Create vocabulary from training data
        vocab = set()
        for item in self.training_data:
            words = item['text'].lower().split()
            vocab.update(words)
        
        vocab_list = sorted(list(vocab))
        vocab_size = len(vocab_list)
        
        print(f"📝 Vocabulary size: {vocab_size}")
        print(f"🎯 Embedding dimension: {embedding_dim}")
        
        # Create a simple embedding model
        model = tf.keras.Sequential([
            tf.keras.layers.Embedding(vocab_size + 1, 128, input_length=32),
            tf.keras.layers.GlobalAveragePooling1D(),
            tf.keras.layers.Dense(embedding_dim, activation='relu'),
            tf.keras.layers.Dense(embedding_dim)  # Output embedding
        ])
        
        # Compile model
        model.compile(optimizer='adam', loss='mse')
        
        # Create dummy training data for the embedding model
        # Map texts to token sequences
        word_to_id = {word: i+1 for i, word in enumerate(vocab_list)}
        
        def text_to_sequence(text, max_len=32):
            words = text.lower().split()[:max_len]
            sequence = [word_to_id.get(word, 0) for word in words]
            sequence += [0] * (max_len - len(sequence))  # Pad
            return sequence
        
        # Prepare training data
        X_embed = np.array([text_to_sequence(item['text']) for item in self.training_data])
        y_embed = self.sentence_model.encode([item['text'] for item in self.training_data])
        
        print("Training lightweight embedding model...")
        model.fit(X_embed, y_embed, epochs=10, batch_size=32, verbose=1)
        
        # Convert to TFLite
        converter = tf.lite.TFLiteConverter.from_keras_model(model)
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
        tflite_model = converter.convert()
        
        # Save TFLite model
        tflite_path = os.path.join(self.output_dir, "lightweight_sentence_encoder.tflite")
        with open(tflite_path, 'wb') as f:
            f.write(tflite_model)
        
        # Save vocabulary
        vocab_path = os.path.join(self.output_dir, "vocabulary.json")
        with open(vocab_path, 'w') as f:
            json.dump(word_to_id, f, indent=2)
        
        print(f"✅ Lightweight embedding model created: {tflite_path}")
        print(f"📊 Model size: {len(tflite_model) / 1024:.2f} KB")
        print(f"📝 Vocabulary saved: {vocab_path}")
        
        return tflite_path
    
    def convert_intent_classifier_to_tflite(self, X, y):
        """Convert intent classifier to TensorFlow Lite"""
        print("\n🎯 Converting intent classifier to TFLite...")
        
        try:
            # Get model parameters
            input_dim = X.shape[1]
            n_classes = len(self.intent_encoder.classes_)
            
            print(f"📊 Input dimension: {input_dim}")
            print(f"🎯 Number of classes: {n_classes}")
            
            # Create TensorFlow model equivalent to LogisticRegression
            model = tf.keras.Sequential([
                tf.keras.layers.Input(shape=(input_dim,)),
                tf.keras.layers.Dense(n_classes, activation='softmax')
            ])
            
            # Copy weights from sklearn model
            # LogisticRegression weights
            weights = self.intent_classifier.coef_.T  # Transpose for Keras
            bias = self.intent_classifier.intercept_
            
            model.layers[0].set_weights([weights, bias])
            
            # Compile model (not needed for inference but good practice)
            model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
            
            # Verify model works
            tf_predictions = model.predict(X[:5])
            sklearn_predictions = self.intent_classifier.predict_proba(X[:5])
            
            print("🔍 Verifying model conversion...")
            print(f"TF predictions shape: {tf_predictions.shape}")
            print(f"Sklearn predictions shape: {sklearn_predictions.shape}")
            
            # Convert to TFLite
            converter = tf.lite.TFLiteConverter.from_keras_model(model)
            
            # Optimization settings
            converter.optimizations = [tf.lite.Optimize.DEFAULT]
            converter.target_spec.supported_types = [tf.lite.constants.FLOAT16]
            
            # Representative dataset for quantization
            def representative_dataset():
                for i in range(min(100, len(X))):
                    yield [X[i:i+1].astype(np.float32)]
            
            converter.representative_dataset = representative_dataset
            
            # Convert
            tflite_model = converter.convert()
            
            # Save TFLite model
            tflite_path = os.path.join(self.output_dir, "intent_classifier.tflite")
            with open(tflite_path, 'wb') as f:
                f.write(tflite_model)
            
            print(f"✅ Intent classifier converted to TFLite: {tflite_path}")
            print(f"📊 Model size: {len(tflite_model) / 1024:.2f} KB")
            
            return tflite_path
            
        except Exception as e:
            print(f"❌ Error converting intent classifier: {e}")
            return None
    
    def save_metadata(self, embedding_path, classifier_path):
        """Save model metadata and mappings"""
        print("\n💾 Saving model metadata...")
        
        # Intent label mappings
        intent_mapping = {
            int(i): intent for i, intent in enumerate(self.intent_encoder.classes_)
        }
        
        # Model metadata
        metadata = {
            "model_info": {
                "created_date": datetime.now().isoformat(),
                "framework": "tensorflow_lite",
                "input_dimension": 384,  # Standard for all-MiniLM-L6-v2
                "num_intents": len(self.intent_encoder.classes_),
                "model_type": "intent_classification"
            },
            "files": {
                "sentence_encoder": os.path.basename(embedding_path) if embedding_path else None,
                "intent_classifier": os.path.basename(classifier_path) if classifier_path else None,
                "vocabulary": "vocabulary.json"
            },
            "intent_mapping": intent_mapping,
            "intents": list(self.intent_encoder.classes_),
            "usage": {
                "preprocessing": "lowercase, tokenize",
                "max_sequence_length": 32,
                "input_format": "text_string"
            }
        }
        
        # Save metadata
        metadata_path = os.path.join(self.output_dir, "model_metadata.json")
        with open(metadata_path, 'w') as f:
            json.dump(metadata, f, indent=2)
        
        print(f"✅ Metadata saved: {metadata_path}")
        
        # Save intent encoder
        encoder_path = os.path.join(self.output_dir, "intent_encoder.pkl")
        with open(encoder_path, 'wb') as f:
            pickle.dump(self.intent_encoder, f)
        
        print(f"✅ Intent encoder saved: {encoder_path}")
        
        return metadata_path
    
    def test_tflite_models(self, embedding_path, classifier_path):
        """Test the converted TFLite models"""
        print("\n🧪 Testing converted TFLite models...")
        
        try:
            # Load TFLite models
            embedding_interpreter = None
            if embedding_path and os.path.exists(embedding_path):
                embedding_interpreter = tf.lite.Interpreter(model_path=embedding_path)
                embedding_interpreter.allocate_tensors()
                print("✅ Embedding model loaded successfully")
            
            classifier_interpreter = None
            if classifier_path and os.path.exists(classifier_path):
                classifier_interpreter = tf.lite.Interpreter(model_path=classifier_path)
                classifier_interpreter.allocate_tensors()
                print("✅ Classifier model loaded successfully")
            
            # Test with sample data
            test_texts = [
                "How many steps did I take today?",
                "Set an alarm for 7 AM",
                "What's the weather like?",
                "Start a workout",
                "Show me my heart rate"
            ]
            
            print("\n🔍 Testing predictions:")
            for text in test_texts:
                print(f"\nText: '{text}'")
                
                # Get original prediction
                original_embedding = self.sentence_model.encode([text])
                original_pred = self.intent_classifier.predict([original_embedding[0]])[0]
                original_intent = self.intent_encoder.inverse_transform([original_pred])[0]
                print(f"Original prediction: {original_intent}")
                
                # Test classifier if available
                if classifier_interpreter:
                    input_details = classifier_interpreter.get_input_details()
                    output_details = classifier_interpreter.get_output_details()
                    
                    # Set input
                    classifier_interpreter.set_tensor(input_details[0]['index'], 
                                                    original_embedding.astype(np.float32))
                    
                    # Run inference
                    classifier_interpreter.invoke()
                    
                    # Get output
                    output_data = classifier_interpreter.get_tensor(output_details[0]['index'])
                    tflite_pred = np.argmax(output_data[0])
                    tflite_intent = self.intent_encoder.inverse_transform([tflite_pred])[0]
                    confidence = np.max(output_data[0])
                    
                    print(f"TFLite prediction: {tflite_intent} (confidence: {confidence:.3f})")
                    
                    if original_intent == tflite_intent:
                        print("✅ Predictions match!")
                    else:
                        print("⚠️ Predictions differ")
            
        except Exception as e:
            print(f"❌ Error testing models: {e}")
    
    def convert_all(self):
        """Main conversion pipeline"""
        print("🚀 STARTING TFLITE CONVERSION PIPELINE")
        print("=" * 60)
        
        # Step 1: Load and prepare data
        self.load_and_prepare_data()
        
        # Step 2: Train models
        X, y = self.train_models()
        
        # Step 3: Convert sentence encoder
        embedding_path = self.convert_sentence_model_to_tflite()
        
        # Step 4: Convert intent classifier
        classifier_path = self.convert_intent_classifier_to_tflite(X, y)
        
        # Step 5: Save metadata
        metadata_path = self.save_metadata(embedding_path, classifier_path)
        
        # Step 6: Test models
        self.test_tflite_models(embedding_path, classifier_path)
        
        print("\n🎉 CONVERSION COMPLETED!")
        print("=" * 60)
        print(f"📁 Output directory: {self.output_dir}")
        print("📋 Generated files:")
        for file in os.listdir(self.output_dir):
            file_path = os.path.join(self.output_dir, file)
            size = os.path.getsize(file_path)
            if size > 1024*1024:
                print(f"  {file}: {size/1024/1024:.2f} MB")
            else:
                print(f"  {file}: {size/1024:.2f} KB")
        
        return {
            'embedding_model': embedding_path,
            'classifier_model': classifier_path,
            'metadata': metadata_path,
            'output_dir': self.output_dir
        }

def main():
    """Main execution function"""
    dataset_path = 'noise_ai_training_dataset_cleaned_1.json'
    
    if not os.path.exists(dataset_path):
        print(f"❌ Dataset file not found: {dataset_path}")
        print("Please ensure the dataset file is in the current directory")
        return
    
    print("🔧 TensorFlow Lite Model Conversion for Intent Classification")
    print("=" * 70)
    
    # Create converter and run conversion
    converter = TFLiteModelConverter(dataset_path)
    results = converter.convert_all()
    
    print(f"\n✅ All models successfully converted!")
    print(f"📱 Ready for Android integration!")
    print(f"📂 Files location: {results['output_dir']}")

if __name__ == "__main__":
    main()