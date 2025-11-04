#!/usr/bin/env python3
"""
Create a working sentence encoder using the compatible TensorFlow Lite setup
"""

import json
import numpy as np
import tensorflow as tf
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder

print("🔧 CREATING WORKING SENTENCE ENCODER")
print("=" * 50)

# Load training data
with open('noise_ai_training_dataset_cleaned_1.json', 'r') as f:
    training_data = json.load(f)

print(f"📊 Loaded {len(training_data)} training examples")

# Train models
print("🤖 Loading sentence transformer...")
sentence_model = SentenceTransformer('all-MiniLM-L6-v2')

texts = [item['text'] for item in training_data]
intents = [item['intent'] for item in training_data]

print("🔤 Encoding texts...")
X = sentence_model.encode(texts, show_progress_bar=True)

# Encode labels
intent_encoder = LabelEncoder()
y = intent_encoder.fit_transform(intents)

# Train classifier
print("🎯 Training classifier...")
classifier = LogisticRegression(max_iter=1000, random_state=42)
classifier.fit(X, y)

print(f"✅ Training accuracy: {classifier.score(X, y):.3f}")

# Load existing vocabulary
with open('tflite_models/vocabulary.json', 'r') as f:
    vocabulary = json.load(f)

print(f"📝 Using existing vocabulary: {len(vocabulary)} words")

# Create a simple but working embedding layer
print("🏗️ Creating working sentence encoder...")

def create_working_encoder():
    """Create a sentence encoder with a simple embedding layer that actually works"""
    
    vocab_size = len(vocabulary) + 1  # +1 for unknown token (0)
    max_length = 32
    embedding_dim = 64  # Smaller embedding for simplicity
    
    # Input layer
    inputs = tf.keras.Input(shape=(max_length,), dtype=tf.float32, name='embedding_input')
    
    # Convert float input to int for embedding lookup
    int_inputs = tf.cast(inputs, tf.int32)
    
    # Embedding layer - this will learn meaningful representations
    embeddings = tf.keras.layers.Embedding(
        input_dim=vocab_size,
        output_dim=embedding_dim,
        mask_zero=True,
        name='word_embeddings'
    )(int_inputs)
    
    # Global average pooling to get fixed-size representation
    pooled = tf.keras.layers.GlobalAveragePooling1D(name='global_avg_pool')(embeddings)
    
    # Dense layers to expand to full embedding dimension
    x = tf.keras.layers.Dense(128, activation='relu', name='dense1')(pooled)
    x = tf.keras.layers.Dropout(0.2, name='dropout')(x)
    x = tf.keras.layers.Dense(256, activation='relu', name='dense2')(x)
    
    # Final output layer to match expected embedding dimension
    output = tf.keras.layers.Dense(384, activation='tanh', name='final_embedding')(x)
    
    model = tf.keras.Model(inputs=inputs, outputs=output, name='sentence_encoder')
    return model

encoder_model = create_working_encoder()
print(f"📋 Model architecture:")
encoder_model.summary()

# Train the encoder to approximate the sentence transformer
print("🏋️ Training sentence encoder...")

# Prepare training data
sample_sequences = []
sample_embeddings = []

for text in texts:  # Use all training data
    words = text.lower().split()[:32]
    sequence = [vocabulary.get(word, 0) for word in words]
    sequence += [0] * (32 - len(sequence))  # Pad to 32
    
    # Get actual embedding from sentence transformer
    embedding = sentence_model.encode([text])[0]
    
    sample_sequences.append(sequence)
    sample_embeddings.append(embedding)

sample_sequences = np.array(sample_sequences, dtype=np.float32)  # Convert to float32
sample_embeddings = np.array(sample_embeddings, dtype=np.float32)

print(f"📊 Training data shape: {sample_sequences.shape} -> {sample_embeddings.shape}")

# Compile and train
encoder_model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
    loss='mse',
    metrics=['mae']
)

# Train the model
encoder_model.fit(
    sample_sequences, 
    sample_embeddings,
    epochs=50,
    batch_size=32,
    validation_split=0.2,
    verbose=1
)

# Convert to TFLite with compatibility settings
print("📱 Converting working sentence encoder to TFLite...")
converter = tf.lite.TFLiteConverter.from_keras_model(encoder_model)

# Use compatibility settings that we know work
converter.optimizations = []  # No optimizations for maximum compatibility
converter.target_spec.supported_types = [tf.float32]
converter.experimental_new_converter = False  # Use old converter

working_encoder_tflite = converter.convert()

# Save the working TFLite model
with open('tflite_models/working_sentence_encoder.tflite', 'wb') as f:
    f.write(working_encoder_tflite)

print(f"✅ Working sentence encoder saved: {len(working_encoder_tflite)} bytes")

# Test the new encoder
print("\n🧪 TESTING WORKING ENCODER...")

# Test with TFLite model
interpreter = tf.lite.Interpreter(model_path='tflite_models/working_sentence_encoder.tflite')
interpreter.allocate_tensors()

input_details = interpreter.get_input_details()
output_details = interpreter.get_output_details()

test_commands = [
    "How many steps did I take today?",
    "Set an alarm for 7 AM",
    "What's the weather like?",
    "Turn on do not disturb",
    "Call mom"
]

print(f"🧪 Testing {len(test_commands)} commands...")

for i, command in enumerate(test_commands):
    # Prepare input
    words = command.lower().split()[:32]
    sequence = [vocabulary.get(word, 0) for word in words]
    sequence += [0] * (32 - len(sequence))
    sequence = np.array([sequence], dtype=np.float32)
    
    # Run TFLite inference
    interpreter.set_tensor(input_details[0]['index'], sequence)
    interpreter.invoke()
    tflite_embedding = interpreter.get_tensor(output_details[0]['index'])[0]
    
    # Test with intent classifier
    prediction = classifier.predict([tflite_embedding])[0]
    probabilities = classifier.predict_proba([tflite_embedding])[0]
    predicted_intent = intent_encoder.inverse_transform([prediction])[0]
    confidence = probabilities.max()
    
    print(f"\n📝 Command: '{command}'")
    print(f"🎯 Predicted: {predicted_intent} (confidence: {confidence:.3f})")
    
    # Show top 3 predictions
    top_indices = np.argsort(probabilities)[-3:][::-1]
    print(f"🏆 Top 3:")
    for j, idx in enumerate(top_indices):
        intent = intent_encoder.inverse_transform([idx])[0]
        prob = probabilities[idx]
        print(f"  {j+1}. {intent}: {prob:.3f}")

print("\n✅ Working sentence encoder created!")
print("🚀 This should give much better and varied predictions!")
print("📱 Copy 'working_sentence_encoder.tflite' to Android to test.")