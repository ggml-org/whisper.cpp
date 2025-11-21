#!/usr/bin/env python3
"""
Test TFLite models directly to compare with Android behavior
"""

import json
import numpy as np
import tensorflow as tf

def test_tflite_models():
    print("🔍 TESTING TFLITE MODELS DIRECTLY")
    print("=" * 50)
    
    # Load TFLite models
    print("📱 Loading TFLite models...")
    
    # Load sentence encoder
    sentence_interpreter = tf.lite.Interpreter(model_path='tflite_models/lightweight_sentence_encoder.tflite')
    sentence_interpreter.allocate_tensors()
    
    # Load intent classifier
    intent_interpreter = tf.lite.Interpreter(model_path='tflite_models/intent_classifier.tflite')
    intent_interpreter.allocate_tensors()
    
    # Load vocabulary and metadata
    with open('tflite_models/vocabulary.json', 'r') as f:
        vocabulary = json.load(f)
    
    with open('tflite_models/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    intent_mapping = {int(k): v for k, v in metadata['intent_mapping'].items()}
    
    print(f"📝 Vocabulary size: {len(vocabulary)}")
    print(f"🎯 Intent classes: {len(intent_mapping)}")
    
    # Test the same preprocessing as Android
    def text_to_sequence(text, vocab, max_len=32):
        """Same preprocessing as Android"""
        processed_text = text.lower().strip()
        words = processed_text.split()
        
        sequence = np.zeros(max_len, dtype=np.int32)
        known_words = 0
        
        for i, word in enumerate(words[:max_len]):
            token_id = vocab.get(word, 0)
            sequence[i] = token_id
            if token_id != 0:
                known_words += 1
        
        return sequence, known_words, len(words)
    
    def get_embeddings(sequence):
        """Get embeddings from TFLite sentence encoder"""
        input_details = sentence_interpreter.get_input_details()
        output_details = sentence_interpreter.get_output_details()
        
        # Set input
        input_data = np.array([sequence], dtype=np.int32)
        sentence_interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        sentence_interpreter.invoke()
        
        # Get output
        embeddings = sentence_interpreter.get_tensor(output_details[0]['index'])
        return embeddings[0]
    
    def classify_embeddings(embeddings):
        """Classify embeddings using TFLite intent classifier"""
        input_details = intent_interpreter.get_input_details()
        output_details = intent_interpreter.get_output_details()
        
        # Set input
        input_data = np.array([embeddings], dtype=np.float32)
        intent_interpreter.set_tensor(input_details[0]['index'], input_data)
        
        # Run inference
        intent_interpreter.invoke()
        
        # Get output
        probabilities = intent_interpreter.get_tensor(output_details[0]['index'])
        return probabilities[0]
    
    # Test with the same commands that are failing
    test_commands = [
        "How many steps did I take today?",
        "Set an alarm for 7 AM", 
        "What's the weather like?",
        "Start a workout",
        "Show me my heart rate",
        "Set my daily step goal to 10000",
        "Turn on do not disturb",
        "Call mom"
    ]
    
    print("\n🧪 TESTING TFLITE MODELS:")
    print("-" * 40)
    
    for command in test_commands:
        print(f"\n📝 Testing: '{command}'")
        
        # Step 1: Text to sequence (same as Android)
        sequence, known_words, total_words = text_to_sequence(command, vocabulary)
        print(f"  🔤 Words: {command.lower().split()}")
        print(f"  🔤 Known words: {known_words}/{total_words} ({known_words/total_words*100:.1f}%)")
        print(f"  🔤 Sequence: {sequence[:total_words]}")
        
        # Step 2: Get embeddings
        embeddings = get_embeddings(sequence)
        print(f"  📊 Embeddings: mean={embeddings.mean():.6f}, std={embeddings.std():.6f}")
        print(f"  📊 First 10: {embeddings[:10]}")
        
        # Step 3: Classify
        probabilities = classify_embeddings(embeddings)
        
        # Get top prediction
        best_index = np.argmax(probabilities)
        best_intent = intent_mapping[best_index]
        confidence = probabilities[best_index]
        
        print(f"  🎯 Predicted: {best_intent} (confidence: {confidence:.3f})")
        
        # Show top 3 predictions
        top_indices = np.argsort(probabilities)[-3:][::-1]
        print(f"  🏆 Top 3:")
        for i, idx in enumerate(top_indices):
            intent = intent_mapping[idx]
            prob = probabilities[idx]
            print(f"    {i+1}. {intent}: {prob:.3f}")
    
    print("\n🔍 CHECKING FOR POTENTIAL ISSUES:")
    print("-" * 40)
    
    # Test if all embeddings are similar (potential issue)
    test_embeddings = []
    for command in test_commands[:3]:
        sequence, _, _ = text_to_sequence(command, vocabulary)
        embeddings = get_embeddings(sequence)
        test_embeddings.append(embeddings)
    
    # Calculate similarities
    for i in range(len(test_embeddings)):
        for j in range(i+1, len(test_embeddings)):
            similarity = np.dot(test_embeddings[i], test_embeddings[j])
            print(f"  Similarity between command {i+1} and {j+1}: {similarity:.3f}")
    
    # Test with zero sequence (all unknown words)
    print(f"\n🔍 Testing with zero sequence (all unknown words):")
    zero_sequence = np.zeros(32, dtype=np.int32)
    zero_embeddings = get_embeddings(zero_sequence)
    zero_probabilities = classify_embeddings(zero_embeddings)
    zero_best = intent_mapping[np.argmax(zero_probabilities)]
    print(f"  Zero sequence predicts: {zero_best} (confidence: {zero_probabilities.max():.3f})")
    
    print("\n✅ TFLITE MODEL TESTING COMPLETED!")

if __name__ == "__main__":
    test_tflite_models()