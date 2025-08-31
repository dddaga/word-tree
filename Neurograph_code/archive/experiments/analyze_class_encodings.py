#!/usr/bin/env python3
"""
Analyze class encodings and prediction logic to understand why the model
is biased toward certain digits (1 and 7 getting 80% of predictions).
"""

import torch
import numpy as np
from modules.class_encoding import generate_digit_class_encodings
from modules.output_adapters import predict_label_from_output
from core.tables import ExtendedLookupTableModule
from utils.config import load_config
import matplotlib.pyplot as plt
from torch.nn.functional import cosine_similarity


def analyze_class_encodings():
    """Analyze the separability and characteristics of class encodings"""
    
    cfg = load_config("config/optimized.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    print("="*80)
    print("🔍 CLASS ENCODING ANALYSIS")
    print("="*80)
    
    # Generate class encodings
    class_phase_encodings, class_mag_encodings = generate_digit_class_encodings(
        num_classes=10,
        vector_dim=cfg["vector_dim"],
        phase_bins=cfg["phase_bins"],
        mag_bins=cfg["mag_bins"],
        seed=cfg.get("seed", 42)
    )
    
    # Create lookup table
    lookup = ExtendedLookupTableModule(cfg["phase_bins"], cfg["mag_bins"], device=device)
    
    print(f"📊 Encoding Parameters:")
    print(f"   Vector dimension: {cfg['vector_dim']}")
    print(f"   Phase bins: {cfg['phase_bins']}")
    print(f"   Magnitude bins: {cfg['mag_bins']}")
    print(f"   Random seed: {cfg.get('seed', 42)}")
    
    # Convert to actual vectors
    class_vectors = {}
    for digit in range(10):
        phase_vec = lookup.lookup_phase(class_phase_encodings[digit])
        mag_vec = lookup.lookup_magnitude(class_mag_encodings[digit])
        signal_vec = phase_vec * mag_vec
        class_vectors[digit] = signal_vec
        
        print(f"\n🔢 Digit {digit}:")
        print(f"   Phase indices: {class_phase_encodings[digit].tolist()}")
        print(f"   Mag indices:   {class_mag_encodings[digit].tolist()}")
        print(f"   Signal vector: {signal_vec.detach().cpu().numpy()}")
        print(f"   Vector norm:   {torch.norm(signal_vec).item():.4f}")
    
    # Analyze pairwise similarities
    print(f"\n" + "="*60)
    print("📐 PAIRWISE COSINE SIMILARITIES")
    print("="*60)
    
    similarity_matrix = np.zeros((10, 10))
    
    for i in range(10):
        for j in range(10):
            if i == j:
                similarity_matrix[i][j] = 1.0
            else:
                sim = cosine_similarity(class_vectors[i], class_vectors[j], dim=0).item()
                similarity_matrix[i][j] = sim
    
    # Print similarity matrix
    print("     ", end="")
    for j in range(10):
        print(f"{j:6d}", end="")
    print()
    
    for i in range(10):
        print(f"{i:2d}: ", end="")
        for j in range(10):
            if i == j:
                print("  1.00", end="")
            else:
                print(f"{similarity_matrix[i][j]:6.3f}", end="")
        print()
    
    # Find most and least similar pairs
    max_sim = -1
    min_sim = 1
    max_pair = None
    min_pair = None
    
    for i in range(10):
        for j in range(i+1, 10):
            sim = similarity_matrix[i][j]
            if sim > max_sim:
                max_sim = sim
                max_pair = (i, j)
            if sim < min_sim:
                min_sim = sim
                min_pair = (i, j)
    
    print(f"\n📈 Most similar pair: Digits {max_pair[0]} and {max_pair[1]} (similarity: {max_sim:.4f})")
    print(f"📉 Least similar pair: Digits {min_pair[0]} and {min_pair[1]} (similarity: {min_sim:.4f})")
    
    # Calculate average similarities
    off_diagonal = []
    for i in range(10):
        for j in range(10):
            if i != j:
                off_diagonal.append(similarity_matrix[i][j])
    
    avg_similarity = np.mean(off_diagonal)
    std_similarity = np.std(off_diagonal)
    
    print(f"📊 Average inter-class similarity: {avg_similarity:.4f} ± {std_similarity:.4f}")
    
    if avg_similarity > 0.5:
        print("⚠️  WARNING: High average similarity - classes may be hard to distinguish")
    elif avg_similarity < 0.1:
        print("✅ GOOD: Low average similarity - classes are well separated")
    else:
        print("📝 MODERATE: Average similarity - some confusion expected")
    
    # Plot similarity matrix
    plt.figure(figsize=(10, 8))
    plt.imshow(similarity_matrix, cmap='coolwarm', vmin=-1, vmax=1)
    plt.colorbar(label='Cosine Similarity')
    plt.title('Class Encoding Similarity Matrix')
    plt.xlabel('Digit Class')
    plt.ylabel('Digit Class')
    plt.xticks(range(10))
    plt.yticks(range(10))
    
    # Add text annotations
    for i in range(10):
        for j in range(10):
            if i == j:
                plt.text(j, i, '1.00', ha='center', va='center', color='black', fontweight='bold')
            else:
                color = 'white' if abs(similarity_matrix[i][j]) > 0.5 else 'black'
                plt.text(j, i, f'{similarity_matrix[i][j]:.2f}', ha='center', va='center', color=color)
    
    plt.tight_layout()
    plt.savefig('logs/optimized/class_similarity_matrix.png', dpi=150, bbox_inches='tight')
    print(f"\n📊 Similarity matrix plot saved to: logs/optimized/class_similarity_matrix.png")
    
    return class_vectors, similarity_matrix


def test_prediction_bias():
    """Test if the prediction function has inherent bias"""
    
    print(f"\n" + "="*80)
    print("🎯 PREDICTION BIAS ANALYSIS")
    print("="*80)
    
    cfg = load_config("config/optimized.yaml")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    
    # Generate class encodings
    class_phase_encodings, class_mag_encodings = generate_digit_class_encodings(
        num_classes=10,
        vector_dim=cfg["vector_dim"],
        phase_bins=cfg["phase_bins"],
        mag_bins=cfg["mag_bins"],
        seed=cfg.get("seed", 42)
    )
    
    # Convert to the format expected by predict_label_from_output
    class_encodings = {}
    for digit in range(10):
        class_encodings[digit] = (class_phase_encodings[digit], class_mag_encodings[digit])
    
    lookup = ExtendedLookupTableModule(cfg["phase_bins"], cfg["mag_bins"], device=device)
    
    # Create a mock activation table that returns each class encoding
    class MockActivationTable:
        def __init__(self, target_digit, class_encodings):
            self.target_digit = target_digit
            self.class_encodings = class_encodings
            self.table = {}
            
        def is_active(self, node_id):
            return True  # All nodes active
            
        def get_active_context(self):
            return {f'n{40+i}': None for i in range(10)}  # Mock output nodes
    
    # Test prediction when activation exactly matches each class
    print("🧪 Testing prediction accuracy when output exactly matches class encoding:")
    
    output_nodes = {f'n{40+i}' for i in range(10)}
    correct_predictions = 0
    
    for target_digit in range(10):
        # Create mock activation that returns the exact class encoding
        mock_activation = MockActivationTable(target_digit, class_encodings)
        
        # Set all output nodes to return the target class encoding
        for node_id in output_nodes:
            mock_activation.table[node_id] = (
                class_phase_encodings[target_digit],
                class_mag_encodings[target_digit],
                None  # strength not used
            )
        
        # Predict
        predicted = predict_label_from_output(mock_activation, output_nodes, class_encodings, lookup)
        
        print(f"   Target: {target_digit}, Predicted: {predicted}, {'✅' if predicted == target_digit else '❌'}")
        
        if predicted == target_digit:
            correct_predictions += 1
    
    accuracy = correct_predictions / 10
    print(f"\n📊 Perfect encoding prediction accuracy: {accuracy:.1%} ({correct_predictions}/10)")
    
    if accuracy < 1.0:
        print("❌ CRITICAL: Prediction function fails even with perfect class encodings!")
        print("   This indicates a fundamental issue with the prediction logic.")
    else:
        print("✅ Prediction function works correctly with perfect encodings.")
    
    # Test with random vectors
    print(f"\n🎲 Testing with random vectors:")
    
    random_correct = 0
    for _ in range(100):
        # Create random activation
        mock_activation = MockActivationTable(0, class_encodings)
        for node_id in output_nodes:
            random_phase = torch.randint(0, cfg["phase_bins"], (cfg["vector_dim"],))
            random_mag = torch.randint(0, cfg["mag_bins"], (cfg["vector_dim"],))
            mock_activation.table[node_id] = (random_phase, random_mag, None)
        
        predicted = predict_label_from_output(mock_activation, output_nodes, class_encodings, lookup)
        # For random vectors, we just check if prediction is valid
        if 0 <= predicted <= 9:
            random_correct += 1
    
    print(f"   Valid predictions with random vectors: {random_correct}/100")
    
    return accuracy


def suggest_improvements(class_vectors, similarity_matrix):
    """Suggest improvements based on analysis"""
    
    print(f"\n" + "="*80)
    print("💡 IMPROVEMENT SUGGESTIONS")
    print("="*80)
    
    avg_similarity = np.mean([similarity_matrix[i][j] for i in range(10) for j in range(10) if i != j])
    
    if avg_similarity > 0.3:
        print("🔧 HIGH PRIORITY - Class Encoding Issues:")
        print("   1. Class encodings are too similar (avg similarity > 0.3)")
        print("   2. Try orthogonal encoding generation instead of random")
        print("   3. Increase vector dimension for better separation")
        print("   4. Use structured encodings (e.g., one-hot style)")
        
    print("\n🎯 PREDICTION METHOD Issues:")
    print("   1. Current method averages all active output nodes")
    print("   2. Try using the most confident single node instead")
    print("   3. Weight nodes by activation strength")
    print("   4. Use voting mechanism across output nodes")
    
    print("\n📈 TRAINING Issues:")
    print("   1. All output nodes learn the same target for each sample")
    print("   2. Try specialized output nodes (each learns specific digits)")
    print("   3. Increase learning rate further (0.1, 0.2)")
    print("   4. Add regularization to encourage node specialization")
    
    # Find which digits are most confused
    most_similar_pairs = []
    for i in range(10):
        for j in range(i+1, 10):
            most_similar_pairs.append((similarity_matrix[i][j], i, j))
    
    most_similar_pairs.sort(reverse=True)
    
    print(f"\n⚠️  Most Confusable Digit Pairs:")
    for sim, i, j in most_similar_pairs[:5]:
        print(f"   Digits {i} and {j}: similarity {sim:.3f}")


if __name__ == "__main__":
    torch.manual_seed(42)
    
    # Run analysis
    class_vectors, similarity_matrix = analyze_class_encodings()
    prediction_accuracy = test_prediction_bias()
    suggest_improvements(class_vectors, similarity_matrix)
    
    print(f"\n" + "="*80)
    print("🏁 ANALYSIS COMPLETE")
    print("="*80)
    print("Key findings will help guide the next round of improvements.")
