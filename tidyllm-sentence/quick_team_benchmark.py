#!/usr/bin/env python3
"""
QUICK TEAM BENCHMARK: Show TLM team advantage in action
Compare single-algorithm vs team-algorithm performance
"""

import sys
import time
sys.path.insert(0, '../tlm')

import tidyllm_sentence as tls
import tlm
from tlm_team_advantage import TidyllmTeamEmbeddings

# Test sentences for semantic similarity
test_cases = [
    # Similar pairs (should have high similarity)
    ("Machine learning algorithms for classification", "AI classification methods and techniques"),
    ("Deep neural networks and architectures", "Neural network models for deep learning"),  
    ("Database systems and query optimization", "SQL database performance optimization"),
    ("Natural language processing techniques", "NLP methods for text analysis"),
    
    # Different pairs (should have low similarity)
    ("Machine learning classification", "Database query optimization"),
    ("Neural networks", "Cybersecurity protection"),
    ("Text processing", "Image recognition"),
]

def benchmark_single_algorithm():
    """Test standard TF-IDF performance."""
    print("SINGLE ALGORITHM: Standard TF-IDF")
    print("-" * 40)
    
    all_sentences = []
    for sent1, sent2 in test_cases:
        all_sentences.extend([sent1, sent2])
    
    start_time = time.time()
    embeddings, model = tls.tfidf_fit_transform(all_sentences)
    normalized = tlm.l2_normalize(embeddings)
    embedding_time = time.time() - start_time
    
    similarities = []
    for i in range(0, len(normalized), 2):
        sim = tls.cosine_similarity(normalized[i], normalized[i+1])
        similarities.append(sim)
    
    print(f"Embedding time: {embedding_time:.3f}s")
    print(f"Vocabulary size: {model.vocab_size}")
    print(f"Dimensions: {len(normalized[0])}")
    
    return similarities, embedding_time

def benchmark_team_algorithm():
    """Test TLM team approach."""
    print("\nTEAM ALGORITHM: Multi-algorithm ensemble")  
    print("-" * 40)
    
    all_sentences = []
    for sent1, sent2 in test_cases:
        all_sentences.extend([sent1, sent2])
    
    start_time = time.time()
    
    # Use TLM team embeddings
    team_embedder = TidyllmTeamEmbeddings(
        target_dims=32,  # Smaller for quick demo
        n_clusters=4,
        use_lsa=False,  # Skip LSA for speed  
        use_ngrams=False  # Skip n-grams for speed
    )
    
    embeddings, team_models = team_embedder.fit_transform(all_sentences)
    embedding_time = time.time() - start_time
    
    similarities = []
    for i in range(0, len(embeddings), 2):
        sim = tls.cosine_similarity(embeddings[i], embeddings[i+1])
        similarities.append(sim)
    
    print(f"Embedding time: {embedding_time:.3f}s")
    print(f"Algorithms used: {team_models['stats']['n_algorithms']}")
    print(f"Final dimensions: {team_models['stats']['final_dims']}")
    
    return similarities, embedding_time

def analyze_results(single_sims, team_sims):
    """Analyze and compare results."""
    print(f"\n" + "="*60)
    print("SEMANTIC SIMILARITY COMPARISON")
    print("="*60)
    
    print(f"{'Test Case':<45} {'Single':<8} {'Team':<8} {'Improve':<8}")
    print("-" * 60)
    
    improvements = []
    for i, (single_sim, team_sim) in enumerate(zip(single_sims, team_sims)):
        sent1, sent2 = test_cases[i]
        case_name = f"{sent1[:20]}... vs {sent2[:20]}..."
        improvement = team_sim - single_sim
        improvements.append(improvement)
        
        print(f"{case_name:<45} {single_sim:<8.3f} {team_sim:<8.3f} {improvement:<+8.3f}")
    
    avg_improvement = sum(improvements) / len(improvements)
    positive_improvements = sum(1 for imp in improvements if imp > 0)
    
    print("-" * 60)
    print(f"Average improvement: {avg_improvement:+.3f}")
    print(f"Positive improvements: {positive_improvements}/{len(improvements)} cases")
    
    if avg_improvement > 0:
        print(f"\n🏆 TLM TEAM WINS with {avg_improvement:.3f} average improvement!")
    else:
        print(f"\nSingle algorithm wins, but team provides other benefits...")
    
    return avg_improvement

if __name__ == "__main__":
    print("QUICK TLM TEAM BENCHMARK")
    print("Testing semantic similarity improvements")
    print(f"Test cases: {len(test_cases)}")
    
    # Benchmark both approaches
    single_sims, single_time = benchmark_single_algorithm()
    team_sims, team_time = benchmark_team_algorithm()
    
    # Analyze results
    improvement = analyze_results(single_sims, team_sims)
    
    # Summary
    print(f"\n" + "="*60)
    print("FINAL ANALYSIS")
    print("="*60)
    print(f"Single Algorithm Time: {single_time:.3f}s")
    print(f"Team Algorithm Time: {team_time:.3f}s") 
    print(f"Time Overhead: {team_time/single_time:.1f}x")
    print(f"Semantic Quality Gain: {improvement:+.3f}")
    
    if improvement > 0.1:  # Significant improvement
        print(f"\n🎯 CONCLUSION: TLM Team Advantage VALIDATED!")
        print(f"Worth the {team_time/single_time:.1f}x time cost for {improvement:.3f} quality gain")
    elif improvement > 0:
        print(f"\n✅ CONCLUSION: TLM Team shows promise")  
        print(f"Small quality gain with additional capabilities")
    else:
        print(f"\n⚖️ CONCLUSION: Trade-off scenario")
        print(f"Team provides transparency & flexibility vs pure performance")
    
    print(f"\nRemember: Team approach provides:")
    print(f"  • Complete algorithmic transparency")
    print(f"  • Hierarchical search capabilities") 
    print(f"  • Quality filtering via anomaly detection")
    print(f"  • Educational value & customizability")
    print(f"  • Potential for domain-specific optimization")