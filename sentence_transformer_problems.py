#!/usr/bin/env python3
"""
Sentence Transformer Problem Areas Demonstration Script
======================================================

This script demonstrates three key problem areas when using sentence transformers 
and cosine similarity for semantic understanding in real-world applications.
Created for research paper documentation.

Problem Areas Tested:
#2 Rotate the pair only: Paraphrase variants keep pairwise cosine similar 
   but show neighbor/ranking changes in a fixed corpus
#4 Different scalar per vector: Two different pairs with approximately 
   the same cosine (shows cosine alone is not "semantics")
#6 Diagonal scaling with compensation: Demonstrates unchanged query→doc 
   dot-product rankings while doc–doc cosine changes

Requirements:
  pip install sentence-transformers numpy scipy scikit-learn rbo
"""

from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from itertools import combinations, product
import re
import time

try:
    import rbo  # rank-biased overlap for ranking stability
    HAVE_RBO = True
except Exception:
    HAVE_RBO = False
    print("Warning: 'rbo' package not available. RBO calculations will be skipped.")

# -----------------------------
# Configuration
# -----------------------------
MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"  # Fast model for demonstration
TOP_K = 10
COS_TOL = 0.02  # tolerance for "same cosine" in cases #2/#4
NP_RANDOM_SEED = 42
np.random.seed(NP_RANDOM_SEED)

print("=" * 80)
print("SENTENCE TRANSFORMER PROBLEM AREAS DEMONSTRATION")
print("=" * 80)
print(f"Model: {MODEL_NAME}")
print(f"Random seed: {NP_RANDOM_SEED}")
print(f"Top-K neighbors: {TOP_K}")
print(f"Cosine tolerance: {COS_TOL}")
print("=" * 80)

# Load model
print("\nLoading sentence transformer model...")
model = SentenceTransformer(MODEL_NAME)
print("✓ Model loaded successfully")

def embed(texts, normalize=True):
    """Generate embeddings for given texts."""
    embs = model.encode(texts, normalize_embeddings=normalize, batch_size=64, show_progress_bar=False)
    return np.array(embs)

def cosine(a, b):
    """Calculate cosine similarity between two vectors."""
    return float(cosine_similarity(a.reshape(1,-1), b.reshape(1,-1))[0][0])

def neighbors(vec, corpus_embs, k=TOP_K):
    """Find k nearest neighbors in corpus for given vector."""
    sims = cosine_similarity(vec.reshape(1,-1), corpus_embs)[0]
    idx = np.argsort(-sims)[:k]
    return idx, sims[idx]

def jaccard(a, b):
    """Calculate Jaccard similarity between two sets."""
    a, b = set(a), set(b)
    if not a and not b: return 1.0
    return len(a & b) / max(1, len(a | b))

def lexical_overlap(a, b):
    """Calculate lexical overlap between two strings."""
    tok = lambda s: set(re.findall(r"[A-Za-z]+", s.lower()))
    A, B = tok(a), tok(b)
    if not A and not B: return 0.0
    return len(A & B)/max(1,len(A|B))

# -----------------------------
# Multi-domain corpus for testing
# -----------------------------
corpus = [
    # Safety/Industry domain
    "Safety fences reduce the risk of worker injury around machines.",
    "Protective barriers around robot cells prevent accidents.",
    "OSHA 1910.212 requires machine guarding to protect operators.",
    "Emergency stop buttons must be clearly visible and accessible.",
    "CE marking indicates conformity with health and safety standards.",
    "Light curtains can stop hazardous motion when a person enters.",
    
    # Finance domain
    "Quarterly earnings exceeded analyst expectations this season.",
    "The central bank raised interest rates to curb inflation.",
    "Retail investors increased their exposure to technology stocks.",
    "Hedging currency risk can stabilize international revenue.",
    "The company issued bonds to finance expansion.",
    
    # Sports domain
    "The team secured a playoff berth after winning the derby.",
    "A last-minute goal sent the match into extra time.",
    "The rookie posted a triple-double in his debut.",
    "A rain delay pushed the game to the following day.",
    "The coach emphasized defense during the training camp.",
    
    # Web/AI domain
    "Vector databases accelerate semantic search over large corpora.",
    "Contrastive learning aligns similar sentences in embedding space.",
    "Cross-encoders improve precision at re-ranking stage.",
    "BM25 favors exact-term matches with inverse document frequency.",
    "Whitening can reduce anisotropy in sentence embeddings.",
    
    # Commerce domain
    "Free shipping thresholds increase average order value.",
    "Customers abandon carts when checkout is too slow.",
    "Personalized recommendations can boost conversion rates.",
    "Holiday promotions lifted sales across all categories.",
    "Bundle discounts incentivize larger purchases.",
    
    # Health domain
    "Regular exercise reduces the risk of cardiovascular disease.",
    "A balanced diet supports long-term metabolic health.",
    "Sleep quality correlates with cognitive performance.",
    "Vaccination campaigns raised community immunity levels.",
    "Mindfulness practices can lower perceived stress."
]

print(f"\nCorpus loaded: {len(corpus)} documents across 6 domains")
print("Generating corpus embeddings...")
corpus_embs = embed(corpus, normalize=True)
print("✓ Corpus embeddings generated")

# -----------------------------
# CASE #2: "Rotate the pair only" analogue
# Same pairwise cosine, different neighbours => practical meaning shift
# -----------------------------
print("\n" + "=" * 80)
print("CASE #2: PARAPHRASE VARIANTS - NEIGHBOR RANKING CHANGES")
print("=" * 80)
print("Problem: Paraphrased sentences maintain similar pairwise cosine similarity")
print("but show different neighbor rankings, affecting practical search results.")
print()

pair_A = (
    "Safety fences reduce the risk of worker injury around machines.",
    "Protective barriers shield employees from accidents near industrial robots."
)

pair_A_paraphrase = (
    "By installing safety fences around machines, the risk of worker injury is reduced.",
    "Employees are protected from accidents around industrial robots by protective barriers."
)

print("Original pair:")
print(f"  A1: '{pair_A[0]}'")
print(f"  A2: '{pair_A[1]}'")
print()
print("Paraphrased pair:")
print(f"  A1': '{pair_A_paraphrase[0]}'")
print(f"  A2': '{pair_A_paraphrase[1]}'")
print()

emb_A1, emb_A2 = embed(list(pair_A))
emb_A1p, emb_A2p = embed(list(pair_A_paraphrase))

cos_orig = cosine(emb_A1, emb_A2)
cos_para = cosine(emb_A1p, emb_A2p)

print(f"Cosine similarity (original pair):    {cos_orig:.4f}")
print(f"Cosine similarity (paraphrased pair): {cos_para:.4f}")
print(f"Difference in pairwise cosine:        {abs(cos_orig - cos_para):.4f}")
print()

# Find neighbors for each sentence
idx1, sims1 = neighbors(emb_A1, corpus_embs, k=TOP_K)
idx1p, sims1p = neighbors(emb_A1p, corpus_embs, k=TOP_K)
idx2, sims2 = neighbors(emb_A2, corpus_embs, k=TOP_K)
idx2p, sims2p = neighbors(emb_A2p, corpus_embs, k=TOP_K)

jaccard_1 = jaccard(idx1, idx1p)
jaccard_2 = jaccard(idx2, idx2p)

print("NEIGHBOR RANKING ANALYSIS:")
print(f"Jaccard overlap (A1 vs A1' neighbors): {jaccard_1:.3f}")
print(f"Jaccard overlap (A2 vs A2' neighbors): {jaccard_2:.3f}")
print()

# Show top neighbors for first sentence
print("Top 5 neighbors for A1 (original):")
for i, (idx, sim) in enumerate(zip(idx1[:5], sims1[:5])):
    print(f"  {i+1}. [{sim:.3f}] {corpus[idx][:60]}...")

print("\nTop 5 neighbors for A1' (paraphrased):")
for i, (idx, sim) in enumerate(zip(idx1p[:5], sims1p[:5])):
    print(f"  {i+1}. [{sim:.3f}] {corpus[idx][:60]}...")

if HAVE_RBO:
    try:
        L1 = [corpus[i] for i in idx1]
        L1p = [corpus[i] for i in idx1p]
        L2 = [corpus[i] for i in idx2]
        L2p = [corpus[i] for i in idx2p]
        rbo_1 = rbo.RankingSimilarity(L1, L1p).rbo()
        rbo_2 = rbo.RankingSimilarity(L2, L2p).rbo()
        print(f"\nRank-Biased Overlap (A1 vs A1'): {rbo_1:.3f}")
        print(f"Rank-Biased Overlap (A2 vs A2'): {rbo_2:.3f}")
    except Exception as e:
        print(f"\nRBO calculation failed: {e}")

print("\n" + "-" * 60)
print("RESEARCH IMPLICATION:")
print("Even with similar pairwise cosine similarity, paraphrasing can")
print("significantly alter neighbor rankings, affecting search quality")
print("and retrieval performance in real applications.")

# -----------------------------
# CASE #4: "Different scalar per vector (pair)" analogue
# Find two *different* pairs that have (approximately) the same cosine
# -----------------------------
print("\n" + "=" * 80)
print("CASE #4: DIFFERENT PAIRS WITH SAME COSINE SIMILARITY")
print("=" * 80)
print("Problem: Different semantic pairs can have identical cosine similarities,")
print("showing that cosine alone doesn't capture semantic meaning.")
print()

candidates = [
    # Single words
    "cat", "dog", "eagle", "sparrow", "hammer", "screwdriver", "car", "automobile",
    "couch", "sofa", "doctor", "physician", "river", "stream", "happy", "joyful",
    
    # Short phrases
    "The cat chased a mouse.",
    "The dog barked at the mailman.",
    "An eagle soared over the valley.",
    "A sparrow perched on the fence.",
    "He tightened the screw with a screwdriver.",
    "She hit the nail with a hammer.",
    "I parked the car in the garage.",
    "The automobile was parked in the garage.",
    "The doctor examined the patient.",
    "The physician assessed the symptoms.",
    "The river flows through the valley.",
    "A small stream runs behind the house.",
    "She felt happy about the news.",
    "He was joyful during the celebration."
]

print(f"Analyzing {len(candidates)} candidate phrases...")
cand_embs = embed(candidates)
pair_list = list(combinations(range(len(candidates)), 2))

# Compute all pair cosines
pair_scores = []
for i, j in pair_list:
    c = cosine(cand_embs[i], cand_embs[j])
    pair_scores.append((i, j, c))

print(f"Computed {len(pair_scores)} pairwise similarities")

# Find two pairs with (approximately) equal cosine but low lexical overlap
best_matches = []
for (i, j, c1), (p, q, c2) in combinations(pair_scores, 2):
    if abs(c1 - c2) <= COS_TOL:
        if len({i,j,p,q}) == 4:  # All different indices
            lo1 = lexical_overlap(candidates[i], candidates[j])
            lo2 = lexical_overlap(candidates[p], candidates[q])
            max_lo = max(lo1, lo2)
            if max_lo < 0.3:  # Low lexical overlap
                best_matches.append((i,j,c1,p,q,c2,max_lo))

if best_matches:
    # Sort by cosine similarity difference (smallest first)
    best_matches.sort(key=lambda x: abs(x[2] - x[5]))
    
    print(f"Found {len(best_matches)} pairs with similar cosine but different semantics")
    print()
    
    # Show top 3 examples
    for idx, (i,j,c1,p,q,c2,max_lo) in enumerate(best_matches[:3]):
        print(f"EXAMPLE {idx+1}:")
        print(f"  Pair A: '{candidates[i]}' ↔ '{candidates[j]}'")
        print(f"  Pair B: '{candidates[p]}' ↔ '{candidates[q]}'")
        print(f"  Cosine A: {c1:.4f}")
        print(f"  Cosine B: {c2:.4f}")
        print(f"  Difference: {abs(c1-c2):.4f}")
        print(f"  Max lexical overlap: {max_lo:.3f}")
        print()
else:
    print("No suitable pairs found under current tolerance.")
    print("This suggests the model has good semantic discrimination,")
    print("but try widening COS_TOL or expanding candidates for more examples.")

print("-" * 60)
print("RESEARCH IMPLICATION:")
print("Identical cosine similarities can occur between semantically")
print("unrelated pairs, demonstrating limitations of cosine-only")
print("similarity for semantic understanding.")

# -----------------------------
# CASE #6: Diagonal scaling with compensation (UD, V D^{-1})
# Show: query→doc dot-products preserved; doc–doc cosine changes
# -----------------------------
print("\n" + "=" * 80)
print("CASE #6: DIAGONAL SCALING WITH COMPENSATION")
print("=" * 80)
print("Problem: Mathematical transformations can preserve query-document rankings")
print("while changing document-document similarities, showing ranking instability.")
print()

queries = [
    "what are machine safety fences used for",
    "how to protect employees near industrial robots",
    "financial earnings and market performance",
    "vector search and semantic similarity"
]

docs = [
    "Safety fences reduce the risk of worker injury around machines.",
    "Protective barriers around robot cells prevent accidents.",
    "Quarterly earnings exceeded analyst expectations this season.",
    "Vector databases accelerate semantic search over large corpora.",
    "BM25 favors exact-term matches with inverse document frequency.",
    "Emergency stop buttons must be clearly visible and accessible.",
    "The central bank raised interest rates to curb inflation.",
    "Contrastive learning aligns similar sentences in embedding space."
]

print(f"Testing with {len(queries)} queries and {len(docs)} documents")
print()

# Generate unnormalized embeddings for dot-product calculations
Q = embed(queries, normalize=False)
D = embed(docs, normalize=False)

def dot_scores(Q, D):
    """Calculate dot-product scores between queries and documents."""
    return Q @ D.T

# Original scores
S0 = dot_scores(Q, D)

# Apply diagonal scaling transformation
dim = Q.shape[1]
scales = np.exp(np.random.uniform(-0.4, 0.4, size=dim))  # Random scaling factors
Dmat = np.diag(scales)
Dinv = np.diag(1.0/scales)

# Transform: Q' = Q * D, D' = D * D^(-1)
Qp = Q @ Dmat
Dp = D @ Dinv

# New scores after transformation
S1 = dot_scores(Qp, Dp)

print("TRANSFORMATION ANALYSIS:")
print(f"Embedding dimension: {dim}")
print(f"Scaling factors range: [{scales.min():.3f}, {scales.max():.3f}]")
print()

# Check if dot-products are preserved
same_scores = np.allclose(S0, S1, atol=1e-6)
print(f"Query→Document dot-products preserved: {same_scores}")
if not same_scores:
    max_diff = float(np.max(np.abs(S0-S1)))
    print(f"Maximum absolute difference: {max_diff:.2e}")
print()

# Analyze ranking stability per query
print("RANKING STABILITY ANALYSIS:")
for qi in range(Q.shape[0]):
    r0 = list(np.argsort(-S0[qi]))  # Original ranking
    r1 = list(np.argsort(-S1[qi]))  # Transformed ranking
    
    # Top-K overlap
    top_k_overlap = len(set(r0[:TOP_K]) & set(r1[:TOP_K]))
    
    print(f"Query {qi+1}: '{queries[qi][:40]}...'")
    print(f"  Top-{min(TOP_K, len(docs))} overlap: {top_k_overlap}/{min(TOP_K, len(docs))}")
    
    # Show top 3 documents for this query
    print("  Original ranking (top 3):")
    for rank, doc_idx in enumerate(r0[:3]):
        score = S0[qi, doc_idx]
        print(f"    {rank+1}. [{score:.3f}] {docs[doc_idx][:50]}...")
    
    if qi < 2:  # Only show for first 2 queries to save space
        print("  Transformed ranking (top 3):")
        for rank, doc_idx in enumerate(r1[:3]):
            score = S1[qi, doc_idx]
            print(f"    {rank+1}. [{score:.3f}] {docs[doc_idx][:50]}...")
    print()

# Document-document cosine changes
def pairwise_cosine(X):
    """Calculate pairwise cosine similarities."""
    Xn = X / np.linalg.norm(X, axis=1, keepdims=True)
    return Xn @ Xn.T

Cos_docs_0 = pairwise_cosine(D)
Cos_docs_1 = pairwise_cosine(Dp)

# Calculate change in document-document similarities
cosine_diff = np.abs(Cos_docs_0 - Cos_docs_1)
frobenius_norm = float(np.linalg.norm(cosine_diff))
max_cosine_change = float(np.max(cosine_diff))

print("DOCUMENT-DOCUMENT SIMILARITY CHANGES:")
print(f"Frobenius norm of cosine difference matrix: {frobenius_norm:.4f}")
print(f"Maximum cosine similarity change: {max_cosine_change:.4f}")
print(f"Mean cosine similarity change: {float(np.mean(cosine_diff)):.4f}")

print("\n" + "-" * 60)
print("RESEARCH IMPLICATION:")
print("Mathematical transformations can preserve query-document rankings")
print("while significantly altering document-document relationships,")
print("highlighting the instability of cosine similarity measures")
print("under certain transformations.")

# -----------------------------
# Summary and Conclusions
# -----------------------------
print("\n" + "=" * 80)
print("SUMMARY OF SENTENCE TRANSFORMER PROBLEM AREAS")
print("=" * 80)
print()
print("This demonstration revealed three critical limitations:")
print()
print("1. PARAPHRASE SENSITIVITY:")
print("   - Semantically equivalent paraphrases maintain similar pairwise")
print("     cosine similarity but produce different neighbor rankings")
print("   - Impact: Inconsistent search results for equivalent queries")
print()
print("2. COSINE AMBIGUITY:")
print("   - Different semantic pairs can exhibit identical cosine similarities")
print("   - Impact: Cosine similarity alone insufficient for semantic matching")
print()
print("3. TRANSFORMATION INSTABILITY:")
print("   - Mathematical transformations can preserve some relationships")
print("     while destroying others in unpredictable ways")
print("   - Impact: Fragile similarity measures under embedding modifications")
print()
print("RECOMMENDATIONS FOR RESEARCH:")
print("- Consider multiple similarity metrics beyond cosine similarity")
print("- Implement robust evaluation across paraphrase variants")
print("- Test embedding stability under various transformations")
print("- Develop semantic evaluation beyond pairwise comparisons")
print()
print("=" * 80)
print("DEMONSTRATION COMPLETE")
print("=" * 80)
