"""
Evaluation metrics for language model generation quality.
Implements both specific (training-data-dependent) and general (training-data-independent) metrics.
"""

import numpy as np
import pickle
import os
from collections import Counter, defaultdict
from typing import List, Dict, Tuple
import math


class EvaluationMetrics:
    """Compute various evaluation metrics for generated text."""
    
    def __init__(self, data_dir: str = 'data/shakespeare_char'):
        """
        Initialize with training data for specific metrics.
        
        Args:
            data_dir: Directory containing train.bin and meta.pkl
        """
        self.data_dir = data_dir
        self.train_text = None
        self.itos = None
        self.stoi = None
        self._load_training_data()
    
    def _load_training_data(self):
        """Load training data for computing specific metrics."""
        # Load meta information
        meta_path = os.path.join(self.data_dir, 'meta.pkl')
        if os.path.exists(meta_path):
            with open(meta_path, 'rb') as f:
                meta = pickle.load(f)
            self.itos = meta['itos']
            self.stoi = meta['stoi']
        
        # Load training data
        train_path = os.path.join(self.data_dir, 'train.bin')
        if os.path.exists(train_path):
            train_data = np.memmap(train_path, dtype=np.uint16, mode='r')
            # Convert to text
            self.train_text = ''.join([self.itos[int(i)] for i in train_data[:100000]])  # Use first 100k chars for efficiency
    
    # ============================================================================
    # SPECIFIC METRICS (compare to training data)
    # ============================================================================
    
    def character_ngram_overlap(self, generated_texts: List[str], n: int = 1) -> float:
        """
        Compute n-gram overlap between training and generated text.
        
        Args:
            generated_texts: List of generated text samples
            n: N-gram size (1=unigram, 2=bigram, 3=trigram)
        
        Returns:
            Overlap percentage (0-100): |train_ngrams ∩ gen_ngrams| / |train_ngrams|
        """
        if self.train_text is None:
            return 0.0
        
        # Extract n-grams from training data
        train_ngrams = self._extract_ngrams(self.train_text, n)
        
        # Extract n-grams from generated texts
        gen_text = ' '.join(generated_texts)
        gen_ngrams = self._extract_ngrams(gen_text, n)
        
        # Compute overlap
        if len(train_ngrams) == 0:
            return 0.0
        
        overlap = len(train_ngrams & gen_ngrams)
        overlap_percentage = (overlap / len(train_ngrams)) * 100
        
        return overlap_percentage
    
    def _extract_ngrams(self, text: str, n: int) -> set:
        """Extract unique n-grams from text."""
        ngrams = set()
        for i in range(len(text) - n + 1):
            ngrams.add(text[i:i+n])
        return ngrams
    
    def perplexity_from_loss(self, val_loss: float) -> float:
        """
        Compute perplexity from validation loss.
        
        Args:
            val_loss: Validation loss (cross-entropy)
        
        Returns:
            Perplexity = exp(val_loss)
        """
        return math.exp(val_loss)
    
    def kl_divergence(self, generated_texts: List[str]) -> float:
        """
        Compute KL divergence between training and generated character distributions.
        KL(train || generated)
        
        Args:
            generated_texts: List of generated text samples
        
        Returns:
            KL divergence value (lower is better)
        """
        if self.train_text is None:
            return float('inf')
        
        # Get character distributions
        train_dist = self._char_distribution(self.train_text)
        gen_text = ''.join(generated_texts)
        gen_dist = self._char_distribution(gen_text)
        
        # Compute KL divergence: KL(P||Q) = Σ P(x) log(P(x)/Q(x))
        kl_div = 0.0
        epsilon = 1e-10  # Smoothing to avoid log(0)
        
        all_chars = set(train_dist.keys()) | set(gen_dist.keys())
        
        for char in all_chars:
            p = train_dist.get(char, epsilon)
            q = gen_dist.get(char, epsilon)
            kl_div += p * math.log((p + epsilon) / (q + epsilon))
        
        return kl_div
    
    def _char_distribution(self, text: str) -> Dict[str, float]:
        """Compute normalized character frequency distribution."""
        counts = Counter(text)
        total = sum(counts.values())
        return {char: count / total for char, count in counts.items()}
    
    # ============================================================================
    # GENERAL METRICS (no training data needed)
    # ============================================================================
    
    def self_bleu(self, generated_texts: List[str], n: int = 4) -> float:
        """
        Compute Self-BLEU: average BLEU score of each sample against all others.
        Lower values indicate more diversity (less repetition).
        
        Args:
            generated_texts: List of generated text samples
            n: Maximum n-gram size for BLEU
        
        Returns:
            Average Self-BLEU score (0-100, lower is better for diversity)
        """
        if len(generated_texts) < 2:
            return 0.0
        
        total_bleu = 0.0
        count = 0
        
        for i, hypothesis in enumerate(generated_texts):
            # Use all other texts as references
            references = [generated_texts[j] for j in range(len(generated_texts)) if i != j]
            bleu_score = self._compute_bleu(hypothesis, references, max_n=n)
            total_bleu += bleu_score
            count += 1
        
        return total_bleu / count if count > 0 else 0.0
    
    def _compute_bleu(self, hypothesis: str, references: List[str], max_n: int = 4) -> float:
        """
        Compute BLEU score (simplified version).
        
        Args:
            hypothesis: Generated text
            references: List of reference texts
            max_n: Maximum n-gram size
        
        Returns:
            BLEU score (0-100)
        """
        # Tokenize by character
        hyp_tokens = list(hypothesis)
        ref_tokens_list = [list(ref) for ref in references]
        
        if len(hyp_tokens) == 0:
            return 0.0
        
        # Compute n-gram precisions
        precisions = []
        for n in range(1, max_n + 1):
            hyp_ngrams = self._get_ngrams_list(hyp_tokens, n)
            
            if len(hyp_ngrams) == 0:
                precisions.append(0.0)
                continue
            
            max_ref_counts = defaultdict(int)
            for ref_tokens in ref_tokens_list:
                ref_ngrams = self._get_ngrams_list(ref_tokens, n)
                ref_counts = Counter(ref_ngrams)
                for ngram in ref_counts:
                    max_ref_counts[ngram] = max(max_ref_counts[ngram], ref_counts[ngram])
            
            hyp_counts = Counter(hyp_ngrams)
            clipped_counts = {
                ngram: min(count, max_ref_counts[ngram])
                for ngram, count in hyp_counts.items()
            }
            
            precision = sum(clipped_counts.values()) / len(hyp_ngrams)
            precisions.append(precision)
        
        # Compute geometric mean of precisions
        if any(p == 0 for p in precisions):
            return 0.0
        
        geo_mean = math.exp(sum(math.log(p) for p in precisions) / len(precisions))
        
        # Brevity penalty
        ref_lengths = [len(ref) for ref in ref_tokens_list]
        closest_ref_len = min(ref_lengths, key=lambda x: abs(x - len(hyp_tokens)))
        
        if len(hyp_tokens) >= closest_ref_len:
            bp = 1.0
        else:
            bp = math.exp(1 - closest_ref_len / len(hyp_tokens))
        
        return bp * geo_mean * 100
    
    def _get_ngrams_list(self, tokens: List[str], n: int) -> List[Tuple[str, ...]]:
        """Extract n-grams from token list."""
        return [tuple(tokens[i:i+n]) for i in range(len(tokens) - n + 1)]
    
    def distinct_n(self, generated_texts: List[str], n: int = 1) -> float:
        """
        Compute Distinct-n: ratio of unique n-grams to total n-grams.
        Higher values indicate more diversity.
        
        Args:
            generated_texts: List of generated text samples
            n: N-gram size
        
        Returns:
            Distinct-n ratio (0-100)
        """
        all_text = ''.join(generated_texts)
        
        if len(all_text) < n:
            return 0.0
        
        ngrams = []
        for i in range(len(all_text) - n + 1):
            ngrams.append(all_text[i:i+n])
        
        if len(ngrams) == 0:
            return 0.0
        
        unique_ngrams = len(set(ngrams))
        total_ngrams = len(ngrams)
        
        return (unique_ngrams / total_ngrams) * 100
    
    def entropy(self, generated_texts: List[str]) -> float:
        """
        Compute Shannon entropy of character distribution in generated text.
        Higher values indicate more uniform/diverse character usage.
        
        Args:
            generated_texts: List of generated text samples
        
        Returns:
            Shannon entropy in bits
        """
        all_text = ''.join(generated_texts)
        
        if len(all_text) == 0:
            return 0.0
        
        # Get character distribution
        char_dist = self._char_distribution(all_text)
        
        # Compute Shannon entropy: H = -Σ p(x) log2(p(x))
        entropy = 0.0
        for prob in char_dist.values():
            if prob > 0:
                entropy -= prob * math.log2(prob)
        
        return entropy
    
    # ============================================================================
    # COMBINED EVALUATION
    # ============================================================================
    
    def compute_all_metrics(self, generated_texts: List[str], val_loss: float) -> Dict[str, float]:
        """
        Compute all evaluation metrics.
        
        Args:
            generated_texts: List of generated text samples
            val_loss: Validation loss from training
        
        Returns:
            Dictionary of all metrics
        """
        metrics = {}
        
        # Specific metrics
        metrics['ngram_overlap_1'] = self.character_ngram_overlap(generated_texts, n=1)
        metrics['ngram_overlap_2'] = self.character_ngram_overlap(generated_texts, n=2)
        metrics['ngram_overlap_3'] = self.character_ngram_overlap(generated_texts, n=3)
        metrics['perplexity'] = self.perplexity_from_loss(val_loss)
        metrics['kl_divergence'] = self.kl_divergence(generated_texts)
        
        # General metrics
        metrics['self_bleu'] = self.self_bleu(generated_texts, n=4)
        metrics['distinct_1'] = self.distinct_n(generated_texts, n=1)
        metrics['distinct_2'] = self.distinct_n(generated_texts, n=2)
        metrics['distinct_3'] = self.distinct_n(generated_texts, n=3)
        metrics['entropy'] = self.entropy(generated_texts)
        
        return metrics


if __name__ == '__main__':
    # Test the metrics with sample data
    print("Testing evaluation metrics...")
    
    evaluator = EvaluationMetrics()
    
    # Sample generated texts
    sample_texts = [
        "ROMEO:\nWhat light through yonder window breaks?",
        "JULIET:\nO Romeo, Romeo! wherefore art thou Romeo?",
        "HAMLET:\nTo be, or not to be, that is the question.",
    ]
    
    sample_val_loss = 1.5
    
    metrics = evaluator.compute_all_metrics(sample_texts, sample_val_loss)
    
    print("\nMetrics computed:")
    for metric_name, value in metrics.items():
        print(f"  {metric_name}: {value:.4f}")

