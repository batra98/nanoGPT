"""
Domain transition metrics to measure the shift from Shakespeare to C code.

These metrics quantify how "code-like" vs "Shakespeare-like" the generated text is.
"""

import re
from typing import List, Dict
from collections import Counter
import numpy as np


class DomainTransitionMetrics:
    """Compute metrics to measure Shakespeare → C code domain transition."""
    
    # C programming keywords and common identifiers
    C_KEYWORDS = [
        'static', 'void', 'struct', 'return', 'int', 'const', 'unsigned',
        'char', 'if', 'else', 'for', 'while', 'do', 'switch', 'case',
        'break', 'continue', 'sizeof', 'typedef', 'enum', 'union',
        'long', 'short', 'float', 'double', 'goto', 'volatile',
        'register', 'extern', 'auto', 'signed', 'inline', 'include',
        'define', 'ifdef', 'ifndef', 'endif', 'NULL', 'true', 'false'
    ]
    
    # Shakespeare-specific words
    SHAKESPEARE_WORDS = [
        'thou', 'thee', 'thy', 'thine', 'thyself',
        'wherefore', 'whence', 'whither', 'whilst',
        'hath', 'doth', 'dost', 'art', 'wilt', 'shalt',
        'tis', 'twas', 'ere', 'hither', 'thither',
        'yonder', 'mine', 'methinks', 'prithee', 'forsooth',
        'anon', 'verily', 'nay', 'aye', 'hark'
    ]
    
    def __init__(self):
        """Initialize domain transition metrics calculator."""
        pass
    
    def compute_keyword_frequency(self, text: str, keywords: List[str]) -> float:
        """
        Compute normalized frequency of keywords in text.
        
        Args:
            text: Input text
            keywords: List of keywords to search for
        
        Returns:
            Frequency as keywords per 1000 characters
        """
        text_lower = text.lower()
        
        # Count keyword occurrences (whole word matches)
        total_count = 0
        for keyword in keywords:
            # Use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword.lower()) + r'\b'
            total_count += len(re.findall(pattern, text_lower))
        
        # Normalize by text length (per 1000 chars)
        if len(text) == 0:
            return 0.0
        
        return (total_count / len(text)) * 1000
    
    def compute_bracket_balance(self, text: str) -> Dict[str, float]:
        """
        Compute bracket/brace balance and frequency.
        
        Args:
            text: Input text
        
        Returns:
            Dictionary with balance ratios and frequencies
        """
        bracket_pairs = [('(', ')'), ('[', ']'), ('{', '}')]
        results = {}
        
        for open_b, close_b in bracket_pairs:
            open_count = text.count(open_b)
            close_count = text.count(close_b)
            
            # Balance ratio (1.0 = perfectly balanced)
            if open_count + close_count == 0:
                balance = 1.0
            else:
                balance = 1.0 - abs(open_count - close_count) / (open_count + close_count)
            
            # Frequency per 1000 chars
            frequency = ((open_count + close_count) / len(text)) * 1000 if len(text) > 0 else 0.0
            
            pair_name = open_b + close_b
            results[f'{pair_name}_balance'] = balance
            results[f'{pair_name}_frequency'] = frequency
        
        return results
    
    def compute_statement_density(self, text: str) -> float:
        """
        Compute semicolon density (proxy for C statements).
        
        Args:
            text: Input text
        
        Returns:
            Semicolons per 1000 characters
        """
        if len(text) == 0:
            return 0.0
        
        semicolon_count = text.count(';')
        return (semicolon_count / len(text)) * 1000
    
    def compute_character_distribution_similarity(self, text: str, reference_text: str) -> float:
        """
        Compute character distribution similarity using KL divergence.
        
        Args:
            text: Generated text
            reference_text: Reference text (kernel or Shakespeare)
        
        Returns:
            KL divergence (lower = more similar)
        """
        if len(text) == 0 or len(reference_text) == 0:
            return float('inf')
        
        # Get character distributions
        text_dist = Counter(text.lower())
        ref_dist = Counter(reference_text.lower())
        
        # Get all characters
        all_chars = set(text_dist.keys()) | set(ref_dist.keys())
        
        # Convert to probabilities with smoothing
        text_probs = np.array([text_dist.get(c, 0) + 1e-10 for c in all_chars])
        ref_probs = np.array([ref_dist.get(c, 0) + 1e-10 for c in all_chars])
        
        text_probs = text_probs / text_probs.sum()
        ref_probs = ref_probs / ref_probs.sum()
        
        # KL divergence
        kl_div = np.sum(text_probs * np.log(text_probs / ref_probs))
        
        return float(kl_div)
    
    def compute_code_likeness_score(self, text: str, kernel_reference: str = None) -> float:
        """
        Compute overall code-likeness score (0-1, higher = more code-like).
        
        Args:
            text: Generated text
            kernel_reference: Optional reference kernel code for distribution comparison
        
        Returns:
            Code-likeness score (0-1)
        """
        scores = []
        
        # 1. C keyword frequency (normalize to 0-1 range, expecting ~5-20 per 1000 chars)
        c_freq = self.compute_keyword_frequency(text, self.C_KEYWORDS)
        c_score = min(c_freq / 20.0, 1.0)
        scores.append(c_score)
        
        # 2. Bracket balance and frequency
        brackets = self.compute_bracket_balance(text)
        # Average balance across all bracket types
        avg_balance = np.mean([brackets[f'{p}_balance'] for p in ['()', '[]', '{}']])
        scores.append(avg_balance)
        
        # Curly brace frequency (common in C, rare in Shakespeare)
        curly_freq = brackets['{}_frequency']
        curly_score = min(curly_freq / 5.0, 1.0)  # Expecting ~1-5 per 1000 chars
        scores.append(curly_score)
        
        # 3. Statement density (semicolons)
        semicolon_freq = self.compute_statement_density(text)
        semicolon_score = min(semicolon_freq / 10.0, 1.0)  # Expecting ~3-10 per 1000 chars
        scores.append(semicolon_score)
        
        # 4. Character distribution similarity (if reference provided)
        if kernel_reference:
            kl_div = self.compute_character_distribution_similarity(text, kernel_reference)
            # Convert KL to similarity score (lower KL = higher similarity)
            # Typical KL range: 0-2, so we invert and normalize
            kl_score = max(0, 1.0 - (kl_div / 2.0))
            scores.append(kl_score)
        
        # Average all component scores
        return float(np.mean(scores))
    
    def compute_shakespeare_likeness_score(self, text: str, shakespeare_reference: str = None) -> float:
        """
        Compute overall Shakespeare-likeness score (0-1, higher = more Shakespeare-like).
        
        Args:
            text: Generated text
            shakespeare_reference: Optional reference Shakespeare text
        
        Returns:
            Shakespeare-likeness score (0-1)
        """
        scores = []
        
        # 1. Shakespeare word frequency (expecting ~2-10 per 1000 chars in Shakespeare)
        shakes_freq = self.compute_keyword_frequency(text, self.SHAKESPEARE_WORDS)
        shakes_score = min(shakes_freq / 10.0, 1.0)
        scores.append(shakes_score)
        
        # 2. Absence of code constructs (inverse of code features)
        brackets = self.compute_bracket_balance(text)
        curly_freq = brackets['{}_frequency']
        # Lower curly brace frequency = more Shakespeare-like
        no_curly_score = max(0, 1.0 - (curly_freq / 5.0))
        scores.append(no_curly_score)
        
        semicolon_freq = self.compute_statement_density(text)
        # Lower semicolon frequency = more Shakespeare-like
        no_semicolon_score = max(0, 1.0 - (semicolon_freq / 10.0))
        scores.append(no_semicolon_score)
        
        # 3. Character distribution similarity (if reference provided)
        if shakespeare_reference:
            kl_div = self.compute_character_distribution_similarity(text, shakespeare_reference)
            kl_score = max(0, 1.0 - (kl_div / 2.0))
            scores.append(kl_score)
        
        return float(np.mean(scores))
    
    def compute_transition_score(self, text: str, 
                                 kernel_reference: str = None,
                                 shakespeare_reference: str = None) -> float:
        """
        Compute transition score: 0 = Shakespeare, 1 = C code.
        
        Args:
            text: Generated text
            kernel_reference: Reference kernel code
            shakespeare_reference: Reference Shakespeare text
        
        Returns:
            Transition score (0-1, 0=Shakespeare, 1=C code)
        """
        code_score = self.compute_code_likeness_score(text, kernel_reference)
        shakespeare_score = self.compute_shakespeare_likeness_score(text, shakespeare_reference)
        
        # Normalize to 0-1 range
        total = code_score + shakespeare_score
        if total == 0:
            return 0.5  # Neutral if both are zero
        
        return code_score / total
    
    def compute_all_transition_metrics(self, 
                                       generated_texts: List[str],
                                       kernel_reference: str = None,
                                       shakespeare_reference: str = None) -> Dict[str, float]:
        """
        Compute all transition metrics for a list of generated texts.
        
        Args:
            generated_texts: List of generated text samples
            kernel_reference: Reference kernel code
            shakespeare_reference: Reference Shakespeare text
        
        Returns:
            Dictionary of aggregated metrics
        """
        # Concatenate all texts for overall analysis
        combined_text = '\n'.join(generated_texts)
        
        metrics = {}
        
        # Code-likeness metrics
        metrics['c_keyword_freq'] = self.compute_keyword_frequency(combined_text, self.C_KEYWORDS)
        metrics['shakespeare_word_freq'] = self.compute_keyword_frequency(combined_text, self.SHAKESPEARE_WORDS)
        
        # Bracket/brace metrics
        bracket_metrics = self.compute_bracket_balance(combined_text)
        metrics.update(bracket_metrics)
        
        # Statement density
        metrics['semicolon_density'] = self.compute_statement_density(combined_text)
        
        # Overall scores
        metrics['code_likeness_score'] = self.compute_code_likeness_score(combined_text, kernel_reference)
        metrics['shakespeare_likeness_score'] = self.compute_shakespeare_likeness_score(combined_text, shakespeare_reference)
        metrics['transition_score'] = self.compute_transition_score(combined_text, kernel_reference, shakespeare_reference)
        
        # Character distribution KL divergences (if references provided)
        if kernel_reference:
            metrics['kl_div_vs_kernel'] = self.compute_character_distribution_similarity(combined_text, kernel_reference)
        
        if shakespeare_reference:
            metrics['kl_div_vs_shakespeare'] = self.compute_character_distribution_similarity(combined_text, shakespeare_reference)
        
        return metrics


if __name__ == '__main__':
    # Test the metrics with sample texts
    print("Testing Domain Transition Metrics...")
    
    evaluator = DomainTransitionMetrics()
    
    # Sample C code
    c_code = """
    static int my_function(struct device *dev) {
        int result = 0;
        if (dev != NULL) {
            result = process_device(dev);
        }
        return result;
    }
    """
    
    # Sample Shakespeare
    shakespeare = """
    ROMEO:
    But soft! What light through yonder window breaks?
    It is the east, and Juliet is the sun.
    Arise, fair sun, and kill the envious moon,
    Who is already sick and pale with grief.
    """
    
    # Test on C code
    c_metrics = evaluator.compute_all_transition_metrics([c_code], c_code, shakespeare)
    print("\nC Code Metrics:")
    print(f"  Code likeness: {c_metrics['code_likeness_score']:.3f}")
    print(f"  Shakespeare likeness: {c_metrics['shakespeare_likeness_score']:.3f}")
    print(f"  Transition score: {c_metrics['transition_score']:.3f} (should be close to 1.0)")
    
    # Test on Shakespeare
    shakes_metrics = evaluator.compute_all_transition_metrics([shakespeare], c_code, shakespeare)
    print("\nShakespeare Metrics:")
    print(f"  Code likeness: {shakes_metrics['code_likeness_score']:.3f}")
    print(f"  Shakespeare likeness: {shakes_metrics['shakespeare_likeness_score']:.3f}")
    print(f"  Transition score: {shakes_metrics['transition_score']:.3f} (should be close to 0.0)")
    
    print("\n✓ Tests complete")

