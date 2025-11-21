"""
Preference Heuristic for RLHF.

Rule: Prefer text with more dialogue structure (character names followed by colons).
Metric: Count dialogue markers (lines with character names + colons) per 100 characters.

This provides a simple but meaningful quality signal for Shakespeare text generation:
well-structured dialogue with proper character attribution is more readable and authentic.

Pattern: Matches lines like "ROMEO:", "First Citizen:", "MENENIUS:", etc.
"""

import re
from typing import Tuple


def compute_dialogue_density(text: str) -> float:
    """
    Compute dialogue density as the number of character name markers per 100 characters.
    
    Counts lines that match dialogue patterns:
    - All caps names: ROMEO:, JULIET:, MENENIUS:
    - Title case names: First Citizen:, Second Citizen:
    - Mixed: Lord Capulet:, Lady Macbeth:
    
    Args:
        text: String of Shakespeare text to analyze
    
    Returns:
        Float representing dialogue markers per 100 characters
    """
    if len(text) == 0:
        return 0.0
    
    # Count character name patterns
    # Pattern: Start of line (or after newline), capital letter(s), optional spaces/words, colon
    # Matches: "ROMEO:", "First Citizen:", "LADY MACBETH:", etc.
    dialogue_markers = len(re.findall(r'(?:^|\n)[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*:', text))
    
    # Normalize to per 100 characters
    density = (dialogue_markers / len(text)) * 100.0
    
    return density


def compute_dialogue_stats(text: str) -> dict:
    """
    Compute detailed statistics about dialogue structure in text.
    
    Args:
        text: String of Shakespeare text to analyze
    
    Returns:
        Dictionary with dialogue statistics
    """
    # Count different patterns
    all_caps = len(re.findall(r'(?:^|\n)[A-Z]{2,}:', text))  # ROMEO:, JULIET:
    title_case = len(re.findall(r'(?:^|\n)[A-Z][a-z]+(?:\s+[A-Z][a-z]+)*:', text))  # First Citizen:
    
    total_dialogue = len(re.findall(r'(?:^|\n)[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*:', text))
    
    # Count lines
    lines = text.split('\n')
    total_lines = len(lines)
    dialogue_lines = sum(1 for line in lines if re.match(r'^[A-Z][a-zA-Z]*(?:\s+[A-Z][a-zA-Z]*)*:', line))
    
    return {
        'all_caps_names': all_caps,
        'title_case_names': title_case,
        'total_dialogue_markers': total_dialogue,
        'total_lines': total_lines,
        'dialogue_lines': dialogue_lines,
        'text_length': len(text),
        'density': compute_dialogue_density(text)
    }


def assign_preference(text_a: str, text_b: str) -> Tuple[int, float, float]:
    """
    Assign preference between two text samples based on dialogue density.
    
    Args:
        text_a: First text sample
        text_b: Second text sample
    
    Returns:
        Tuple of (preference, density_a, density_b)
        - preference: 0 if A is preferred, 1 if B is preferred
        - density_a: Dialogue density of text A
        - density_b: Dialogue density of text B
    """
    density_a = compute_dialogue_density(text_a)
    density_b = compute_dialogue_density(text_b)
    
    # Prefer the text with higher dialogue density
    preference = 0 if density_a >= density_b else 1
    
    return preference, density_a, density_b


def report_statistics(texts: list, name: str = "Dataset"):
    """
    Report statistics about dialogue density in a collection of texts.
    
    Args:
        texts: List of text strings to analyze
        name: Name for the dataset being analyzed
    """
    import numpy as np
    
    densities = [compute_dialogue_density(text) for text in texts]
    
    print(f"\n{'='*60}")
    print(f"{name} Dialogue Density Statistics")
    print(f"{'='*60}")
    print(f"Number of samples: {len(texts)}")
    print(f"Mean density: {np.mean(densities):.4f} dialogue markers per 100 chars")
    print(f"Median density: {np.median(densities):.4f}")
    print(f"Std deviation: {np.std(densities):.4f}")
    print(f"Min density: {np.min(densities):.4f}")
    print(f"Max density: {np.max(densities):.4f}")
    print(f"{'='*60}\n")
    
    # Show distribution
    print("Distribution:")
    bins = [0, 0.5, 1.0, 2.0, 5.0, float('inf')]
    bin_labels = ['0-0.5', '0.5-1.0', '1.0-2.0', '2.0-5.0', '5.0+']
    
    for i in range(len(bins) - 1):
        count = sum(1 for d in densities if bins[i] <= d < bins[i+1])
        pct = (count / len(densities)) * 100
        print(f"  {bin_labels[i]:>10}: {count:5d} samples ({pct:5.1f}%)")
    
    print()


if __name__ == '__main__':
    # Example usage and testing
    
    # Example 1: Well-structured Shakespeare dialogue
    good_dialogue = """
ROMEO:
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.

JULIET:
O Romeo, Romeo! wherefore art thou Romeo?
Deny thy father and refuse thy name.

ROMEO:
Shall I hear more, or shall I speak at this?

JULIET:
'Tis but thy name that is my enemy.
"""
    
    # Example 2: Shakespeare with less dialogue structure (narrative/monologue)
    poor_dialogue = """
But, soft! what light through yonder window breaks?
It is the east, and Juliet is the sun.
Arise, fair sun, and kill the envious moon,
Who is already sick and pale with grief,
That thou her maid art far more fair than she.
"""
    
    # Example 3: Mixed structure
    mixed_dialogue = """
First Citizen:
Before we proceed any further, hear me speak.

All:
Speak, speak.

The people gather in the marketplace,
awaiting word from their leaders.
"""
    
    print("Testing preference heuristic for Shakespeare dialogue:")
    print("\nExample 1: Well-structured dialogue")
    stats = compute_dialogue_stats(good_dialogue)
    print(f"  Dialogue density: {stats['density']:.4f}")
    print(f"  Total markers: {stats['total_dialogue_markers']}")
    print(f"  Dialogue lines: {stats['dialogue_lines']}/{stats['total_lines']}")
    
    print("\nExample 2: Narrative/monologue (less dialogue structure)")
    stats = compute_dialogue_stats(poor_dialogue)
    print(f"  Dialogue density: {stats['density']:.4f}")
    print(f"  Total markers: {stats['total_dialogue_markers']}")
    print(f"  Dialogue lines: {stats['dialogue_lines']}/{stats['total_lines']}")
    
    print("\nExample 3: Mixed structure")
    stats = compute_dialogue_stats(mixed_dialogue)
    print(f"  Dialogue density: {stats['density']:.4f}")
    print(f"  Total markers: {stats['total_dialogue_markers']}")
    print(f"  Dialogue lines: {stats['dialogue_lines']}/{stats['total_lines']}")
    
    print("\n" + "="*60)
    print("Preference Assignment Test")
    print("="*60)
    
    pref, dens_a, dens_b = assign_preference(good_dialogue, poor_dialogue)
    print(f"Comparing structured dialogue vs narrative:")
    print(f"  Structured dialogue density: {dens_a:.4f}")
    print(f"  Narrative density: {dens_b:.4f}")
    print(f"  Preferred: {'A (structured dialogue)' if pref == 0 else 'B (narrative)'}")
    
    pref, dens_a, dens_b = assign_preference(poor_dialogue, good_dialogue)
    print(f"\nComparing narrative vs structured dialogue:")
    print(f"  Narrative density: {dens_a:.4f}")
    print(f"  Structured dialogue density: {dens_b:.4f}")
    print(f"  Preferred: {'A (narrative)' if pref == 0 else 'B (structured dialogue)'}")

