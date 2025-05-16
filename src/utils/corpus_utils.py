"""
Utility functions for corpus analysis.
"""

import nltk
from nltk.corpus import brown
from nltk.tokenize import word_tokenize
from nltk.collocations import BigramCollocationFinder
from nltk.metrics import BigramAssocMeasures
import pandas as pd
from typing import List, Dict, Tuple
from lexicalrichness import LexicalRichness

def load_brown_corpus() -> List[str]:
    """
    Load the Brown Corpus and return a list of sentences.
    
    Returns:
        List[str]: List of sentences from the Brown Corpus
    """
    return brown.sents()

def calculate_lexical_diversity(text: str) -> Dict[str, float]:
    """
    Calculate various lexical diversity metrics for a given text.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict[str, float]: Dictionary containing various lexical diversity metrics
    """
    lex = LexicalRichness(text)
    return {
        'ttr': lex.ttr,  # Type-Token Ratio
        'mtld': lex.mtld,  # Measure of Textual Lexical Diversity
        'hdd': lex.hdd,  # Hypergeometric Distribution D
        'vocd': lex.vocd,  # Vocabulary Diversity
        'rttr': lex.rttr,  # Root Type-Token Ratio
        'cttr': lex.cttr,  # Corrected Type-Token Ratio
        'maas': lex.maas,  # Maas's Index
    }

def find_collocations(text: str, n: int = 20) -> List[Tuple[str, str, float]]:
    """
    Find the top n collocations in a text using PMI (Pointwise Mutual Information).
    
    Args:
        text (str): Input text
        n (int): Number of collocations to return
        
    Returns:
        List[Tuple[str, str, float]]: List of (word1, word2, score) tuples
    """
    tokens = word_tokenize(text.lower())
    finder = BigramCollocationFinder.from_words(tokens)
    finder.apply_freq_filter(3)  # Filter out bigrams that appear less than 3 times
    return finder.nbest(BigramAssocMeasures.pmi, n)

def get_word_frequencies(text: str) -> pd.Series:
    """
    Calculate word frequencies in a text.
    
    Args:
        text (str): Input text
        
    Returns:
        pd.Series: Series with words as index and frequencies as values
    """
    tokens = word_tokenize(text.lower())
    freq_dist = nltk.FreqDist(tokens)
    return pd.Series(freq_dist)

def calculate_basic_stats(text: str) -> Dict[str, float]:
    """
    Calculate basic text statistics.
    
    Args:
        text (str): Input text
        
    Returns:
        Dict[str, float]: Dictionary containing basic text statistics
    """
    tokens = word_tokenize(text)
    sentences = nltk.sent_tokenize(text)
    
    return {
        'word_count': len(tokens),
        'sentence_count': len(sentences),
        'avg_sentence_length': len(tokens) / len(sentences),
        'unique_words': len(set(tokens)),
        'lexical_diversity': len(set(tokens)) / len(tokens)
    } 