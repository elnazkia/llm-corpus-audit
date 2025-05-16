# scripts/run_demo.py
"""
This script demonstrates the usage of our corpus analysis modules
for comparing LLM outputs with reference corpora.
"""

import sys
import os
import pandas as pd
import matplotlib.pyplot as plt # Import for plt.show() if create_wordcloud is used directly

# Adjust the Python path to include the 'src' directory
# This assumes the script is run from the project root or within the 'scripts' directory
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.utils.corpus_utils import load_brown_corpus, calculate_basic_stats
from src.analysis.llm_analysis import LLMAnalyzer
from src.visualization.visualization import (
    plot_lexical_diversity_comparison,
    plot_word_frequency_comparison,
    create_wordcloud,
    plot_metric_distribution,
    plot_correlation_matrix
)

def main():
    """Main function to run the demo."""

    print("Starting Corpus Analysis Demo...")

    # --- 1. Load Reference Corpus ---
    print("\n--- Loading Reference Corpus ---")
    # Ensure NLTK data (brown, punkt) is downloaded by running:
    # import nltk
    # nltk.download('brown')
    # nltk.download('punkt')
    try:
        reference_corpus_sents = load_brown_corpus() # Returns list of sentences (lists of words)
        reference_corpus_text = ' '.join([' '.join(sent) for sent in reference_corpus_sents])
        print(f"Number of sentences in Brown Corpus: {len(reference_corpus_sents)}")
        if reference_corpus_sents:
            print(f"First sentence: {' '.join(reference_corpus_sents[0])}")
        
        # Calculate and print some basic stats for the reference corpus for context
        ref_stats = calculate_basic_stats(reference_corpus_text)
        print("\nReference Corpus Basic Stats:")
        for key, value in ref_stats.items():
            print(f"  {key.replace('_', ' ').title()}: {value:.2f}" if isinstance(value, float) else f"  {key.replace('_', ' ').title()}: {value}")

    except Exception as e:
        print(f"Error loading Brown Corpus: {e}")
        print("Please ensure you have run `nltk.download('brown')` and `nltk.download('punkt')`.")
        return

    # --- 2. Initialize LLM Analyzer ---
    print("\n--- Initializing LLM Analyzer ---")
    analyzer = LLMAnalyzer(reference_corpus_sents)
    print("LLM Analyzer initialized with Brown Corpus.")
    if 'lexical_diversity' in analyzer.reference_stats and 'ttr' in analyzer.reference_stats['lexical_diversity']:
        print("Reference corpus stats (e.g., TTR):", analyzer.reference_stats['lexical_diversity']['ttr'])
    else:
        print("Reference corpus TTR could not be determined from analyzer.reference_stats")


    # --- 3. Sample LLM Outputs ---
    print("\n--- Defining Sample LLM Outputs ---")
    sample_outputs = [
        "The quick brown fox jumps over the lazy dog. This is a sample text for demonstration purposes, aiming to show how different texts can be analyzed.",
        "In the beginning, there was light. And the light was good. This is another sample text, quite short but illustrative nonetheless.",
        "Machine learning models can generate text that mimics human writing with varying degrees of success. This is a third sample, providing more content.",
        "Consider the implications of artificial intelligence on creative writing. The possibilities are vast, yet challenges remain in achieving true artistry and emotional depth.",
        "Yet another example to showcase the capabilities of our analytical tools. We strive for robust and informative metrics."
    ]
    for i, output in enumerate(sample_outputs):
        print(f"Sample Output {i+1}: \"{output[:50]}...\"")

    # --- 4. Analyze a Single LLM Output ---
    analysis_single = None # Initialize to None
    if sample_outputs:
        print("\n--- Analyzing a Single LLM Output (Sample 1) ---")
        analysis_single = analyzer.analyze_llm_output(sample_outputs[0])
        print("Analysis of first LLM output:")
        if analysis_single and 'llm_stats' in analysis_single and 'reference_stats' in analysis_single:
            print(f"  Word count (LLM): {analysis_single['llm_stats']['basic_stats']['word_count']}")
            print(f"  Sentence count (LLM): {analysis_single['llm_stats']['basic_stats']['sentence_count']}")
            print(f"  Lexical diversity (TTR) (LLM): {analysis_single['llm_stats']['lexical_diversity']['ttr']:.3f}")
            print(f"  Lexical diversity (TTR) (Reference): {analysis_single['reference_stats']['lexical_diversity']['ttr']:.3f}")
            if 'comparison' in analysis_single and 'lexical_diversity_diff' in analysis_single['comparison']:
                 print(f"  Difference in TTR: {analysis_single['comparison']['lexical_diversity_diff']['ttr']:.3f}")
        else:
            print("Could not retrieve all stats for single LLM output analysis.")

    # --- 5. Compare Multiple LLM Outputs ---
    print("\n--- Comparing Multiple LLM Outputs ---")
    comparison_df = analyzer.compare_multiple_outputs(sample_outputs)
    print("Comparison DataFrame of multiple LLM outputs:")
    pd.set_option('display.width', 120) # Adjust width for better terminal display
    pd.set_option('display.max_columns', None)
    print(comparison_df)

    # --- 6. Visualize Results (Plots will be shown or saved) ---
    print("\n--- Generating Visualizations (Figures will be shown or saved) ---")
    
    output_dir = "output"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    print(f"Plots will be saved to the '{output_dir}' directory.")

    if analysis_single: # Check if analysis_single was successfully populated
        # a. Lexical Diversity Comparison for the first sample
        try:
            fig_lex_div = plot_lexical_diversity_comparison(
                analysis_single['llm_stats']['lexical_diversity'],
                analysis_single['reference_stats']['lexical_diversity'],
                title="Lexical Diversity: LLM Output 1 vs. Brown Corpus"
            )
            fig_lex_div_path = os.path.join(output_dir, "lexical_diversity_comparison.png")
            fig_lex_div.write_image(fig_lex_div_path)
            print(f"Lexical diversity comparison plot saved to {fig_lex_div_path}")
        except Exception as e:
            print(f"Could not generate/save lexical diversity plot: {e}. Make sure 'kaleido' is installed (`pip install kaleido`).")

        # b. Word Frequency Comparison for the first sample
        try:
            fig_word_freq = plot_word_frequency_comparison(
                analysis_single['llm_stats']['word_frequencies'],
                analysis_single['reference_stats']['word_frequencies'],
                top_n=15,
                title="Word Frequencies (Top 15): LLM Output 1 vs. Brown Corpus"
            )
            fig_word_freq_path = os.path.join(output_dir, "word_frequency_comparison.png")
            fig_word_freq.write_image(fig_word_freq_path)
            print(f"Word frequency comparison plot saved to {fig_word_freq_path}")
        except Exception as e:
            print(f"Could not generate/save word frequency plot: {e}.")

        # c. Word Cloud for the first LLM output
        try:
            fig_wordcloud = create_wordcloud(analysis_single['llm_stats']['word_frequencies'], "LLM Output 1 Word Cloud")
            fig_wordcloud_path = os.path.join(output_dir, "llm_output_wordcloud.png")
            fig_wordcloud.savefig(fig_wordcloud_path)
            plt.close(fig_wordcloud)
            print(f"Word cloud for LLM output saved to {fig_wordcloud_path}")
        except Exception as e:
            print(f"Could not generate/save word cloud: {e}.")

    # d. Metric Distribution from the comparison DataFrame
    if not comparison_df.empty:
        try:
            fig_metric_dist = plot_metric_distribution(comparison_df, 'ttr', title="Distribution of TTR in Sample LLM Outputs")
            fig_metric_dist_path = os.path.join(output_dir, "ttr_distribution.png")
            fig_metric_dist.write_image(fig_metric_dist_path)
            print(f"TTR distribution plot saved to {fig_metric_dist_path}")
        except Exception as e:
            print(f"Could not generate/save TTR distribution plot: {e}.")

        # e. Correlation Matrix from the comparison DataFrame
        import numpy as np # ensure numpy is imported for .number
        numeric_df = comparison_df.select_dtypes(include=np.number)
        if numeric_df.shape[1] > 1:
            try:
                fig_corr_matrix = plot_correlation_matrix(numeric_df, title="Correlation Matrix of Metrics in Sample LLM Outputs")
                fig_corr_matrix_path = os.path.join(output_dir, "metrics_correlation_matrix.png")
                fig_corr_matrix.write_image(fig_corr_matrix_path)
                print(f"Correlation matrix plot saved to {fig_corr_matrix_path}")
            except Exception as e:
                print(f"Could not generate/save correlation matrix: {e}.")
        else:
            print("Not enough numeric columns in comparison_df to generate a correlation matrix.")

    print("\nDemo finished. Check the 'output' directory for saved plots.")

if __name__ == "__main__":
    main() 