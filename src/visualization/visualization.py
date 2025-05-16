"""
Visualization module for corpus analysis results.
"""

import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from wordcloud import WordCloud
from typing import Dict, List, Any
import plotly.express as px
import plotly.graph_objects as go

def plot_lexical_diversity_comparison(
    llm_stats: Dict[str, float],
    reference_stats: Dict[str, float],
    title: str = "Lexical Diversity Comparison"
) -> go.Figure:
    """
    Create a bar chart comparing lexical diversity metrics between LLM output and reference corpus.
    
    Args:
        llm_stats (Dict[str, float]): Lexical diversity metrics for LLM output
        reference_stats (Dict[str, float]): Lexical diversity metrics for reference corpus
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly figure object
    """
    metrics = list(llm_stats.keys())
    llm_values = list(llm_stats.values())
    ref_values = list(reference_stats.values())
    
    fig = go.Figure(data=[
        go.Bar(name='LLM Output', x=metrics, y=llm_values),
        go.Bar(name='Reference Corpus', x=metrics, y=ref_values)
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Metrics",
        yaxis_title="Value",
        barmode='group'
    )
    
    return fig

def plot_word_frequency_comparison(
    llm_freq: pd.Series,
    ref_freq: pd.Series,
    top_n: int = 20,
    title: str = "Word Frequency Comparison"
) -> go.Figure:
    """
    Create a bar chart comparing word frequencies between LLM output and reference corpus.
    
    Args:
        llm_freq (pd.Series): Word frequencies for LLM output
        ref_freq (pd.Series): Word frequencies for reference corpus
        top_n (int): Number of top words to display
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly figure object
    """
    # Get top N words from both corpora
    top_words = set(
        list(llm_freq.nlargest(top_n).index) + 
        list(ref_freq.nlargest(top_n).index)
    )
    
    # Create comparison DataFrame
    comparison = pd.DataFrame({
        'word': list(top_words),
        'llm_freq': [llm_freq.get(word, 0) for word in top_words],
        'ref_freq': [ref_freq.get(word, 0) for word in top_words]
    })
    
    # Sort by LLM frequency
    comparison = comparison.sort_values('llm_freq', ascending=False)
    
    fig = go.Figure(data=[
        go.Bar(name='LLM Output', x=comparison['word'], y=comparison['llm_freq']),
        go.Bar(name='Reference Corpus', x=comparison['word'], y=comparison['ref_freq'])
    ])
    
    fig.update_layout(
        title=title,
        xaxis_title="Words",
        yaxis_title="Frequency",
        barmode='group',
        xaxis_tickangle=-45
    )
    
    return fig

def create_wordcloud(
    word_freq: pd.Series,
    title: str = "Word Cloud"
) -> plt.Figure:
    """
    Create a word cloud from word frequencies.
    
    Args:
        word_freq (pd.Series): Word frequencies
        title (str): Plot title
        
    Returns:
        plt.Figure: Matplotlib figure object
    """
    wordcloud = WordCloud(
        width=800,
        height=400,
        background_color='white'
    ).generate_from_frequencies(word_freq)
    
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.imshow(wordcloud, interpolation='bilinear')
    ax.axis('off')
    ax.set_title(title)
    
    return fig

def plot_metric_distribution(
    df: pd.DataFrame,
    metric: str,
    title: str = None
) -> go.Figure:
    """
    Create a histogram of a metric's distribution.
    
    Args:
        df (pd.DataFrame): DataFrame containing metrics
        metric (str): Name of the metric to plot
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly figure object
    """
    if title is None:
        title = f"Distribution of {metric}"
    
    fig = px.histogram(
        df,
        x=metric,
        title=title,
        nbins=30
    )
    
    fig.update_layout(
        xaxis_title=metric,
        yaxis_title="Count"
    )
    
    return fig

def plot_correlation_matrix(
    df: pd.DataFrame,
    title: str = "Metric Correlations"
) -> go.Figure:
    """
    Create a correlation matrix heatmap.
    
    Args:
        df (pd.DataFrame): DataFrame containing metrics
        title (str): Plot title
        
    Returns:
        go.Figure: Plotly figure object
    """
    corr = df.corr()
    
    fig = go.Figure(data=go.Heatmap(
        z=corr,
        x=corr.columns,
        y=corr.columns,
        colorscale='RdBu',
        zmin=-1,
        zmax=1
    ))
    
    fig.update_layout(
        title=title,
        xaxis_title="Metrics",
        yaxis_title="Metrics"
    )
    
    return fig 