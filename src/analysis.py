"""
Main analysis pipeline: load data, run all analyses, save results.
"""

import os
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple

from preprocessing import load_corpus, filter_relevant_sentences
from passive_detection import analyze_passive_corpus
from sentiment_analysis import analyze_rhetoric_corpus
from visualization import (
    plot_passive_comparison,
    plot_rhetoric_comparison,
    plot_temporal_dynamics,
    plot_ttr_comparison
)


def calculate_ttr(text: str) -> float:
    """
    Calculate Type-Token Ratio (lexical diversity).
    
    Args:
        text: Text string
        
    Returns:
        TTR = unique_words / total_words
    """
    import re
    words = re.findall(r'\b[a-zа-я]+\b', text.lower())
    if not words:
        return 0.0
    unique_words = set(words)
    return len(unique_words) / len(words)


def analyze_ttr_corpus(corpus: List[Dict]) -> List[Dict]:
    """
    Calculate TTR for all articles.
    
    Args:
        corpus: List of article dicts
        
    Returns:
        List of articles with TTR added
    """
    results = []
    for article in corpus:
        ttr = calculate_ttr(article['text'])
        results.append({
            'filename': article['filename'],
            'date': article.get('date'),
            'ttr': ttr,
            'text_length': len(article['text'])
        })
    return results


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 95) -> Tuple[float, float]:
    """
    Calculate confidence intervals using bootstrap.
    
    Args:
        data: Array of values
        n_bootstrap: Number of bootstrap iterations
        ci: Confidence level (e.g., 95)
        
    Returns:
        Tuple: (lower_bound, upper_bound)
    """
    from sklearn.utils import resample
    
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = resample(data, replace=True)
        bootstrapped_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
    
    return lower, upper


def run_full_pipeline(us_folder: str, ussr_folder: str, output_dir: str = 'results'):
    """
    Run complete analysis pipeline.
    
    Args:
        us_folder: Path to US articles folder
        ussr_folder: Path to USSR articles folder
        output_dir: Directory for saving results
    """
    print("Loading US corpus...")
    us_raw = load_corpus(us_folder)
    us_filtered = filter_relevant_sentences(us_raw, language='en')
    
    print("Loading USSR corpus...")
    ussr_raw = load_corpus(ussr_folder)
    ussr_filtered = filter_relevant_sentences(ussr_raw, language='ru')
    
    print(f"US: {len(us_filtered)} articles with relevant content")
    print(f"USSR: {len(ussr_filtered)} articles with relevant content")
    
    print("\nAnalyzing passive voice...")
    us_passive = analyze_passive_corpus(us_filtered, language='en')
    ussr_passive = analyze_passive_corpus(ussr_filtered, language='ru')
    
    print("Analyzing rhetoric...")
    us_rhetoric = analyze_rhetoric_corpus(us_filtered, language='en')
    ussr_rhetoric = analyze_rhetoric_corpus(ussr_filtered, language='ru')
    
    print("Calculating TTR...")
    us_ttr = analyze_ttr_corpus(us_filtered)
    ussr_ttr = analyze_ttr_corpus(ussr_filtered)
    
    df_passive = pd.DataFrame(us_passive + ussr_passive)
    df_rhetoric = pd.DataFrame(us_rhetoric + ussr_rhetoric)
    df_ttr = pd.DataFrame(us_ttr + ussr_ttr)
    
    df_passive['country'] = ['US'] * len(us_passive) + ['USSR'] * len(ussr_passive)
    df_rhetoric['country'] = ['US'] * len(us_rhetoric) + ['USSR'] * len(ussr_rhetoric)
    df_ttr['country'] = ['US'] * len(us_ttr) + ['USSR'] * len(ussr_ttr)
    
    us_passive_vals = df_passive[df_passive['country'] == 'US']['passive_no_agent_percent'].dropna().values
    ussr_passive_vals = df_passive[df_passive['country'] == 'USSR']['passive_no_agent_percent'].dropna().values
    
    us_passive_ci = bootstrap_ci(us_passive_vals)
    ussr_passive_ci = bootstrap_ci(ussr_passive_vals)
    
    us_acc_vals = df_rhetoric[df_rhetoric['country'] == 'US']['accusatory_frequency'].dropna().values
    ussr_acc_vals = df_rhetoric[df_rhetoric['country'] == 'USSR']['accusatory_frequency'].dropna().values
    
    us_ttr_vals = df_ttr[df_ttr['country'] == 'US']['ttr'].dropna().values
    ussr_ttr_vals = df_ttr[df_ttr['country'] == 'USSR']['ttr'].dropna().values
    
    summary = pd.DataFrame({
        'Metric': ['Passive without agent (%)', 'Accusatory frequency', 'Lexical diversity (TTR)'],
        'US_mean': [np.mean(us_passive_vals), np.mean(us_acc_vals), np.mean(us_ttr_vals)],
        'US_CI_lower': [us_passive_ci[0], bootstrap_ci(us_acc_vals)[0], bootstrap_ci(us_ttr_vals)[0]],
        'US_CI_upper': [us_passive_ci[1], bootstrap_ci(us_acc_vals)[1], bootstrap_ci(us_ttr_vals)[1]],
        'USSR_mean': [np.mean(ussr_passive_vals), np.mean(ussr_acc_vals), np.mean(ussr_ttr_vals)],
        'USSR_CI_lower': [ussr_passive_ci[0], bootstrap_ci(ussr_acc_vals)[0], bootstrap_ci(ussr_ttr_vals)[0]],
        'USSR_CI_upper': [ussr_passive_ci[1], bootstrap_ci(ussr_acc_vals)[1], bootstrap_ci(ussr_ttr_vals)[1]]
    })
    
    os.makedirs(output_dir, exist_ok=True)
    summary.to_csv(os.path.join(output_dir, 'metrics.csv'), index=False)
    print(f"\nSummary saved to {output_dir}/metrics.csv")
    
    print("\nGenerating visualizations...")
    os.makedirs(os.path.join(output_dir, 'figures'), exist_ok=True)
    
    plot_passive_comparison(
        df_passive,
        save_path=os.path.join(output_dir, 'figures', 'voice_analysis.png')
    )
    
    plot_rhetoric_comparison(
        df_rhetoric,
        save_path=os.path.join(output_dir, 'figures', 'rhetoric_heatmap.png')
    )
    
    plot_temporal_dynamics(
        df_passive[df_passive['country'] == 'US'],
        save_path=os.path.join(output_dir, 'figures', 'temporal_voice.png')
    )
    
    plot_ttr_comparison(
        df_ttr,
        save_path=os.path.join(output_dir, 'figures', 'ttr_comparison.png')
    )
    
    print("All visualizations saved.")
    return df_passive, df_rhetoric, df_ttr


if __name__ == '__main__':
    US_FOLDER = '../data/american_data'
    USSR_FOLDER = '../data/russian_data'
    OUTPUT_DIR = '../results'
    
    run_full_pipeline(US_FOLDER, USSR_FOLDER, OUTPUT_DIR)