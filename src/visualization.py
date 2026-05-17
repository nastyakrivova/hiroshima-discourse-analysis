"""
Visualization functions for discourse analysis results.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from typing import Tuple


def bootstrap_ci(data: np.ndarray, n_bootstrap: int = 1000, ci: float = 95) -> Tuple[float, float]:
    """
    Calculate confidence intervals using bootstrap.
    """
    from sklearn.utils import resample
    
    bootstrapped_means = []
    for _ in range(n_bootstrap):
        sample = resample(data, replace=True)
        bootstrapped_means.append(np.mean(sample))
    
    lower = np.percentile(bootstrapped_means, (100 - ci) / 2)
    upper = np.percentile(bootstrapped_means, 100 - (100 - ci) / 2)
    return lower, upper


def plot_passive_comparison(df: pd.DataFrame, save_path: str = None):
    """
    Plot passive voice comparison with confidence intervals.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    us_data = df[df['country'] == 'US']['passive_no_agent_percent'].dropna().values
    ussr_data = df[df['country'] == 'USSR']['passive_no_agent_percent'].dropna().values
    
    us_mean = np.mean(us_data)
    ussr_mean = np.mean(ussr_data)
    
    us_lower, us_upper = bootstrap_ci(us_data)
    ussr_lower, ussr_upper = bootstrap_ci(ussr_data)
    
    countries = ['USA', 'USSR']
    means = [us_mean, ussr_mean]
    errors = [[us_mean - us_lower, us_upper - us_mean],
              [ussr_mean - ussr_lower, ussr_upper - ussr_mean]]
    
    bars = ax.bar(countries, means, color=['#90be6d', '#f9a26c'], edgecolor='black',
                  yerr=errors, capsize=10, error_kw={'linewidth': 2})
    
    for bar, mean in zip(bars, means):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1.5,
                f'{mean:.1f}%', ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    ax.set_ylabel('Passive without agent (%)', fontsize=12)
    ax.set_title('Figure 1. Passive voice without agent: USA vs USSR\n(bars: mean, black lines: 95% CI)')
    ax.set_ylim(0, max(means) + 10)
    ax.grid(axis='y', alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_rhetoric_comparison(df: pd.DataFrame, save_path: str = None):
    """
    Plot accusatory lexicon and modal markers comparison.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Accusatory lexicon
    us_acc = df[df['country'] == 'US']['accusatory_frequency'].dropna()
    ussr_acc = df[df['country'] == 'USSR']['accusatory_frequency'].dropna()
    
    us_acc_mean = us_acc.mean()
    ussr_acc_mean = ussr_acc.mean()
    
    us_acc_lower, us_acc_upper = bootstrap_ci(us_acc.values)
    ussr_acc_lower, ussr_acc_upper = bootstrap_ci(ussr_acc.values)
    
    bars1 = ax1.bar(['USA', 'USSR'], [us_acc_mean, ussr_acc_mean],
                    color=['#90be6d', '#f9a26c'], edgecolor='black',
                    yerr=[[us_acc_mean - us_acc_lower, ussr_acc_mean - ussr_acc_lower],
                          [us_acc_upper - us_acc_mean, ussr_acc_upper - ussr_acc_mean]],
                    capsize=8, error_kw={'linewidth': 2})
    
    for bar, val in zip(bars1, [us_acc_mean, ussr_acc_mean]):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10)
    
    ax1.set_ylabel('Frequency (normalized)', fontsize=11)
    ax1.set_title('Accusatory lexicon\n(condemn, denounce, blame, etc.)')
    ax1.grid(axis='y', alpha=0.3)
    
    # Modal markers
    us_mod = df[df['country'] == 'US']['modal_frequency'].dropna()
    ussr_mod = df[df['country'] == 'USSR']['modal_frequency'].dropna()
    
    us_mod_mean = us_mod.mean()
    ussr_mod_mean = ussr_mod.mean()
    
    us_mod_lower, us_mod_upper = bootstrap_ci(us_mod.values)
    ussr_mod_lower, ussr_mod_upper = bootstrap_ci(ussr_mod.values)
    
    bars2 = ax2.bar(['USA', 'USSR'], [us_mod_mean, ussr_mod_mean],
                    color=['#90be6d', '#f9a26c'], edgecolor='black',
                    yerr=[[us_mod_mean - us_mod_lower, ussr_mod_mean - ussr_mod_lower],
                          [us_mod_upper - us_mod_mean, ussr_mod_upper - ussr_mod_mean]],
                    capsize=8, error_kw={'linewidth': 2})
    
    for bar, val in zip(bars2, [us_mod_mean, ussr_mod_mean]):
        ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.3f}', ha='center', fontsize=10)
    
    ax2.set_ylabel('Frequency (normalized)', fontsize=11)
    ax2.set_title('Imperative modality\n(must, should, etc.)')
    ax2.grid(axis='y', alpha=0.3)
    
    plt.suptitle('Figure 2. Rhetorical strategies comparison')
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_temporal_dynamics(df: pd.DataFrame, save_path: str = None):
    """
    Plot temporal evolution of passive voice in US press.
    """
    if df.empty or len(df) < 3:
        print("Insufficient data for temporal plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    df = df.sort_values('date')
    df['rolling_mean'] = df['passive_no_agent_percent'].rolling(window=3, center=True).mean()
    
    ax.scatter(df['date'], df['passive_no_agent_percent'],
              color='#2d6a4f', s=80, alpha=0.7, edgecolors='white', linewidth=1.5,
              label='Individual articles')
    
    ax.plot(df['date'], df['rolling_mean'], color='#d4a373', linewidth=2.5,
            label='Rolling mean (3 articles)', marker='o', markersize=6)
    
    if len(df) > 2:
        x_numeric = np.arange(len(df))
        z = np.polyfit(x_numeric, df['passive_no_agent_percent'], 1)
        p = np.poly1d(z)
        ax.plot(df['date'], p(x_numeric), '--', color='#9e2a2b', linewidth=1.5, alpha=0.7,
                label=f'Linear trend (slope: {z[0]:.1f}% per period)')
    
    ax.set_xlabel('Date', fontsize=11)
    ax.set_ylabel('Passive without agent (%)', fontsize=11)
    ax.set_title('Figure 3. Temporal dynamics of passive voice in US press\n(each dot: one article)')
    ax.legend(loc='upper right')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()


def plot_ttr_comparison(df: pd.DataFrame, save_path: str = None):
    """
    Plot lexical diversity (TTR) comparison.
    """
    fig, ax = plt.subplots(figsize=(8, 6))
    
    us_ttr = df[df['country'] == 'US']['ttr'].dropna().values
    ussr_ttr = df[df['country'] == 'USSR']['ttr'].dropna().values
    
    us_mean = np.mean(us_ttr)
    ussr_mean = np.mean(ussr_ttr)
    
    us_lower, us_upper = bootstrap_ci(us_ttr)
    ussr_lower, ussr_upper = bootstrap_ci(ussr_ttr)
    
    bars = ax.bar(['USA', 'USSR'], [us_mean, ussr_mean],
                  color=['#90be6d', '#f9a26c'], edgecolor='black',
                  yerr=[[us_mean - us_lower, ussr_mean - ussr_lower],
                        [us_upper - us_mean, ussr_upper - ussr_mean]],
                  capsize=10, error_kw={'linewidth': 2})
    
    for bar, val in zip(bars, [us_mean, ussr_mean]):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                f'{val:.2f}', ha='center', fontsize=12, fontweight='bold')
    
    ax.set_ylabel('Type-Token Ratio (TTR)', fontsize=12)
    ax.set_ylim(0, 0.55)
    ax.set_title('Figure 4. Lexical diversity (TTR)\n(higher = more varied language)')
    ax.grid(axis='y', alpha=0.3)
    
    ax.text(0.5, -0.15, "TTR = unique words / total words\nLow TTR indicates formulaic, repetitive language",
            transform=ax.transAxes, ha='center', fontsize=9, style='italic')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150)
    plt.show()