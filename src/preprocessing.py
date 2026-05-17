"""
Text preprocessing and filtering for newspaper articles.
Functions for loading, cleaning, and filtering relevant sentences.
"""

import os
import re
from typing import List, Dict, Tuple


def load_corpus(folder_path: str) -> List[Dict]:
    """
    Load all text files from a folder.
    
    Args:
        folder_path: Path to folder containing .txt files
        
    Returns:
        List of dicts with filename, date (if parsed), and text content
    """
    corpus = []
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt'):
            filepath = os.path.join(folder_path, filename)
            with open(filepath, 'r', encoding='utf-8') as f:
                text = f.read()
            
            # Try to extract date from filename (format: source_YYYY-MM-DD.txt)
            date_match = re.search(r'(\d{4}-\d{2}-\d{2})', filename)
            date = date_match.group(1) if date_match else None
            
            corpus.append({
                'filename': filename,
                'date': date,
                'text': text
            })
    return corpus


def clean_text(text: str) -> str:
    """
    Basic text cleaning: normalize whitespace, remove headers/footers.
    
    Args:
        text: Raw text string
        
    Returns:
        Cleaned text string
    """
    text = re.sub(r'\s+', ' ', text)
    lines = text.split('. ')
    cleaned_lines = [line for line in lines if len(line) > 25]
    return '. '.join(cleaned_lines)


def get_keyword_markers(language: str) -> List[str]:
    """
    Return keyword markers for filtering relevant sentences.
    
    Args:
        language: 'en' for English, 'ru' for Russian
        
    Returns:
        List of keyword strings
    """
    if language == 'en':
        return [
            'hiroshima', 'nagasaki', 'atomic bomb', 'atom bomb',
            'nuclear weapon', 'blast', 'radiation', 'destroyed',
            'killed', 'victims', 'casualties', 'devastation',
            'surrender', 'japan', 'war ended'
        ]
    elif language == 'ru':
        return [
            'хиросим', 'нагасак', 'атомн', 'бомб',
            'разрушен', 'убит', 'жертв', 'варвар',
            'новое оружие', 'капитуляц', 'япони'
        ]
    else:
        raise ValueError(f"Unsupported language: {language}")


def filter_relevant_sentences(corpus: List[Dict], language: str) -> List[Dict]:
    """
    Filter only sentences containing relevant keywords.
    
    Args:
        corpus: List of article dicts from load_corpus
        language: 'en' or 'ru'
        
    Returns:
        Same structure but with text replaced by filtered sentences
    """
    keywords = get_keyword_markers(language)
    pattern = re.compile('|'.join(keywords), re.IGNORECASE)
    
    filtered_corpus = []
    for article in corpus:
        sentences = re.split(r'[.!?]', article['text'])
        relevant_sentences = []
        
        for sent in sentences:
            sent = sent.strip()
            if len(sent) > 25 and pattern.search(sent):
                relevant_sentences.append(sent)
        
        if relevant_sentences:
            filtered_article = article.copy()
            filtered_article['text'] = ' '.join(relevant_sentences)
            filtered_article['sentence_count'] = len(relevant_sentences)
            filtered_corpus.append(filtered_article)
    
    return filtered_corpus