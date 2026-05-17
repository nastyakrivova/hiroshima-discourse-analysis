"""
Sentiment and rhetorical analysis for historical texts.
Adapted for 1945 propaganda discourse with lexicon-based correction.
"""

import re
from typing import List, Dict, Tuple


ACCUSATORY_WORDS_EN = {
    'condemn': 2, 'denounce': 2, 'blame': 2, 'atrocity': 3,
    'crime': 3, 'barbaric': 3, 'inhuman': 3, 'massacre': 2,
    'criminal': 2, 'outrage': 2, 'indiscriminate': 1
}

ACCUSATORY_WORDS_RU = {
    'осудить': 2, 'разоблачить': 2, 'обвинить': 2, 'варвар': 3,
    'преступление': 3, 'бесчеловечный': 3, 'зверство': 3,
    'преступный': 2, 'возмутительный': 2, 'уничтожение': 1
}

# Modal imperative markers
MODAL_IMPERATIVE_EN = ['must', 'should', 'ought', 'need to', 'have to', 'required']
MODAL_IMPERATIVE_RU = ['должен', 'необходимо', 'нужно', 'обязан', 'следует', 'требуется']

JUSTIFICATION_MARKERS_EN = [
    'to end the war', 'to save lives', 'to avoid invasion',
    'military necessity', 'shorter war', 'surrender'
]
JUSTIFICATION_MARKERS_RU = [
    'ради победы', 'спасение жизней', 'избежать вторжения',
    'военная необходимость', 'сократить войну', 'капитуляция'
]


def analyze_accusatory_lexicon(text: str, language: str) -> Tuple[float, Dict]:
    """
    Count accusatory words and return normalized frequency.
    
    Args:
        text: Text string
        language: 'en' or 'ru'
        
    Returns:
        Tuple: (normalized frequency, word_counts dictionary)
    """
    text_lower = text.lower()
    lexicon = ACCUSATORY_WORDS_EN if language == 'en' else ACCUSATORY_WORDS_RU
    
    total_score = 0
    word_counts = {}
    
    for word, weight in lexicon.items():
        count = text_lower.count(word.lower())
        if count > 0:
            word_counts[word] = count
            total_score += count * weight
    
    # Normalize by text length (per 1000 chars)
    norm_freq = total_score / (len(text) / 1000) if len(text) > 0 else 0
    
    return norm_freq, word_counts


def analyze_modal_imperative(text: str, language: str) -> Tuple[float, List]:
    """
    Count modal imperative constructions.
    
    Args:
        text: Text string
        language: 'en' or 'ru'
        
    Returns:
        Tuple: (normalized frequency, list of found markers)
    """
    text_lower = text.lower()
    markers = MODAL_IMPERATIVE_EN if language == 'en' else MODAL_IMPERATIVE_RU
    
    found_markers = []
    for marker in markers:
        if marker in text_lower:
            found_markers.append(marker)
    
    count = len(found_markers)
    norm_freq = count / (len(text) / 1000) if len(text) > 0 else 0
    
    return norm_freq, found_markers


def check_justification(text: str, language: str) -> bool:
    """
    Check if text contains justification rhetoric.
    Used for post-processing sentiment predictions.
    
    Args:
        text: Text string
        language: 'en' or 'ru'
        
    Returns:
        True if justification markers found
    """
    text_lower = text.lower()
    markers = JUSTIFICATION_MARKERS_EN if language == 'en' else JUSTIFICATION_MARKERS_RU
    
    for marker in markers:
        if marker in text_lower:
            return True
    return False


def analyze_rhetoric_corpus(corpus: List[Dict], language: str) -> List[Dict]:
    """
    Analyze accusatory lexicon and modal markers for all articles.
    
    Args:
        corpus: List of article dicts from preprocessing
        language: 'en' or 'ru'
        
    Returns:
        List of articles with added rhetoric metrics
    """
    results = []
    
    for article in corpus:
        text = article['text']
        
        acc_freq, acc_words = analyze_accusatory_lexicon(text, language)
        mod_freq, mod_markers = analyze_modal_imperative(text, language)
        has_justification = check_justification(text, language)
        
        results.append({
            'filename': article['filename'],
            'date': article.get('date'),
            'accusatory_frequency': acc_freq,
            'accusatory_words': acc_words,
            'modal_frequency': mod_freq,
            'modal_markers': mod_markers,
            'has_justification': has_justification,
            'language': language
        })
    
    return results