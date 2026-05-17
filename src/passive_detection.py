"""
Passive voice detection for English and Russian texts.
Hybrid approach: spaCy for English, pymorphy2 + rules for Russian.
"""

import re
import spacy
import pymorphy2
from typing import Tuple, Dict, List

try:
    nlp_en = spacy.load('en_core_web_sm')
except OSError:
    print("Error: spaCy model 'en_core_web_sm' not found.")
    print("Run: python -m spacy download en_core_web_sm")
    nlp_en = None

morph_ru = pymorphy2.MorphAnalyzer()


def detect_passive_english(sentence: str, check_agent: bool = True) -> Tuple[bool, bool, str]:
    """
    Detect passive voice in English sentence using spaCy.
    
    Args:
        sentence: English text string
        check_agent: If True, also check whether agent (by + NP) is present
        
    Returns:
        Tuple: (is_passive, has_agent, passive_type)
        passive_type: 'full' (with agent), 'truncated' (without agent), or 'none'
    """
    if nlp_en is None:
        return False, False, 'none'
    
    doc = nlp_en(sentence[:5000])  # Limit length
    has_passive = False
    has_agent = False
    
    for token in doc:
        if token.dep_ == 'nsubjpass' or (token.tag_ == 'VBN' and token.head.dep_ == 'auxpass'):
            has_passive = True
            
            for child in token.head.children:
                if child.lower_ == 'by' and child.dep_ == 'agent':
                    has_agent = True
                    break
    
    if not has_passive:
        return False, False, 'none'
    elif has_agent:
        return True, True, 'full'
    else:
        return True, False, 'truncated'


def detect_passive_russian(sentence: str, check_agent: bool = True) -> Tuple[bool, bool, str]:
    """
    Detect passive voice in Russian sentence using pymorphy2 and rules.
    
    Args:
        sentence: Russian text string
        check_agent: If True, also check for agent (instrumental case noun)
        
    Returns:
        Tuple: (is_passive, has_agent, passive_type)
    """
    words = sentence.lower().split()
    has_passive = False
    has_agent = False
    
    passive_participle_pattern = re.compile(
        r'(был|была|было|были)\s+(\w+[а-я]?[нт])'
    )
    if passive_participle_pattern.search(sentence):
        has_passive = True
    
    for word in words:
        if ('ся' in word or 'сь' in word) and len(word) > 4:
            parsed = morph_ru.parse(word)[0]
            if 'VERB' in parsed.tag and '3per' in parsed.tag:
                has_passive = True
    
    impersonal_markers = ['сообщаетс', 'подвергл', 'разрушаетс', 'уничтожаетс']
    for marker in impersonal_markers:
        if marker in sentence:
            has_passive = True
            break
    
    if check_agent and has_passive:
        agent_markers = ['американц', 'сша', 'америк', 'союзник', 'трумэн']
        for marker in agent_markers:
            if marker in sentence:
                has_agent = True
                break
    
    if not has_passive:
        return False, False, 'none'
    elif has_agent:
        return True, True, 'full'
    else:
        return True, False, 'truncated'


def analyze_passive_corpus(corpus: List[Dict], language: str) -> List[Dict]:
    """
    Analyze passive voice for all articles in corpus.
    
    Args:
        corpus: List of article dicts from preprocessing
        language: 'en' or 'ru'
        
    Returns:
        List of articles with added passive metrics
    """
    results = []
    detect_func = detect_passive_english if language == 'en' else detect_passive_russian
    
    for article in corpus:
        sentences = re.split(r'[.!?]', article['text'])
        sentences = [s.strip() for s in sentences if len(s.strip()) > 20]
        
        passive_count = 0
        passive_no_agent_count = 0
        passive_with_agent_count = 0
        
        for sent in sentences:
            is_pass, has_agent, ptype = detect_func(sent)
            if is_pass:
                passive_count += 1
                if has_agent:
                    passive_with_agent_count += 1
                else:
                    passive_no_agent_count += 1
        
        total_sentences = len(sentences)
        passive_percent = (passive_count / total_sentences * 100) if total_sentences > 0 else 0
        passive_no_agent_percent = (passive_no_agent_count / total_sentences * 100) if total_sentences > 0 else 0
        
        results.append({
            'filename': article['filename'],
            'date': article.get('date'),
            'total_sentences': total_sentences,
            'passive_count': passive_count,
            'passive_no_agent_count': passive_no_agent_count,
            'passive_percent': passive_percent,
            'passive_no_agent_percent': passive_no_agent_percent,
            'language': language
        })
    
    return results