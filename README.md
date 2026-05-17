# Hiroshima Discourse Analysis

**Comparative NLP analysis of US and Soviet newspaper discourse on the atomic bombings of Hiroshima and Nagasaki (August–September 1945)**

[![Python](https://img.shields.io/badge/Python-3.9%2B-blue)](https://www.python.org/)
[![License](https://img.shields.io/badge/License-MIT-green)](LICENSE)

## 📋 About the Project

This project analyzes how US and Soviet newspapers covered the atomic bombings of Hiroshima and Nagasaki in August–September 1945. Using computational linguistics methods (NLP), we compare discursive strategies across 60 newspaper articles (30 from each side).

**Main hypothesis:** The grammar of propaganda differs systematically between the party performing an action (US) and the party accusing them (USSR). We examine passive voice patterns, accusatory lexicon, modal markers, and lexical diversity.

**Key findings (preliminary):**
- Higher passive voice frequency in US texts (34.2% vs 18.3%) — explained by stylistic/genre differences, not evasion of responsibility
- Accusatory lexicon is nearly 4x higher in Soviet texts (0.22 vs 0.06)
- Temporal trend: declining passive voice in US press from August to September 1945
- Lexical diversity (TTR) significantly lower in Soviet texts (0.31 vs 0.42), reflecting centralized journalism

## 📊 Dataset

| Parameter | US | USSR |
|-----------|-----|------|
| Number of articles | 30 | 30 |
| Sources | *The New York Times*, *The Washington Post*, *Chicago Tribune* | *Pravda*, *Izvestia*, *Krasnaya Zvezda* |
| Time period | August 6 – September 15, 1945 | August 6 – September 15, 1945 |
| Language | English | Russian |
| Text format | TXT (from Chronicling America) | TXT (manual collection) |

## 🔧 Installation

```bash
# Clone the repository
git clone https://github.com/your_username/hiroshima-discourse-analysis.git
cd hiroshima-discourse-analysis

# Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt