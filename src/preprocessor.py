

import re
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from typing import List, Dict
from loguru import logger


class TextPreprocessor:
   

    def __init__(self):
        # NLTK data download karo (agar pehle nahi kiya)
        self._download_nltk_data()
        self.stop_words = set(stopwords.words('english'))
        logger.info("TextPreprocessor initialized ")

    def _download_nltk_data(self):
       
        packages = ['punkt', 'stopwords', 'punkt_tab']
        for pkg in packages:
            try:
                nltk.download(pkg, quiet=True)
            except Exception:
                pass

    def clean(self, text: str) -> str:
    
        if not text or not text.strip():
            raise ValueError(" Empty text provided")

        logger.info(f"Cleaning text ({len(text)} chars)...")

        # Pipeline — har step ek cleaning operation
        text = self._remove_extra_whitespace(text)
        text = self._fix_encoding_issues(text)
        text = self._remove_special_chars(text)
        text = self._normalize_spaces(text)

        logger.success(f"Cleaned text: {len(text)} chars remaining")
        return text

    def get_sentences(self, text: str) -> List[str]:
       
        sentences = sent_tokenize(text)
        # Empty sentences filter karo
        return [s.strip() for s in sentences if len(s.strip()) > 20]

    def get_stats(self, text: str) -> Dict:
       
        words = word_tokenize(text.lower())
        meaningful_words = [w for w in words if w.isalpha()]
        sentences = self.get_sentences(text)

        return {
            'char_count': len(text),
            'word_count': len(meaningful_words),
            'sentence_count': len(sentences),
            'unique_words': len(set(meaningful_words)),
            'avg_sentence_length': round(
                len(meaningful_words) / max(len(sentences), 1), 1
            ),
            'reading_time_min': round(len(meaningful_words) / 200, 1)
            # Average human reads ~200 words/min
        }


    def _remove_extra_whitespace(self, text: str) -> str:
        """Multiple newlines → single newline"""
        text = re.sub(r'\n{3,}', '\n\n', text)
        text = re.sub(r' {2,}', ' ', text)
        return text.strip()

    def _fix_encoding_issues(self, text: str) -> str:
    
        replacements = {
            '\u2019': "'",   
            '\u2018': "'",   
            '\u201c': '"',   
            '\u201d': '"',   
            '\u2013': '-',   
            '\u2014': '-',   
            '\xa0': ' ',     
        }
        for old, new in replacements.items():
            text = text.replace(old, new)
        return text

    def _remove_special_chars(self, text: str) -> str:
       
      
        text = re.sub(r'Page \d+ of \d+', '', text, flags=re.IGNORECASE)
        text = re.sub(r'^\d+\s*$', '', text, flags=re.MULTILINE)
        
        text = re.sub(r'[-_=]{3,}', '', text)
        return text

    def _normalize_spaces(self, text: str) -> str:
      
        lines = [line.strip() for line in text.split('\n')]
        lines = [line for line in lines if line]  # Empty lines hataao
        return '\n'.join(lines)