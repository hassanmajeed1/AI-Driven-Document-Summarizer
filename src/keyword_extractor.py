

from keybert import KeyBERT
from typing import List, Tuple, Dict
from loguru import logger


class KeywordExtractor:
    

    def __init__(self):
        logger.info("Loading KeyBERT model...")
        self.model = KeyBERT(model='all-MiniLM-L6-v2')
        logger.success("KeyBERT ready ")

    def extract_keywords(
        self,
        text: str,
        top_n: int = 10,
        diversity: float = 0.5
    ) -> List[Tuple[str, float]]:
      
        logger.info(f"Extracting top {top_n} keywords...")

        keywords = self.model.extract_keywords(
            text,
            keyphrase_ngram_range=(1, 3),   
            stop_words='english',
            use_mmr=True,                    
            diversity=diversity,
            top_n=top_n
        )

        logger.success(f"Extracted {len(keywords)} keywords")
        return keywords

    def categorize_keywords(
        self,
        keywords: List[Tuple[str, float]]
    ) -> Dict[str, List[str]]:
       
        categorized = {
            'high_relevance': [],
            'medium_relevance': [],
            'low_relevance': []
        }

        for keyword, score in keywords:
            if score >= 0.7:
                categorized['high_relevance'].append(keyword)
            elif score >= 0.4:
                categorized['medium_relevance'].append(keyword)
            else:
                categorized['low_relevance'].append(keyword)

        return categorized