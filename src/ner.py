

import spacy
from collections import defaultdict
from typing import Dict, List
from loguru import logger


class EntityExtractor:
   

    
    BUSINESS_ENTITIES = {
        'ORG':     ' Organizations',
        'PERSON':  ' People',
        'MONEY':   ' Financial Figures',
        'DATE':    ' Dates',
        'GPE':     ' Locations',
        'PERCENT': ' Percentages',
        'PRODUCT': ' Products',
    }

    def __init__(self):
        logger.info("Loading SpaCy model...")
        try:
            self.nlp = spacy.load("en_core_web_sm")
        except OSError:
            logger.warning("SpaCy model not found. Downloading...")
            import subprocess
            subprocess.run(["python", "-m", "spacy", "download", "en_core_web_sm"])
            self.nlp = spacy.load("en_core_web_sm")
        logger.success("SpaCy NER ready ")

    def extract_entities(self, text: str) -> Dict[str, List[str]]:
      
       
        text = text[:1000000]  

        logger.info("Running NER pipeline...")
        doc = self.nlp(text)

        entities = defaultdict(set)  

        for ent in doc.ents:
            if ent.label_ in self.BUSINESS_ENTITIES:
                
                entity_text = ent.text.strip()
                if len(entity_text) > 1:  
                    entities[ent.label_].add(entity_text)

        
        result = {
            label: sorted(list(items))
            for label, items in entities.items()
        }

        total = sum(len(v) for v in result.values())
        logger.success(f"Found {total} entities across {len(result)} categories")
        return result

    def get_entity_summary(self, entities: Dict[str, List[str]]) -> str:
        
        if not entities:
            return "No significant entities found."

        lines = []
        for label, items in entities.items():
            if items and label in self.BUSINESS_ENTITIES:
                emoji_label = self.BUSINESS_ENTITIES[label]
                items_str = ", ".join(items[:5])  # Max 5 per category
                if len(items) > 5:
                    items_str += f" (+{len(items)-5} more)"
                lines.append(f"**{emoji_label}:** {items_str}")

        return "\n".join(lines)