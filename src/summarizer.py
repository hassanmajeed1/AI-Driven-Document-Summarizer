

from transformers import pipeline, AutoTokenizer
from sklearn.feature_extraction.text import TfidfVectorizer
import numpy as np
from typing import List, Optional
from loguru import logger
import warnings
warnings.filterwarnings('ignore')


class DocumentSummarizer:
   

    MODELS = {
        'fast':    'sshleifer/distilbart-cnn-12-6',   
        'quality': 'facebook/bart-large-cnn',          
        't5':      'google/flan-t5-base',             
    }

    def __init__(self, model_type: str = 'fast'):
       
        self.model_type = model_type
        self._abstractive_pipeline = None  
        logger.info(f"Summarizer ready (model: {model_type}) ")

    def extractive_summary(
        self,
        text: str,
        sentences: List[str],
        num_sentences: int = 5
    ) -> str:
       
        if len(sentences) <= num_sentences:
            return " ".join(sentences)

        logger.info(f"Running extractive summarization on {len(sentences)} sentences...")

       
        vectorizer = TfidfVectorizer(
            stop_words='english',
            ngram_range=(1, 2)
        )

        try:
            tfidf_matrix = vectorizer.fit_transform(sentences)
        except ValueError:
            logger.warning("TF-IDF failed, returning first N sentences")
            return " ".join(sentences[:num_sentences])

        
        scores = np.array(tfidf_matrix.sum(axis=1)).flatten()

        
        top_indices = sorted(
            np.argsort(scores)[-num_sentences:].tolist()
        )

        summary_sentences = [sentences[i] for i in top_indices]
        return " ".join(summary_sentences)

    def abstractive_summary(
        self,
        text: str,
        max_length: int = 150,
        min_length: int = 50
    ) -> str:
       
        
        if self._abstractive_pipeline is None:
            logger.info(f"Loading AI model: {self.model_type} on GPU...")
            logger.info("(First time mein thodi der lagegi — downloading model)")
            
            model_name = self.MODELS[self.model_type]
            self._abstractive_pipeline = pipeline(
                "summarization",
                model=model_name,
                device=0 
            )
            logger.success("AI Model loaded on GPU ")

       
        text_chunks = self._chunk_text(text, max_tokens=900)
        logger.info(f"Processing {len(text_chunks)} chunk(s)...")

        summaries = []
        for i, chunk in enumerate(text_chunks):
            logger.info(f"  Summarizing chunk {i+1}/{len(text_chunks)}...")
            result = self._abstractive_pipeline(
                chunk,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            summaries.append(result[0]['summary_text'])

        if len(summaries) > 1:
            combined = " ".join(summaries)
            logger.info("Creating final summary from chunks...")
            final = self._abstractive_pipeline(
                combined,
                max_length=max_length,
                min_length=min_length,
                do_sample=False,
                truncation=True
            )
            return final[0]['summary_text']

        return summaries[0]

   

    def _chunk_text(self, text: str, max_tokens: int = 900) -> List[str]:
        
        words = text.split()
        chunks = []
        current_chunk = []
        current_count = 0

        for word in words:
            current_chunk.append(word)
            current_count += 1

            if current_count >= max_tokens:
                chunks.append(" ".join(current_chunk))
                current_chunk = []
                current_count = 0

        if current_chunk:
            chunks.append(" ".join(current_chunk))

        return chunks if chunks else [text]