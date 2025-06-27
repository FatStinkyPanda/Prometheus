# backend/io_systems/natural_language_processor.py

import spacy
from spacy.lang.en.stop_words import STOP_WORDS
from string import punctuation
from collections import Counter
from typing import Dict, Any, List, Tuple

from backend.utils.logger import Logger
from backend.io_systems.io_types import InputPayload, OutputPayload, InputType, OutputType


class NaturalLanguageProcessor:
    """
    Handles natural language understanding and generation tasks.

    This class acts as the primary interface for processing raw text input into a
    structured format for the minds, and for formatting system outputs into
    natural language for the user. It leverages a spaCy pipeline for robust
    linguistic analysis.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the NaturalLanguageProcessor.

        Args:
            config (Dict[str, Any]): The I/O systems configuration dictionary.
        """
        self.logger = Logger(__name__)
        self.nlp = None
        self.config = config.get('nlp', {})
        self.model_name = self.config.get('model', 'en_core_web_lg')
        self._load_model()

    def _load_model(self):
        """Loads the spaCy language model."""
        self.logger.info(f"Loading spaCy model: {self.model_name}...")
        try:
            self.nlp = spacy.load(self.model_name)
            self.logger.info(f"spaCy model '{self.model_name}' loaded successfully.")
        except OSError:
            self.logger.critical(
                f"spaCy model '{self.model_name}' not found. "
                f"Please run 'python -m spacy download {self.model_name}' "
                "from your virtual environment's terminal."
            )
            # This is a fatal error for this component, so we raise it.
            raise RuntimeError(f"Could not load required spaCy model: {self.model_name}")
        except Exception as e:
            self.logger.critical(f"An unexpected error occurred while loading the spaCy model: {e}", exc_info=True)
            raise

    def preprocess_input(self, payload: InputPayload) -> Dict[str, Any]:
        """
        Analyzes and structures raw text input for the cognitive minds.

        Args:
            payload (InputPayload): The input payload containing the text.

        Returns:
            A dictionary containing the original text, cleaned text, named entities,
            and other linguistic features. Returns an empty dict for non-text input.
        """
        if payload.type != InputType.TEXT:
            return {}

        text = payload.content
        self.logger.debug(f"Preprocessing text: '{text[:100]}...'")
        
        if not text or not isinstance(text, str):
            self.logger.warning("preprocess_input called with empty or invalid text.")
            return {"original_text": "", "cleaned_text": "", "entities": [], "root_action": None}

        try:
            doc = self.nlp(text)
            
            entities = [(ent.text, ent.label_) for ent in doc.ents]
            
            # Find the root verb of the main clause as a proxy for intent
            root_action = None
            for token in doc:
                if token.dep_ == "ROOT" and token.pos_ == "VERB":
                    root_action = token.lemma_
                    break

            # A simple "cleaned" version could be just the lemmatized text without stopwords
            cleaned_tokens = [token.lemma_.lower() for token in doc if not token.is_stop and not token.is_punct]
            cleaned_text = " ".join(cleaned_tokens)
            
            processed_data = {
                "original_text": text,
                "cleaned_text": cleaned_text,
                "entities": entities,
                "root_action": root_action,
                "language": doc.lang_
            }
            self.logger.debug(f"Preprocessing complete. Entities found: {len(entities)}. Root: {root_action}")
            return processed_data

        except Exception as e:
            self.logger.error(f"An error occurred during NLP preprocessing: {e}", exc_info=True)
            return {"original_text": text, "error": str(e)}

    def summarize(self, text: str, ratio: float = 0.2) -> str:
        """
        Performs extractive summarization on a block of text.

        Args:
            text (str): The text to summarize.
            ratio (float): The ratio of sentences to return (e.g., 0.2 for 20%).

        Returns:
            A string containing the summarized text.
        """
        if not text or not isinstance(text, str):
            self.logger.warning("Summarize called with empty or invalid text.")
            return ""

        self.logger.debug(f"Summarizing text of length {len(text)} with ratio {ratio}.")
        
        try:
            doc = self.nlp(text)
            
            # 1. Tokenize into words, filter stopwords and punctuation
            keywords = [token.text.lower() for token in doc if not token.is_stop and not token.is_punct]
            word_frequencies = Counter(keywords)
            
            # 2. Normalize frequencies
            max_frequency = max(word_frequencies.values()) if word_frequencies else 0
            if max_frequency == 0: return "" # Avoid division by zero for empty/stopword-only text
            
            for word in word_frequencies.keys():
                word_frequencies[word] = (word_frequencies[word] / max_frequency)

            # 3. Score sentences based on word frequencies
            sentence_scores = {}
            for sent in doc.sents:
                for word in sent:
                    if word.text.lower() in word_frequencies:
                        if sent in sentence_scores:
                            sentence_scores[sent] += word_frequencies[word.text.lower()]
                        else:
                            sentence_scores[sent] = word_frequencies[word.text.lower()]

            # 4. Select top N sentences
            summary_length = int(len(list(doc.sents)) * ratio)
            if summary_length < 1 and len(list(doc.sents)) > 0:
                summary_length = 1 # Always return at least one sentence if possible
            
            summarized_sentences = sorted(sentence_scores, key=sentence_scores.get, reverse=True)[:summary_length]
            
            # 5. Reorder sentences back to their original order for coherence
            final_sentences = sorted(summarized_sentences, key=lambda s: s.start)
            
            summary = " ".join([sent.text for sent in final_sentences])
            self.logger.info(f"Successfully summarized text from {len(text)} to {len(summary)} characters.")
            return summary

        except Exception as e:
            self.logger.error(f"An error occurred during summarization: {e}", exc_info=True)
            return f"Error during summarization: {e}"

    def extract_entities(self, text: str) -> List[Tuple[str, str]]:
        """A simple wrapper to extract named entities from text."""
        if not text or not isinstance(text, str): return []
        try:
            doc = self.nlp(text)
            return [(ent.text, ent.label_) for ent in doc.ents]
        except Exception as e:
            self.logger.error(f"Failed to extract entities: {e}", exc_info=True)
            return []