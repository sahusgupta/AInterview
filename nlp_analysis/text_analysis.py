import spacy
import numpy as np
from typing import Dict, List, Tuple, Optional
from collections import Counter
import re
from scipy.stats import entropy
from textblob import TextBlob

class TranscriptAnalyzer:
    def __init__(self):
        """Initialize the transcript analyzer with NLP models and tools."""
        # Load spaCy model for linguistic analysis
        self.nlp = spacy.load("en_core_web_sm")
        
        # Initialize feature extractors
        self.min_segment_length = 50  # minimum words for segment analysis
        self.max_segment_length = 200  # maximum words for segment analysis

    def analyze_transcript(self, text: str) -> Dict[str, float]:
        """
        Analyze transcript text and extract features for AI detection.
        
        Args:
            text (str): The transcript text to analyze
            
        Returns:
            Dict[str, float]: Dictionary of extracted features and their values
        """
        # Basic text cleaning
        cleaned_text = self._clean_text(text)
        
        # Process with spaCy
        doc = self.nlp(cleaned_text)
        
        # Extract features
        features = {
            # Linguistic diversity features
            'lexical_diversity': self._calculate_lexical_diversity(doc),
            'syntactic_complexity': self._calculate_syntactic_complexity(doc),
            'response_coherence': self._calculate_response_coherence(doc),
            
            # Statistical features
            'word_distribution': self._analyze_word_distribution(doc),
            'sentence_variance': self._calculate_sentence_variance(doc),
            
            # Semantic features
            'semantic_consistency': self._analyze_semantic_consistency(doc),
            'topic_coherence': self._calculate_topic_coherence(doc),
            
            # Temporal features
            'pause_pattern_score': self._analyze_pause_patterns(text),
            'speech_rate_consistency': self._analyze_speech_rate(text)
        }
        
        return features

    def _clean_text(self, text: str) -> str:
        """Clean and normalize transcript text."""
        # Remove timestamps and speaker labels
        text = re.sub(r'\[\d{2}:\d{2}\]', '', text)
        text = re.sub(r'Speaker \d+:', '', text)
        
        # Normalize whitespace
        text = re.sub(r'\s+', ' ', text)
        return text.strip()

    def _calculate_lexical_diversity(self, doc) -> float:
        """
        Calculate lexical diversity using multiple metrics.
        Returns a score between 0 (low diversity) and 1 (high diversity).
        """
        words = [token.text.lower() for token in doc if token.is_alpha]
        if not words:
            return 0.0
            
        # Calculate Type-Token Ratio (TTR)
        unique_words = len(set(words))
        total_words = len(words)
        ttr = unique_words / total_words if total_words > 0 else 0
        
        # Calculate Moving Average TTR (MATTR)
        window_size = 50
        if len(words) >= window_size:
            mattrs = []
            for i in range(len(words) - window_size + 1):
                window = words[i:i + window_size]
                window_ttr = len(set(window)) / window_size
                mattrs.append(window_ttr)
            mattr = np.mean(mattrs)
        else:
            mattr = ttr
            
        # Combine metrics
        diversity_score = (ttr + mattr) / 2
        return min(1.0, diversity_score)

    def _calculate_syntactic_complexity(self, doc) -> float:
        """
        Analyze syntactic complexity using dependency parsing.
        Returns a score between 0 (simple) and 1 (complex).
        """
        if not doc:
            return 0.0
            
        # Calculate average dependency tree depth
        depths = []
        for sent in doc.sents:
            depth = max(len(list(token.ancestors)) for token in sent)
            depths.append(depth)
            
        if not depths:
            return 0.0
            
        # Normalize complexity score
        avg_depth = np.mean(depths)
        complexity_score = min(1.0, avg_depth / 10)  # Normalize to 0-1 range
        
        return complexity_score

    def _calculate_response_coherence(self, doc) -> float:
        """
        Analyze coherence between sentences using semantic similarity.
        Returns a score between 0 (incoherent) and 1 (coherent).
        """
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return 1.0  # Single sentence is considered coherent
            
        # Calculate semantic similarity between adjacent sentences
        similarities = []
        for i in range(len(sentences) - 1):
            similarity = sentences[i].similarity(sentences[i + 1])
            similarities.append(similarity)
            
        # Average similarity score
        coherence_score = np.mean(similarities) if similarities else 0.0
        return coherence_score

    def _analyze_word_distribution(self, doc) -> float:
        """
        Analyze word frequency distribution for unusual patterns.
        Returns a score between 0 (natural distribution) and 1 (unusual distribution).
        """
        words = [token.text.lower() for token in doc if token.is_alpha]
        if not words:
            return 0.0
            
        # Calculate word frequencies
        word_freq = Counter(words)
        frequencies = np.array(list(word_freq.values()))
        
        # Calculate entropy of distribution
        prob_dist = frequencies / len(words)
        distribution_entropy = entropy(prob_dist)
        
        # Normalize score (higher entropy = more natural)
        normalized_score = 1 - min(1.0, distribution_entropy / 4)  # 4 is typical entropy for natural text
        return normalized_score

    def _calculate_sentence_variance(self, doc) -> float:
        """
        Analyze variance in sentence structure and length.
        Returns a score between 0 (high variance/natural) and 1 (low variance/suspicious).
        """
        sentences = list(doc.sents)
        if len(sentences) < 2:
            return 0.0
            
        # Calculate sentence lengths
        lengths = [len(sent) for sent in sentences]
        
        # Calculate variance in lengths
        variance = np.var(lengths)
        mean_length = np.mean(lengths)
        
        # Normalize variance score
        normalized_variance = min(1.0, variance / (mean_length * 2))
        variance_score = 1 - normalized_variance  # Invert so high variance = low score
        
        return variance_score

    def _analyze_semantic_consistency(self, doc) -> float:
        """
        Analyze semantic consistency across the text.
        Returns a score between 0 (natural variation) and 1 (suspicious consistency).
        """
        # Split into segments
        segments = list(doc.sents)
        if len(segments) < 2:
            return 0.0
            
        # Calculate semantic similarity between all segment pairs
        similarities = []
        for i in range(len(segments)):
            for j in range(i + 1, len(segments)):
                similarity = segments[i].similarity(segments[j])
                similarities.append(similarity)
                
        # Calculate variance in similarities
        variance = np.var(similarities) if similarities else 0
        
        # Normalize score (higher variance = more natural)
        consistency_score = 1 - min(1.0, variance * 2)
        return consistency_score

    def _calculate_topic_coherence(self, doc) -> float:
        """
        Analyze topic consistency and natural topic flow.
        Returns a score between 0 (natural flow) and 1 (suspicious consistency).
        """
        # Extract main topics (nouns and proper nouns)
        topics = [token.text for token in doc if token.pos_ in ('NOUN', 'PROPN')]
        if not topics:
            return 0.0
            
        # Calculate topic distribution
        topic_freq = Counter(topics)
        frequencies = np.array(list(topic_freq.values()))
        
        # Calculate entropy of topic distribution
        prob_dist = frequencies / len(topics)
        topic_entropy = entropy(prob_dist)
        
        # Normalize score (higher entropy = more natural)
        coherence_score = 1 - min(1.0, topic_entropy / 3)
        return coherence_score

    def _analyze_pause_patterns(self, text: str) -> float:
        """
        Analyze patterns in pauses and hesitations.
        Returns a score between 0 (natural patterns) and 1 (suspicious patterns).
        """
        # Look for pause markers like "...", "uh", "um"
        pause_markers = re.findall(r'\.{3}|(?:uh|um|er|ah)\b', text.lower())
        
        if not pause_markers:
            return 0.5  # Neutral score if no pause markers
            
        # Calculate pause frequency
        words = text.split()
        pause_frequency = len(pause_markers) / len(words) if words else 0
        
        # Normalize score (very low or very high frequencies are suspicious)
        normalized_score = abs(pause_frequency - 0.05) * 10  # 0.05 is typical human pause frequency
        return min(1.0, normalized_score)

    def _analyze_speech_rate(self, text: str) -> float:
        """
        Analyze consistency in speech rate and rhythm.
        Returns a score between 0 (natural variation) and 1 (suspicious consistency).
        """
        # Split into segments
        segments = re.split(r'[.!?]+', text)
        segments = [s.strip() for s in segments if s.strip()]
        
        if len(segments) < 2:
            return 0.0
            
        # Calculate words per segment
        words_per_segment = [len(s.split()) for s in segments]
        
        # Calculate variance in speech rate
        variance = np.var(words_per_segment)
        mean_rate = np.mean(words_per_segment)
        
        # Normalize score (higher variance = more natural)
        normalized_variance = min(1.0, variance / (mean_rate * 2))
        rate_score = 1 - normalized_variance
        
        return rate_score

    def get_detailed_analysis(self, features: Dict[str, float]) -> Dict[str, str]:
        """
        Provide detailed analysis of extracted features.
        
        Args:
            features: Dictionary of feature scores
            
        Returns:
            Dict[str, str]: Detailed explanations of feature values
        """
        analysis = {}
        
        for feature, score in features.items():
            if score > 0.8:
                confidence = "High"
            elif score > 0.6:
                confidence = "Moderate"
            else:
                confidence = "Low"
                
            analysis[feature] = f"{confidence} likelihood of AI generation ({score:.2f})"
            
        return analysis

    def get_summary_report(self, features: Dict[str, float]) -> str:
        """
        Generate a summary report of the analysis.
        
        Args:
            features: Dictionary of feature scores
            
        Returns:
            str: Summary report of the analysis
        """
        # Calculate overall confidence
        overall_score = np.mean(list(features.values()))
        
        # Determine key indicators
        key_indicators = [
            (k, v) for k, v in features.items()
            if v > 0.7  # High confidence threshold
        ]
        
        # Generate report
        report = [
            f"Overall AI Detection Confidence: {overall_score:.2%}",
            "\nKey Indicators:"
        ]
        
        for indicator, score in key_indicators:
            report.append(f"- {indicator}: {score:.2%}")
            
        return "\n".join(report)
