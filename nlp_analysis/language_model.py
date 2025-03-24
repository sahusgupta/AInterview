import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import numpy as np
from typing import Dict, List, Tuple
import re

class AIDetectionModel:
    def __init__(self):
        """Initialize the AI detection model with pre-trained transformers."""
        # Load pre-trained model and tokenizer for AI text detection
        self.model_name = "roberta-base"  # Can be replaced with a fine-tuned model
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        self.model = AutoModelForSequenceClassification.from_pretrained(self.model_name, num_labels=2)
        
        # Thresholds for different features
        self.repetition_threshold = 0.3
        self.complexity_threshold = 0.7
        self.consistency_threshold = 0.8

    def analyze_text(self, text: str) -> Dict[str, float]:
        """
        Analyze text for AI-generated patterns and return confidence scores.
        
        Args:
            text (str): The transcript text to analyze
            
        Returns:
            Dict[str, float]: Dictionary containing various confidence scores
        """
        # Clean and preprocess the text
        cleaned_text = self._preprocess_text(text)
        
        # Extract various features
        repetition_score = self._analyze_repetition(cleaned_text)
        complexity_score = self._analyze_complexity(cleaned_text)
        consistency_score = self._analyze_consistency(cleaned_text)
        perplexity_score = self._calculate_perplexity(cleaned_text)
        
        # Get model prediction
        model_confidence = self._get_model_prediction(cleaned_text)
        
        # Combine scores into final confidence
        final_confidence = self._combine_scores([
            (repetition_score, 0.2),
            (complexity_score, 0.2),
            (consistency_score, 0.2),
            (perplexity_score, 0.2),
            (model_confidence, 0.2)
        ])
        
        return {
            'final_confidence': final_confidence,
            'repetition_score': repetition_score,
            'complexity_score': complexity_score,
            'consistency_score': consistency_score,
            'perplexity_score': perplexity_score,
            'model_confidence': model_confidence
        }

    def _preprocess_text(self, text: str) -> str:
        """Clean and normalize text for analysis."""
        # Remove special characters and normalize whitespace
        text = re.sub(r'[^\w\s.,!?]', '', text)
        text = re.sub(r'\s+', ' ', text)
        return text.strip().lower()

    def _analyze_repetition(self, text: str) -> float:
        """
        Analyze text for unusual repetition patterns common in AI-generated text.
        Returns a score between 0 (human-like) and 1 (likely AI).
        """
        words = text.split()
        if not words:
            return 0.0
            
        # Check for repeated phrases
        phrase_counts = {}
        for i in range(len(words) - 2):
            phrase = ' '.join(words[i:i+3])
            phrase_counts[phrase] = phrase_counts.get(phrase, 0) + 1
            
        # Calculate repetition score
        max_repetition = max(phrase_counts.values()) if phrase_counts else 0
        repetition_score = min(1.0, max_repetition / (len(words) / 10))
        
        return repetition_score

    def _analyze_complexity(self, text: str) -> float:
        """
        Analyze linguistic complexity patterns.
        Returns a score between 0 (varied/human-like) and 1 (suspiciously consistent).
        """
        sentences = text.split('.')
        if not sentences:
            return 0.0
            
        # Calculate sentence length variance
        lengths = [len(s.split()) for s in sentences if s.strip()]
        if not lengths:
            return 0.0
            
        mean_length = np.mean(lengths)
        variance = np.var(lengths)
        
        # Normalize variance (higher variance = more human-like)
        normalized_variance = min(1.0, variance / (mean_length * 2))
        complexity_score = 1 - normalized_variance
        
        return complexity_score

    def _analyze_consistency(self, text: str) -> float:
        """
        Analyze consistency in writing style and vocabulary usage.
        Returns a score between 0 (varied/human-like) and 1 (suspiciously consistent).
        """
        words = text.split()
        if not words:
            return 0.0
            
        # Calculate vocabulary diversity
        unique_words = len(set(words))
        total_words = len(words)
        
        # Type-Token Ratio (TTR)
        ttr = unique_words / total_words if total_words > 0 else 0
        
        # Normalize TTR (lower TTR = more AI-like)
        consistency_score = 1 - min(1.0, ttr * 2)
        
        return consistency_score

    def _calculate_perplexity(self, text: str) -> float:
        """
        Calculate language model perplexity as a measure of text naturalness.
        Returns a score between 0 (natural/human-like) and 1 (potentially AI).
        """
        # Tokenize text
        encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        # Calculate perplexity using model
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            
        # Convert logits to probability score
        probs = torch.softmax(logits, dim=1)
        ai_prob = probs[0][1].item()  # Probability of AI-generated class
        
        return ai_prob

    def _get_model_prediction(self, text: str) -> float:
        """
        Get direct model prediction for AI-generated text.
        Returns a confidence score between 0 (human) and 1 (AI).
        """
        # Tokenize and get model prediction
        encodings = self.tokenizer(text, return_tensors='pt', truncation=True, max_length=512)
        
        with torch.no_grad():
            outputs = self.model(**encodings)
            logits = outputs.logits
            
        # Convert to probability
        probs = torch.softmax(logits, dim=1)
        ai_confidence = probs[0][1].item()
        
        return ai_confidence

    def _combine_scores(self, scores_and_weights: List[Tuple[float, float]]) -> float:
        """
        Combine multiple scores with their weights into a final confidence score.
        
        Args:
            scores_and_weights: List of (score, weight) tuples
            
        Returns:
            float: Final weighted confidence score between 0 and 1
        """
        final_score = sum(score * weight for score, weight in scores_and_weights)
        return min(1.0, max(0.0, final_score))

    def get_feature_importance(self, scores: Dict[str, float]) -> Dict[str, str]:
        """
        Analyze which features contributed most to the AI detection.
        
        Args:
            scores: Dictionary of feature scores
            
        Returns:
            Dict[str, str]: Explanations for each feature's contribution
        """
        explanations = {}
        
        for feature, score in scores.items():
            if feature == 'final_confidence':
                continue
                
            if score > 0.8:
                importance = "Strong"
            elif score > 0.6:
                importance = "Moderate"
            else:
                importance = "Weak"
                
            explanations[feature] = f"{importance} indicator ({score:.2f})"
            
        return explanations
