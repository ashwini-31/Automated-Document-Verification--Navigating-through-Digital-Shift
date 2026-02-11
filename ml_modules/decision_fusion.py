"""
Decision Fusion Module
Combines multiple verification scores into final decision
"""

from typing import Dict, Any

class DecisionFusion:
    def __init__(self):
        """
        Initialize decision fusion with configurable weights
        """
        # Rebalanced weights for better calibration
        self.weights = {
            "classification": 0.5,
            "tamper": 0.3,
            "ocr": 0.2
        }
        
        # Centralized decision threshold
        self.threshold = 0.6
    
    def update_weights(self, new_weights: Dict[str, float]):
        """
        Update fusion weights
        """
        if sum(new_weights.values()) != 1.0:
            raise ValueError("Weights must sum to 1.0")
        
        self.weights.update(new_weights)
        print(f"Updated fusion weights: {self.weights}")
    
    def compute_final_score(self, classification_conf: float, signature_score: float, tamper_score: float, ocr_conf: float = 0.0) -> float:
        """
        Compute final verification score using updated weights
        
        Args:
            classification_conf: 0-1 (classification confidence)
            signature_score: 0-1 (signature verification score)
            tamper_score: 0-1 (0 = clean, 1 = fully tampered)
            ocr_conf: 0-1 (OCR confidence)
        
        Returns:
            final_score: 0-1 (combined verification score)
        """
        # Updated formula with new weights
        final_score = (
            self.weights["classification"] * classification_conf +
            self.weights["tamper"] * (1 - tamper_score) +
            self.weights["ocr"] * ocr_conf
        )
        
        return final_score
    
    def make_decision(self, final_score: float, threshold: float = None) -> str:
        """
        Make final verification decision using centralized threshold
        
        Args:
            final_score: 0-1 (combined verification score)
            threshold: decision threshold (uses centralized threshold if None)
        
        Returns:
            decision: "VERIFIED" or "FLAGGED"
        """
        if threshold is None:
            threshold = self.threshold
            
        if final_score >= threshold:
            return "VERIFIED"
        else:
            return "FLAGGED"
    
    def get_score_breakdown(self, classification_conf: float, signature_score: float, tamper_score: float) -> Dict[str, float]:
        """
        Get breakdown of individual score contributions
        """
        return {
            "classification_contribution": self.weights["classification"] * classification_conf,
            "signature_contribution": self.weights["signature"] * signature_score,
            "tamper_contribution": self.weights["tamper"] * (1 - tamper_score),
            "final_score": self.compute_final_score(classification_conf, signature_score, tamper_score)
        }

# Global instance for reuse
_fusion_instance = None

def get_decision_fusion():
    """
    Get or create decision fusion instance
    """
    global _fusion_instance
    if _fusion_instance is None:
        _fusion_instance = DecisionFusion()
    return _fusion_instance

if __name__ == "__main__":
    # Test the decision fusion
    fusion = DecisionFusion()
    
    # Test cases
    test_cases = [
        (0.9, 0.8, 0.1),  # High confidence, good signature, low tamper
        (0.6, 0.5, 0.7),  # Medium confidence, medium signature, high tamper
        (0.3, 0.2, 0.9),  # Low confidence, poor signature, high tamper
    ]
    
    for i, (class_conf, sig_score, tamper_score) in enumerate(test_cases):
        final_score = fusion.compute_final_score(class_conf, sig_score, tamper_score)
        decision = fusion.make_decision(final_score)
        breakdown = fusion.get_score_breakdown(class_conf, sig_score, tamper_score)
        
        print(f"\nTest Case {i+1}:")
        print(f"  Classification: {class_conf:.2f}, Signature: {sig_score:.2f}, Tamper: {tamper_score:.2f}")
        print(f"  Final Score: {final_score:.3f}")
        print(f"  Decision: {decision}")
        print(f"  Breakdown: {breakdown}")
