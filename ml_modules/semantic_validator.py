"""
Semantic Validator Module
Validates document structure based on document type
Enhanced with dynamic document-aware rules
"""

import re
from typing import Dict, List, Any, Tuple

class SemanticValidator:
    def __init__(self):
        """
        Initialize semantic validator with document-specific rules
        """
        self.validation_rules = {
            "Banking": {
                "required_entities": {
                    "persons": {"min": 1, "description": "Account holder name"},
                    "organizations": {"min": 1, "description": "Bank name"},
                    "numbers": {"min": 2, "description": "Account and other numbers"}
                },
                "required_patterns": [
                    {"pattern": r'\b\d{8,18}\b', "description": "Account number (8-18 digits)"},
                    {"pattern": r'\b[A-Z]{4}0[A-Z0-9]{6}\b', "description": "IFSC code"},
                    {"pattern": r'(?:branch|office|bank)', "description": "Banking keywords"}
                ],
                "required_keywords": [
                    "account", "ifsc", "branch"
                ]
            },
            
            "Legal": {
                "required_entities": {
                    "persons": {"min": 2, "description": "Buyer and seller"},
                    "dates": {"min": 1, "description": "Registration/issue date"},
                    "money": {"min": 1, "description": "Stamp duty or consideration amount"},
                    "organizations": {"min": 1, "description": "Authority or organization"}
                },
                "required_patterns": [
                    {"pattern": r'(?:stamp\s+duty|stamp\s+value)', "description": "Stamp duty mention"},
                    {"pattern": r'(?:serial\s+number|sr\s+no)', "description": "Serial number"},
                    {"pattern": r'(?:registration\s+number|reg\s+no)', "description": "Registration number"},
                    {"pattern": r'\b(?:maharashtra|karnataka|tamil\s+nadu|delhi|gujarat)\b', "description": "State name"}
                ],
                "required_keywords": [
                    "stamp", "denomination", "serial", "agreement"
                ]
            },
            
            "Invoice": {
                "required_entities": {
                    "organizations": {"min": 1, "description": "Seller company"},
                    "money": {"min": 1, "description": "Total amount"},
                    "dates": {"min": 1, "description": "Invoice date"}
                },
                "required_patterns": [
                    {"pattern": r'(?:invoice\s+number|inv\s+no|bill\s+no)', "description": "Invoice number"},
                    {"pattern": r'(?:total|grand\s+total)', "description": "Total amount"},
                    {"pattern": r'(?:gst|tax)', "description": "Tax/GST mention"}
                ],
                "required_keywords": [
                    "invoice", "total", "amount", "tax"
                ]
            },
            
            "Certificate": {
                "required_entities": {
                    "persons": {"min": 1, "description": "Certificate recipient"},
                    "organizations": {"min": 1, "description": "Issuing authority"},
                    "dates": {"min": 1, "description": "Issue date"}
                },
                "required_patterns": [
                    {"pattern": r'(?:certificate|certification|award)', "description": "Certificate keywords"},
                    {"pattern": r'(?:issued\s+by|authority)', "description": "Issuing authority"},
                    {"pattern": r'(?:issued\s+to|awarded\s+to|recipient)', "description": "Recipient mention"}
                ],
                "required_keywords": [
                    "certificate", "issued", "date", "name"
                ]
            },
            
            "Resume": {
                "required_entities": {
                    "persons": {"min": 1, "description": "Person name"},
                    "organizations": {"min": 1, "description": "Companies or institutions"}
                },
                "required_patterns": [
                    {"pattern": r'(?:email|phone|contact)', "description": "Contact information"},
                    {"pattern": r'(?:education|experience|skills)', "description": "Resume sections"}
                ],
                "required_keywords": [
                    "experience", "education", "skills", "summary"
                ]
            },
            
            "Unknown": {
                "required_entities": {
                    "persons": {"min": 0, "description": "Any person names"},
                    "organizations": {"min": 0, "description": "Any organizations"},
                    "dates": {"min": 0, "description": "Any dates"},
                    "numbers": {"min": 0, "description": "Any numbers"}
                },
                "required_patterns": [],
                "required_keywords": []
            }
        }
    
    def validate_document(self, document_type: str, extracted_entities: Dict[str, List[str]], full_text: str) -> Dict[str, Any]:
        """
        Validate document based on its type with dynamic rules
        Returns validation results with scores and flags
        """
        if document_type not in self.validation_rules:
            document_type = "Unknown"
        
        rules = self.validation_rules[document_type]
        
        validation_result = {
            "document_type": document_type,
            "entity_validation": {},
            "pattern_validation": {},
            "keyword_validation": {},
            "completeness_score": 0.0,
            "validation_flags": {},
            "overall_score": 0.0,
            "issues": [],
            "strengths": []
        }
        
        # Validate entities
        entity_score = self._validate_entities(rules["required_entities"], extracted_entities, validation_result)
        
        # Validate patterns
        pattern_score = self._validate_patterns(rules["required_patterns"], full_text, validation_result)
        
        # Validate keywords (new dynamic validation)
        keyword_score = self._validate_keywords(rules["required_keywords"], full_text, validation_result)
        
        # Calculate overall completeness score
        validation_result["completeness_score"] = (entity_score + pattern_score + keyword_score) / 3
        
        # Generate validation flags
        validation_result["validation_flags"] = {
            "has_required_entities": entity_score >= 0.7,
            "has_required_patterns": pattern_score >= 0.7,
            "has_required_keywords": keyword_score >= 0.7,
            "is_structured": validation_result["completeness_score"] >= 0.6
        }
        
        # Calculate overall score
        validation_result["overall_score"] = validation_result["completeness_score"]
        
        # Add summary
        self._generate_summary(validation_result)
        
        return validation_result
    
    def _validate_entities(self, required_entities: Dict, extracted_entities: Dict[str, List[str]], result: Dict) -> float:
        """
        Validate required entities are present
        """
        entity_scores = []
        
        for entity_type, requirement in required_entities.items():
            extracted_count = len(extracted_entities.get(entity_type, []))
            required_min = requirement["min"]
            description = requirement["description"]
            
            # Calculate score for this entity type
            if required_min == 0:
                score = 1.0  # Optional entity
            else:
                score = min(extracted_count / required_min, 1.0)
            
            entity_scores.append(score)
            
            # Store validation result
            result["entity_validation"][entity_type] = {
                "required": required_min,
                "found": extracted_count,
                "score": score,
                "description": description,
                "status": "Pass" if score >= 1.0 else "Partial" if score > 0 else "Fail"
            }
            
            # Add to issues or strengths
            if score < 0.5:
                result["issues"].append(f"Missing {description} (found {extracted_count}, need {required_min})")
            elif score >= 1.0:
                result["strengths"].append(f"Good {description} coverage (found {extracted_count})")
        
        return sum(entity_scores) / len(entity_scores) if entity_scores else 0.0
    
    def _validate_patterns(self, required_patterns: List[Dict], full_text: str, result: Dict) -> float:
        """
        Validate required patterns are present in text
        """
        pattern_scores = []
        
        for i, pattern_info in enumerate(required_patterns):
            pattern = pattern_info["pattern"]
            description = pattern_info["description"]
            
            # Check if pattern exists
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            score = 1.0 if matches else 0.0
            
            pattern_scores.append(score)
            
            # Store validation result
            result["pattern_validation"][f"pattern_{i}"] = {
                "pattern": pattern,
                "description": description,
                "found": len(matches),
                "score": score,
                "status": "Pass" if score > 0 else "Fail"
            }
            
            # Add to issues or strengths
            if score == 0:
                result["issues"].append(f"Missing pattern: {description}")
            else:
                result["strengths"].append(f"Found required pattern: {description}")
        
        return sum(pattern_scores) / len(pattern_scores) if pattern_scores else 1.0
    
    def _validate_keywords(self, required_keywords: List[str], full_text: str, result: Dict) -> float:
        """
        Validate required keywords are present in text
        """
        if not required_keywords:
            return 1.0
        
        text_lower = full_text.lower()
        keyword_scores = []
        
        for keyword in required_keywords:
            keyword_lower = keyword.lower()
            found_count = text_lower.count(keyword_lower)
            score = 1.0 if found_count > 0 else 0.0
            
            keyword_scores.append(score)
            
            # Store validation result
            result["keyword_validation"][keyword] = {
                "found": found_count,
                "score": score,
                "status": "Pass" if score > 0 else "Fail"
            }
            
            # Add to issues or strengths
            if score == 0:
                result["issues"].append(f"Missing keyword: {keyword}")
            else:
                result["strengths"].append(f"Found required keyword: {keyword}")
        
        return sum(keyword_scores) / len(keyword_scores) if keyword_scores else 0.0
    
    def _generate_summary(self, validation_result: Dict):
        """
        Generate summary of validation results
        """
        score = validation_result["completeness_score"]
        
        if score >= 0.8:
            validation_result["summary"] = "Excellent document structure and completeness"
        elif score >= 0.6:
            validation_result["summary"] = "Good document structure with minor gaps"
        elif score >= 0.4:
            validation_result["summary"] = "Moderate document structure with significant gaps"
        else:
            validation_result["summary"] = "Poor document structure, major issues detected"
        
        validation_result["grade"] = self._get_grade(score)
    
    def _get_grade(self, score: float) -> str:
        """
        Get letter grade based on score
        """
        if score >= 0.9:
            return "A+"
        elif score >= 0.8:
            return "A"
        elif score >= 0.7:
            return "B+"
        elif score >= 0.6:
            return "B"
        elif score >= 0.5:
            return "C"
        elif score >= 0.4:
            return "D"
        else:
            return "F"

# Global instance for reuse
_validator_instance = None

def get_semantic_validator():
    """
    Get or create semantic validator instance
    """
    global _validator_instance
    if _validator_instance is None:
        _validator_instance = SemanticValidator()
    return _validator_instance

def validate_document(document_type: str, extracted_entities: Dict[str, List[str]], full_text: str) -> Dict[str, Any]:
    """
    Convenience function to validate document
    """
    validator = get_semantic_validator()
    return validator.validate_document(document_type, extracted_entities, full_text)

if __name__ == "__main__":
    # Test the semantic validator
    test_entities = {
        "persons": ["John Smith"],
        "organizations": ["ABC Bank"],
        "numbers": ["123456789012"],
        "dates": ["15/01/2024"]
    }
    
    test_text = "Account Number: 123456789012 IFSC: SBIN0001234 Branch: Main Branch Account Holder: John Smith"
    
    validator = SemanticValidator()
    result = validator.validate_document("Banking", test_entities, test_text)
    
    print("Validation Result:")
    print(f"Overall Score: {result['overall_score']:.2f}")
    print(f"Grade: {result['grade']}")
    print(f"Summary: {result['summary']}")
    print(f"Issues: {result['issues']}")
    print(f"Strengths: {result['strengths']}")
