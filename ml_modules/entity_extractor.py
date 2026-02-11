"""
Entity Extractor Module
Uses spaCy for Named Entity Recognition and dynamic entity extraction
Enhanced with domain filtering and controlled PII masking
"""

import spacy
import re
from typing import Dict, List, Any

# Whitelist of technical terms that should not be masked
WHITELIST_TERMS = [
    "Python", "TensorFlow", "PyTorch", "SQL", "React", "CNN", 
    "NLP", "ML", "BERT", "Scikit-learn", "Keras", "Pandas",
    "NumPy", "OpenCV", "Tesseract", "Streamlit", "Flask", "Django",
    "JavaScript", "TypeScript", "Node.js", "MongoDB", "PostgreSQL"
]

# Technical terms set for person validation (CRITICAL BUG FIX)
TECH_TERMS = {
    "Python", "TensorFlow", "PyTorch", "NumPy", "Pandas",
    "Matplotlib", "Scikit-learn", "LangChain", "NLTK",
    "SpaCy", "Transformers", "Streamlit", "Ollama",
    "Keras", "HTML", "CSS", "JavaScript", "SQL",
    "LDA", "LSA", "ARIMA", "LSTM", "Jupyter Notebook",
    "Google Gemini", "CNN"
}

class EntityExtractor:
    def __init__(self):
        """
        Initialize spaCy model for NER
        """
        try:
            self.nlp = spacy.load("en_core_web_sm")
            print("spaCy model loaded successfully!")
        except OSError:
            print("Error: spaCy model not found. Please run: python -m spacy download en_core_web_sm")
            raise
    
    def _is_valid_person(self, entity_text: str, full_text: str) -> bool:
        """
        Validate if entity is actually a person name, not a technical term
        """
        entity_lower = entity_text.lower()
        entity_text_clean = entity_text.strip()
        
        # Reject if in tech terms (exact match)
        if entity_text_clean in TECH_TERMS:
            return False  # do not treat tech term as person
        
        # Reject if length < 3
        if len(entity_text_clean) < 3:
            return False
        
        # Reject if all uppercase (likely acronym)
        if entity_text_clean.isupper():
            return False
        
        # Reject if contains digits
        if any(char.isdigit() for char in entity_text_clean):
            return False
        
        # Reject if appears > 3 times in document (likely common term)
        occurrences = full_text.lower().count(entity_lower)
        if occurrences > 3:
            return False
        
        # Accept if looks like a name (starts with capital, contains only letters)
        if entity_text_clean.istitle() and entity_text_clean.replace(' ', '').isalpha():
            return True
        
        return False
    
    def extract_entities(self, text: str) -> Dict[str, List[str]]:
        """
        Extract entities using spaCy NER with post-processing corrections
        Returns structured dictionary of entities
        """
        if not text or not text.strip():
            return {
                "persons": [],
                "organizations": [],
                "dates": [],
                "locations": [],
                "money": [],
                "numbers": [],
                "emails": [],
                "phones": [],
                "tech": []
            }
        
        # Process text with spaCy
        doc = self.nlp(text)
        
        # Initialize entity collections
        entities = {
            "persons": [],
            "organizations": [],
            "dates": [],
            "locations": [],
            "money": [],
            "numbers": [],
            "emails": [],
            "phones": [],
            "tech": []
        }
        
        # Extract spaCy entities with post-processing
        for ent in doc.ents:
            entity_text = ent.text.strip()
            entity_label = ent.label_
            
            # IMPORTANT: Check for TECH_TERMS first to prevent misclassification
            if entity_text in TECH_TERMS:
                # Force TECH classification for technical terms
                if entity_text not in entities["tech"]:
                    entities["tech"].append(entity_text)
                continue  # Skip further processing
            
            # Post-processing corrections for non-tech terms
            if entity_label == "PERSON":
                if self._is_valid_person(entity_text, text):
                    if entity_text not in entities["persons"]:
                        entities["persons"].append(entity_text)
                        
            elif entity_label == "ORG":
                # Add organization (tech terms already handled above)
                if entity_text not in entities["organizations"]:
                    entities["organizations"].append(entity_text)
                        
            elif entity_label == "DATE":
                if entity_text not in entities["dates"]:
                    entities["dates"].append(entity_text)
                    
            elif entity_label in ["GPE", "LOC"]:  # Geopolitical Entity, Location
                if entity_text not in entities["locations"]:
                    entities["locations"].append(entity_text)
                    
            elif entity_label == "MONEY":
                if entity_text not in entities["money"]:
                    entities["money"].append(entity_text)
                    
            elif entity_label == "CARDINAL":
                # Extract numeric values
                numbers = re.findall(r'\d+', entity_text)
                for num in numbers:
                    if num not in entities["numbers"]:
                        entities["numbers"].append(num)
        
        # Additional pattern-based extractions
        
        # Extract emails
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        entities["emails"] = list(set(emails))  # Remove duplicates
        
        # Extract phone numbers (more precise patterns)
        phone_patterns = [
            r'\+?\d{1,3}[-\s]?\d{10}',  # International format
            r'\b\d{10}\b',              # Simple 10-digit
            r'\+?\d{1,3}[-\s]?\d{3}[-\s]?\d{3}[-\s]?\d{4}',  # US format
            r'\+?\d{1,3}[-\s]?\(?\d{3}\)?[-\s]?\d{3}[-\s]?\d{4}'  # US with parentheses
        ]
        phones = []
        for pattern in phone_patterns:
            phones.extend(re.findall(pattern, text))
        entities["phones"] = list(set(phones))  # Remove duplicates
        
        # Extract specific patterns for controlled masking
        self._extract_controlled_patterns(text, entities)
        
        # Clean and deduplicate
        for key in entities:
            entities[key] = [item.strip() for item in entities[key] if item.strip()]
            entities[key] = list(set(entities[key]))  # Remove duplicates
        
        return entities
    
    def _extract_controlled_patterns(self, text: str, entities: Dict[str, List[str]]):
        """
        Extract specific patterns for controlled PII masking
        """
        # Bank account numbers (8-18 digits, not phone numbers)
        account_pattern = r'\b\d{8,18}\b'
        account_numbers = re.findall(account_pattern, text)
        for acc_num in account_numbers:
            # Filter out phone numbers and dates
            if not re.match(r'\d{10}$', acc_num) and not re.match(r'\d{2}[/-]\d{2}[/-]\d{4}', acc_num):
                if acc_num not in entities["numbers"]:
                    entities["numbers"].append(acc_num)
        
        # IFSC codes
        ifsc_pattern = r'\b[A-Z]{4}0[A-Z0-9]{6}\b'
        ifsc_codes = re.findall(ifsc_pattern, text)
        for ifsc in ifsc_codes:
            if ifsc not in entities["numbers"]:
                entities["numbers"].append(ifsc)
        
        # PAN numbers
        pan_pattern = r'\b[A-Z]{5}\d{4}[A-Z]\b'
        pan_numbers = re.findall(pan_pattern, text)
        for pan in pan_numbers:
            if pan not in entities["numbers"]:
                entities["numbers"].append(pan)
        
        # Aadhaar numbers (12 digits)
        aadhaar_pattern = r'\b\d{12}\b'
        aadhaar_numbers = re.findall(aadhaar_pattern, text)
        for aadhaar in aadhaar_numbers:
            if aadhaar not in entities["DATE"]:
                entities["DATE"].append(aadhaar)
    
    def mask_pii_entities(self, text: str, entities: Dict[str, List[str]]) -> str:
        """
        Mask PII entities with whitelist protection for technical terms
        """
        masked_text = text
        
        # Mask PERSON entities (check whitelist first)
        for person in entities.get("PERSON", []):
            if should_mask_entity(person):
                # Mask only the name part, keep surrounding text
                pattern = r'\b' + re.escape(person) + r'\b'
                masked_text = re.sub(pattern, '[PERSON]', masked_text, flags=re.IGNORECASE)
        
        # Mask EMAIL entities
        for email in entities.get("EMAIL", []):
            if should_mask_entity(email):
                pattern = r'\b' + re.escape(email) + r'\b'
                masked_text = re.sub(pattern, '[EMAIL]', masked_text)
        
        # Mask PHONE entities
        for phone in entities.get("PHONE", []):
            if should_mask_entity(phone):
                pattern = r'\b' + re.escape(phone) + r'\b'
                masked_text = re.sub(pattern, '[PHONE]', masked_text)
        
        # Mask GPE entities (locations)
        for gpe in entities.get("GPE", []):
            if should_mask_entity(gpe):
                pattern = r'\b' + re.escape(gpe) + r'\b'
                masked_text = re.sub(pattern, '[LOCATION]', masked_text)
        
        # Mask ORG entities (check whitelist first)
        for org in entities.get("ORG", []):
            if should_mask_entity(org):
                pattern = r'\b' + re.escape(org) + r'\b'
                masked_text = re.sub(pattern, '[ORGANIZATION]', masked_text)
        
        return masked_text
    
    def get_entity_summary(self, entities: Dict[str, List[str]]) -> Dict[str, Any]:
        """
        Get summary statistics of extracted entities
        """
        summary = {}
        for entity_type, entity_list in entities.items():
            summary[entity_type] = {
                "count": len(entity_list),
                "items": entity_list
            }
        
        # Calculate total entities
        total_entities = sum(len(entities[key]) for key in entities)
        summary["total_entities"] = total_entities
        
        # Entity density (entities per 100 words)
        word_count = len(" ".join(entities.values()).split())
        if word_count > 0:
            summary["entity_density"] = (total_entities / word_count) * 100
        else:
            summary["entity_density"] = 0
        
        return summary

def ml_mask_pii(text: str) -> str:
    """
    ML-based PII masking with strict order and whitelist protection
    Order: 1. Emails, 2. Phones, 3. PERSON entities, 4. TECH terms whitelist
    """
    extractor = get_entity_extractor()
    
    # Step 1: Mask EMAIL using regex (always first)
    masked_text = text
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    masked_text = re.sub(email_pattern, '[EMAIL]', masked_text)
    
    # Step 2: Mask PHONE using regex (always second)
    phone_pattern = r'\b\d{10}\b'
    masked_text = re.sub(phone_pattern, '[PHONE]', masked_text)
    
    # Step 3: Resume name regex detection (before spaCy)
    resume_name_pattern = r'^[A-Z]{2,}\s+[A-Z]{2,}'
    if re.match(resume_name_pattern, masked_text.split('\n')[0].strip()):
        # Mask the first line if it looks like a resume header name
        lines = masked_text.split('\n')
        lines[0] = re.sub(resume_name_pattern, '[PERSON]', lines[0])
        masked_text = '\n'.join(lines)
    
    # Step 4: Detect entities using spaCy NER
    doc = extractor.nlp(masked_text)
    
    # Step 5: Process PERSON entities (always mask)
    for ent in doc.ents:
        if ent.label_ == "PERSON":
            # Check if it's a tech term first
            if ent.text not in TECH_TERMS:
                pattern = r'\b' + re.escape(ent.text) + r'\b'
                masked_text = re.sub(pattern, '[PERSON]', masked_text)
            else:
                # Force TECH label for technical terms
                pattern = r'\b' + re.escape(ent.text) + r'\b'
                masked_text = re.sub(pattern, '[TECH]', masked_text)
    
    return masked_text

# Global instance for reuse
_nlp_instance = None

def get_entity_extractor():
    """
    Get or create entity extractor instance
    """
    global _nlp_instance
    if _nlp_instance is None:
        _nlp_instance = EntityExtractor()
    return _nlp_instance

def extract_entities(text: str) -> Dict[str, List[str]]:
    """
    Convenience function to extract entities
    """
    extractor = get_entity_extractor()
    return extractor.extract_entities(text)

def mask_pii(text: str) -> str:
    """
    Convenience function to mask PII in text
    """
    extractor = get_entity_extractor()
    entities = extractor.extract_entities(text)
    return extractor.mask_pii_entities(text, entities)

if __name__ == "__main__":
    # Test the entity extractor
    test_text = """
    John Smith works at TensorFlow in New York.
    His email is john.smith@tensorflow.com and phone is +1-555-123-4567.
    Account number: 123456789012
    Amount: $5000.00
    Date: 15/01/2024
    Python is a programming language.
    """
    
    extractor = EntityExtractor()
    entities = extractor.extract_entities(test_text)
    print("Extracted Entities:")
    for entity_type, entity_list in entities.items():
        print(f"{entity_type}: {entity_list}")
    
    print("\nMasked Text:")
    masked = extractor.mask_pii_entities(test_text, entities)
    print(masked)
