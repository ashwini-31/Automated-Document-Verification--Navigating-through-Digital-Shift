"""
Schema Generator Module
Generates expected fields based on document type
Enhanced with Tobacco3482 dataset classes
"""

from typing import Dict, List, Any

class SchemaGenerator:
    def __init__(self):
        """
        Initialize schema generator with predefined schemas
        """
        self.schemas = {
            # Original document types
            "Banking": {
                "required_fields": [
                    "account_number",
                    "ifsc",
                    "branch",
                    "holder_name"
                ],
                "optional_fields": [
                    "balance",
                    "customer_id",
                    "micr_code",
                    "bank_address",
                    "phone",
                    "email"
                ],
                "patterns": {
                    "account_number": r'\b\d{8,18}\b',
                    "ifsc": r'\b[A-Z]{4}0[A-Z0-9]{6}\b',
                    "branch": r'(?:branch|office)\s+name[:\s]*([A-Za-z\s]+)',
                    "holder_name": r'(?:account\s+holder|customer|name)[:\s]*([A-Za-z\s]+)',
                    "balance": r'(?:balance|amount)[:\s]*[\$₹]?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                    "customer_id": r'(?:customer\s+id|client\s+id)[:\s]*([A-Z0-9]+)',
                    "micr_code": r'micr[:\s]*(\d{9})',
                    "bank_address": r'(?:bank\s+address|address)[:\s]*([A-Za-z0-9\s,.-]+)'
                }
            },
            
            "Legal": {
                "required_fields": [
                    "stamp_value",
                    "serial_number",
                    "registration_number",
                    "purchaser_name",
                    "issue_date"
                ],
                "optional_fields": [
                    "seller_name",
                    "witness_names",
                    "notary_name",
                    "property_details",
                    "consideration_amount"
                ],
                "patterns": {
                    "stamp_value": r'(?:stamp\s+(?:duty|value|denomination))[:\s]*[\$₹]?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                    "serial_number": r'(?:serial\s+number|sr\s+no)[:\s]*([A-Z0-9-]+)',
                    "registration_number": r'(?:registration\s+number|reg\s+no)[:\s]*([A-Z0-9-]+)',
                    "purchaser_name": r'(?:purchaser|buyer|first\s+party)[:\s]*([A-Za-z\s]+)',
                    "issue_date": r'(?:date|executed|dated)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                    "seller_name": r'(?:seller|second\s+party)[:\s]*([A-Za-z\s]+)',
                    "witness_names": r'(?:witness|witnesses)[:\s]*([A-Za-z\s,]+)',
                    "notary_name": r'(?:notary|notary\s+public)[:\s]*([A-Za-z\s]+)',
                    "property_details": r'(?:property|premises|address)[:\s]*([A-Za-z0-9\s,.-]+)',
                    "consideration_amount": r'(?:consideration|amount|price)[:\s]*[\$₹]?\s*(\d+(?:,\d+)*(?:\.\d{2})?)'
                }
            },
            
            "Invoice": {
                "required_fields": [
                    "invoice_number",
                    "total_amount",
                    "seller_name",
                    "invoice_date"
                ],
                "optional_fields": [
                    "buyer_name",
                    "tax_amount",
                    "items_list",
                    "payment_terms",
                    "bank_details"
                ],
                "patterns": {
                    "invoice_number": r'(?:invoice\s+number|inv\s+no|bill\s+no)[:\s]*([A-Z0-9-]+)',
                    "total_amount": r'(?:total|grand\s+total|amount)[:\s]*[\$₹]?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                    "seller_name": r'(?:seller|vendor|company|from)[:\s]*([A-Za-z\s&]+)',
                    "invoice_date": r'(?:invoice\s+date|date|dated)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                    "buyer_name": r'(?:buyer|customer|to|bill\s+to)[:\s]*([A-Za-z\s]+)',
                    "tax_amount": r'(?:tax|gst|vat|service\s+tax)[:\s]*[\$₹]?\s*(\d+(?:,\d+)*(?:\.\d{2})?)',
                    "items_list": r'(?:items|description|products)[:\s]*([A-Za-z0-9\s,.-]+)',
                    "payment_terms": r'(?:payment\s+terms|terms)[:\s]*([A-Za-z0-9\s]+)',
                    "bank_details": r'(?:bank|account|ifsc)[:\s]*([A-Za-z0-9\s,-]+)'
                }
            },
            
            "Certificate": {
                "required_fields": [
                    "recipient_name",
                    "issuing_authority",
                    "certificate_date",
                    "certificate_type"
                ],
                "optional_fields": [
                    "certificate_id",
                    "grade",
                    "duration",
                    "institution_address"
                ],
                "patterns": {
                    "recipient_name": r'(?:awarded\s+to|issued\s+to|name|recipient)[:\s]*([A-Za-z\s]+)',
                    "issuing_authority": r'(?:issued\s+by|authority|institution)[:\s]*([A-Za-z\s&]+)',
                    "certificate_date": r'(?:date|issued|dated)[:\s]*(\d{1,2}[/-]\d{1,2}[/-]\d{2,4})',
                    "certificate_type": r'(?:certificate|certification|degree|diploma)[:\s]*([A-Za-z\s]+)',
                    "certificate_id": r'(?:certificate\s+id|id|number)[:\s]*([A-Z0-9-]+)',
                    "grade": r'(?:grade|score|percentage)[:\s]*([A-Z0-9.%]+)',
                    "duration": r'(?:duration|period|valid)[:\s]*([A-Za-z0-9\s]+)',
                    "institution_address": r'(?:address|location)[:\s]*([A-Za-z0-9\s,.-]+)'
                }
            },
            
            "Resume": {
                "required_fields": [
                    "candidate_name",
                    "contact_info",
                    "education",
                    "experience"
                ],
                "optional_fields": [
                    "skills",
                    "projects",
                    "certifications",
                    "objective"
                ],
                "patterns": {
                    "candidate_name": r'(?:name|candidate|applicant)[:\s]*([A-Za-z\s]+)',
                    "contact_info": r'(?:email|phone|contact)[:\s]*([A-Za-z0-9@.+-]+)',
                    "education": r'(?:education|qualification|degree)[:\s]*([A-Za-z0-9\s,.-]+)',
                    "experience": r'(?:experience|work|employment)[:\s]*([A-Za-z0-9\s,.-]+)',
                    "skills": r'(?:skills|technical\s+skills|expertise)[:\s]*([A-Za-z0-9\s,.-]+)',
                    "projects": r'(?:projects|portfolio)[:\s]*([A-Za-z0-9\s,.-]+)',
                    "certifications": r'(?:certifications|certificates)[:\s]*([A-Za-z0-9\s,.-]+)',
                    "objective": r'(?:objective|summary|profile)[:\s]*([A-Za-z0-9\s,.-]+)'
                }
            },
            
            # Tobacco3482 dataset classes
            "ADVE": {
                "required_fields": ["advertiser_name", "product", "contact_info"],
                "optional_fields": ["offer_details", "validity_period", "terms"],
                "description": "Advertisement document"
            },
            
            "Email": {
                "required_fields": ["sender", "recipient", "subject"],
                "optional_fields": ["date", "attachments", "cc_bcc"],
                "description": "Email document"
            },
            
            "Form": {
                "required_fields": ["form_title", "fields"],
                "optional_fields": ["instructions", "submission_info"],
                "description": "Form document"
            },
            
            "Letter": {
                "required_fields": ["sender", "recipient", "content"],
                "optional_fields": ["date", "signature", "subject"],
                "description": "Letter document"
            },
            
            "Memo": {
                "required_fields": ["to", "from", "subject"],
                "optional_fields": ["date", "cc", "attachments"],
                "description": "Memorandum document"
            },
            
            "News": {
                "required_fields": ["headline", "content"],
                "optional_fields": ["date", "author", "source"],
                "description": "News article document"
            },
            
            "Note": {
                "required_fields": ["content"],
                "optional_fields": ["date", "author", "subject"],
                "description": "Note document"
            },
            
            "Report": {
                "required_fields": ["title", "content"],
                "optional_fields": ["author", "date", "summary"],
                "description": "Report document"
            },
            
            "Scientific": {
                "required_fields": ["title", "abstract", "content"],
                "optional_fields": ["authors", "references", "methodology"],
                "description": "Scientific paper document"
            },
            
            "Unknown": {
                "required_fields": [],
                "optional_fields": [],
                "description": "Unknown document type"
            }
        }
    
    def generate_expected_schema(self, document_type: str) -> Dict[str, Any]:
        """
        Generate expected schema based on document type
        """
        if document_type not in self.schemas:
            document_type = "Unknown"
        
        return self.schemas[document_type]
    
    def get_field_patterns(self, document_type: str) -> Dict[str, str]:
        """
        Get regex patterns for a document type
        """
        schema = self.generate_expected_schema(document_type)
        return schema.get("patterns", {})
    
    def validate_schema_completeness(self, document_type: str, extracted_fields: List[str]) -> Dict[str, Any]:
        """
        Validate if required fields are present
        """
        schema = self.generate_expected_schema(document_type)
        required_fields = schema.get("required_fields", [])
        optional_fields = schema.get("optional_fields", [])
        
        # Check required fields
        missing_required = [field for field in required_fields if field not in extracted_fields]
        found_required = [field for field in required_fields if field in extracted_fields]
        
        # Check optional fields
        found_optional = [field for field in optional_fields if field in extracted_fields]
        
        completeness_score = len(found_required) / len(required_fields) if required_fields else 1.0
        
        return {
            "document_type": document_type,
            "required_fields": required_fields,
            "optional_fields": optional_fields,
            "found_required": found_required,
            "found_optional": found_optional,
            "missing_required": missing_required,
            "completeness_score": completeness_score,
            "is_complete": len(missing_required) == 0
        }

# Global instance for reuse
_schema_generator_instance = None

def get_schema_generator():
    """
    Get or create schema generator instance
    """
    global _schema_generator_instance
    if _schema_generator_instance is None:
        _schema_generator_instance = SchemaGenerator()
    return _schema_generator_instance

def generate_expected_schema(document_type: str) -> Dict[str, Any]:
    """
    Convenience function to generate expected schema
    """
    generator = get_schema_generator()
    return generator.generate_expected_schema(document_type)

if __name__ == "__main__":
    # Test the schema generator
    generator = SchemaGenerator()
    
    # Test with different document types
    for doc_type in ["Banking", "Legal", "Invoice", "Certificate", "Resume", "ADVE", "Email"]:
        schema = generator.generate_expected_schema(doc_type)
        print(f"\n{doc_type} Schema:")
        print(f"Required Fields: {schema['required_fields']}")
        print(f"Optional Fields: {schema['optional_fields']}")
        if 'description' in schema:
            print(f"Description: {schema['description']}")
