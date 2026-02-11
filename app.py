import streamlit as st
import cv2
import pytesseract
import numpy as np
from PIL import Image
import re
from skimage import io, color
import math
import tempfile
import os
import fitz  # PyMuPDF
import io as io_module

# Import ML modules
from ml_modules.image_classifier import predict_from_pil_image
from ml_modules.entity_extractor import extract_entities, mask_pii as ml_mask_pii
from ml_modules.schema_generator import generate_expected_schema
from ml_modules.semantic_validator import validate_document

# Configure Tesseract path for Windows
pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

def convert_pdf_to_images(pdf_file):
    """
    Convert PDF file to list of PIL Images using PyMuPDF (in-memory processing)
    
    Args:
        pdf_file: Uploaded file from Streamlit (BytesIO-like object)
        
    Returns:
        List of PIL Images
    """
    images = []
    
    try:
        pdf_bytes = pdf_file.read()
        doc = fitz.open(stream=pdf_bytes, filetype="pdf")

        for page_index in range(len(doc)):
            page = doc.load_page(page_index)
            pix = page.get_pixmap(dpi=150)

            img_data = pix.tobytes("png")
            image = Image.open(io_module.BytesIO(img_data)).convert("RGB")

            images.append(image)

        doc.close()
        return images
        
    except Exception as e:
        st.error(f"PDF conversion failed: {str(e)}")
        return []


def ensure_rgb_image(image):
    """
    Ensure image is 3-channel RGB for CNN classifier
    Handles grayscale, RGB, and RGBA images
    """
    # Convert PIL to ensure RGB mode
    if image.mode != "RGB":
        image = image.convert("RGB")
    
    # Convert to numpy array
    img_array = np.array(image)
    
    # Ensure 3 channels for CNN
    if len(img_array.shape) == 2:
        # Grayscale (H, W) → convert to 3-channel
        rgb_array = np.stack([img_array] * 3, axis=-1)
    elif img_array.shape[2] == 3:
        # Already RGB (H, W, 3)
        rgb_array = img_array
    elif img_array.shape[2] == 4:
        # RGBA (H, W, 4) → convert to RGB
        rgb_array = img_array[:, :, :3]
    else:
        # Unexpected format → safe fallback
        print(f"Unexpected image format: {img_array.shape}")
        rgb_array = img_array[:, :, :3] if len(img_array.shape) == 3 else img_array
    
    # Convert back to PIL RGB image
    return Image.fromarray(rgb_array.astype(np.uint8), mode='RGB')

def preprocess_image(image):
    """
    Apply image preprocessing: grayscale conversion, median blur, and sharpening
    Safely handles grayscale, RGB, and RGBA images
    """
    # Convert PIL to OpenCV format
    img_array = np.array(image)
    
    # Safely convert to grayscale based on image channels
    try:
        if len(img_array.shape) == 2:
            # Already grayscale (H, W)
            gray = img_array
        elif img_array.shape[2] == 3:
            # RGB image (H, W, 3)
            gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
        elif img_array.shape[2] == 4:
            # RGBA image (H, W, 4) - convert to RGB first
            rgb = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
            gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)
        else:
            # Unexpected format - try safe fallback
            print(f"Unexpected image shape: {img_array.shape}")
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        print(f"Grayscale conversion failed: {str(e)}")
        # Safe fallback
        try:
            gray = cv2.cvtColor(img_array, cv2.COLOR_BGR2GRAY)
        except:
            # Last resort - use first channel if available
            if len(img_array.shape) >= 2:
                gray = img_array[:, :, 0] if len(img_array.shape) == 3 else img_array
            else:
                gray = img_array
    
    # Apply median blur for noise reduction (median filter)
    median_blur = cv2.medianBlur(gray, 3)
    
    # Apply sharpening filter
    kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
    sharpened = cv2.filter2D(median_blur, -1, kernel)
    
    return gray, median_blur, sharpened

def extract_text(image):
    """
    Extract text using OCR with confidence scores and bounding boxes
    Safely handles grayscale, RGB, and RGBA images
    Returns structured dictionary with real confidence computation
    """
    # Convert PIL to numpy array for OpenCV
    img_array = np.array(image)
    
    # Ensure image is in proper format for Tesseract
    try:
        if len(img_array.shape) == 2:
            # Already grayscale - ensure 3 channels for Tesseract
            img_for_ocr = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
        elif img_array.shape[2] == 3:
            # RGB image - ready for Tesseract
            img_for_ocr = img_array
        elif img_array.shape[2] == 4:
            # RGBA image - convert to RGB
            img_for_ocr = cv2.cvtColor(img_array, cv2.COLOR_RGBA2RGB)
        else:
            # Unexpected format - use safe fallback
            print(f"Unexpected image shape for OCR: {img_array.shape}")
            img_for_ocr = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"OCR image preparation failed: {str(e)}")
        # Safe fallback - convert to RGB
        try:
            img_for_ocr = cv2.cvtColor(img_array, cv2.COLOR_BGR2RGB)
        except:
            img_for_ocr = img_array
    
    # Extract text with confidence scores
    data = pytesseract.image_to_data(img_for_ocr, output_type=pytesseract.Output.DICT)
    
    # Extract full text
    full_text = pytesseract.image_to_string(img_for_ocr)
    
    # Calculate real average confidence (excluding confidence = -1 for empty text)
    confidences = [conf for conf in data['conf'] if conf > 0]
    avg_confidence = np.mean(confidences) if confidences else 0
    
    # Normalize confidence to 0-1 range
    ocr_confidence = avg_confidence / 100.0
    
    # Get bounding boxes for words
    boxes = []
    for i in range(len(data['text'])):
        if int(data['conf'][i]) > 0:
            x, y, w, h = data['left'][i], data['top'][i], data['width'][i], data['height'][i]
            boxes.append({
                'text': data['text'][i],
                'bbox': (x, y, x+w, y+h),
                'confidence': data['conf'][i] / 100.0  # Normalize to 0-1
            })
    
    # Return structured format with real confidence
    return {
        "text": full_text,
        "language": "en",  # placeholder for now
        "confidence": ocr_confidence,  # Real computed confidence
        "detailed_boxes": boxes  # Additional detailed information
    }

def mask_all_caps_names(text):
    """
    Detect and mask ALL CAPS names like PHILIP R. GRANT, JAMES N RAVLIN
    """
    pattern = r'\b[A-Z]{2,}(?:\s+[A-Z]\.)?(?:\s+[A-Z]{2,})+\b'
    return re.sub(pattern, '[PERSON]', text)

def mask_pii(text):
    """
    Enhanced PII masking with regex + spaCy for robust dataset-independent coverage
    """
    print("Masking applied")
    
    # Step 1: Strong email masking (always applied)
    text = re.sub(r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b',
                  '[EMAIL]', text)
    
    # Step 2: Strong phone masking (10+ digits)
    text = re.sub(r'\b\d{10,}\b', '[PHONE]', text)
    
    # Step 3: ALL CAPS name detection (critical for Tobacco dataset)
    text = mask_all_caps_names(text)
    
    # Step 4: "Dear Mr./Ms." pattern masking
    text = re.sub(r'Dear\s+(Mr|Ms|Mrs|Dr)\.\s+[A-Z][a-z]+',
                  'Dear [PERSON]', text)
    
    # Step 5: ML-based masking with spaCy (enhanced, not only that)
    try:
        # Use enhanced ML-based PII masking
        masked_text = ml_mask_pii(text)
        print("After ML masking preview:", masked_text[:200])
    except Exception as e:
        print(f"ML PII masking failed, using enhanced regex: {str(e)}")
        masked_text = text
        
        # Fallback to spaCy NER
        try:
            import spacy
            nlp = spacy.load("en_core_web_sm")
            doc = nlp(masked_text)
            
            # Mask PERSON entities detected by spaCy
            for ent in doc.ents:
                if ent.label_ == "PERSON":
                    pattern = r'\b' + re.escape(ent.text) + r'\b'
                    masked_text = re.sub(pattern, '[PERSON]', masked_text)
                    
        except Exception as spacy_error:
            print(f"spaCy fallback failed: {str(spacy_error)}")
    
    print("After masking preview:", masked_text[:200])
    return masked_text

def semantic_validation(text):
    """
    Perform semantic validation using rule-based checks
    """
    score = 0
    checks = {
        'has_date': False,
        'has_numeric': False,
        'has_keywords': False
    }
    
    # Check for date formats (DD/MM/YYYY or DD-MM-YYYY)
    date_pattern = r'\b\d{2}[/-]\d{2}[/-]\d{4}\b'
    if re.search(date_pattern, text):
        checks['has_date'] = True
        score += 1
    
    # Check for numeric fields
    numeric_pattern = r'\b\d+\b'
    if re.search(numeric_pattern, text):
        checks['has_numeric'] = True
        score += 1
    
    # Check for mandatory keywords
    keywords = ['Name', 'Date', 'ID', 'Total']
    found_keywords = sum(1 for keyword in keywords if keyword.lower() in text.lower())
    if found_keywords >= 2:  # At least 2 keywords found
        checks['has_keywords'] = True
        score += 1
    
    # Normalize score to 0-1 range
    semantic_score = score / 3
    
    return semantic_score, checks

def perform_ela(image):
    """
    Perform Error Level Analysis (ELA) for tamper detection
    """
    # Convert PIL to numpy array
    img_array = np.array(image)
    
    # Save original image temporarily
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_original:
        original_path = tmp_original.name
        image.save(original_path, quality=95)
    
    # Save recompressed image at lower quality
    with tempfile.NamedTemporaryFile(suffix='.jpg', delete=False) as tmp_recompressed:
        recompressed_path = tmp_recompressed.name
        image.save(recompressed_path, quality=75)
    
    try:
        # Read both images
        original = cv2.imread(original_path)
        recompressed = cv2.imread(recompressed_path)
        
        # Compute absolute difference
        diff = cv2.absdiff(original, recompressed)
        
        # Amplify the difference
        ela_image = cv2.multiply(diff, 15)
        
        # Convert to grayscale for analysis
        ela_gray = cv2.cvtColor(ela_image, cv2.COLOR_BGR2GRAY)
        
        # Calculate tamper score based on intensity variation
        mean_intensity = np.mean(ela_gray)
        std_intensity = np.std(ela_gray)
        
        # Normalize tamper score (higher variation = higher tamper probability)
        tamper_score = min(1.0, (std_intensity / 128.0))
        
        return ela_image, tamper_score
        
    finally:
        # Clean up temporary files
        try:
            os.unlink(original_path)
            os.unlink(recompressed_path)
        except:
            pass

def extract_structured_fields(ocr_text):
    """
    Extract structured fields from OCR text using regex and keyword-based matching
    Returns a dictionary with extracted fields or "Not Detected" if not found
    """
    fields = {}
    
    # Name extraction - Look for patterns like "Name:" followed by text
    name_patterns = [
        r'Name[:\s]+([A-Za-z\s]+?)(?=\n|$|[A-Z][a-z]+:)',
        r'Customer[:\s]+([A-Za-z\s]+?)(?=\n|$)',
        r'Account Holder[:\s]+([A-Za-z\s]+?)(?=\n|$)',
        r'([A-Z][a-z]+\s+[A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'  # Capitalized names
    ]
    for pattern in name_patterns:
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            fields['name'] = match.group(1).strip()
            break
    else:
        fields['name'] = "Not Detected"
    
    # Account Number extraction - Various formats
    account_patterns = [
        r'Account[:\s\#]+(\d{10,20})',
        r'Acc[:\s\#]+(\d{10,20})',
        r'A/C[:\s\#]+(\d{10,20})',
        r'Account No[:\s]+(\d{10,20})',
        r'(\d{12,20})'  # Long numbers that could be account numbers
    ]
    for pattern in account_patterns:
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            fields['account_number'] = match.group(1).strip()
            break
    else:
        fields['account_number'] = "Not Detected"
    
    # IFSC Code extraction - Standard IFSC format
    ifsc_patterns = [
        r'IFSC[:\s]+([A-Z]{4}0[A-Z0-9]{6})',
        r'IFS C[:\s]+([A-Z]{4}0[A-Z0-9]{6})',
        r'([A-Z]{4}0[A-Z0-9]{6})'  # Direct IFSC pattern
    ]
    for pattern in ifsc_patterns:
        match = re.search(pattern, ocr_text)
        if match:
            fields['ifsc'] = match.group(1).strip()
            break
    else:
        fields['ifsc'] = "Not Detected"
    
    # Email extraction
    email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
    email_match = re.search(email_pattern, ocr_text)
    fields['email'] = email_match.group(0).strip() if email_match else "Not Detected"
    
    # Phone Number extraction - Various formats
    phone_patterns = [
        r'Phone[:\s]+(\+?\d{1,3}[-\s]?\d{10})',
        r'Mobile[:\s]+(\+?\d{1,3}[-\s]?\d{10})',
        r'Contact[:\s]+(\+?\d{1,3}[-\s]?\d{10})',
        r'\b(\+?\d{1,3}[-\s]?\d{10})\b',  # Direct phone pattern
        r'\b\d{10}\b'  # Simple 10-digit
    ]
    for pattern in phone_patterns:
        match = re.search(pattern, ocr_text)
        if match:
            fields['phone'] = match.group(1).strip()
            break
    else:
        fields['phone'] = "Not Detected"
    
    # Date extraction - DD/MM/YYYY or DD-MM-YYYY formats
    date_patterns = [
        r'Date[:\s]+(\d{2}[/-]\d{2}[/-]\d{4})',
        r'(\d{2}[/-]\d{2}[/-]\d{4})'  # Direct date pattern
    ]
    for pattern in date_patterns:
        match = re.search(pattern, ocr_text)
        if match:
            fields['date'] = match.group(1).strip()
            break
    else:
        fields['date'] = "Not Detected"
    
    # Branch Name extraction
    branch_patterns = [
        r'Branch[:\s]+([A-Za-z\s]+?)(?=\n|$|[A-Z][a-z]+:)',
        r'([A-Za-z\s]+Branch)',
        r'([A-Za-z\s]+Branch\s+[A-Za-z\s]+?)'
    ]
    for pattern in branch_patterns:
        match = re.search(pattern, ocr_text, re.IGNORECASE)
        if match:
            fields['branch'] = match.group(1).strip()
            break
    else:
        fields['branch'] = "Not Detected"
    
    return fields

def detect_document_type(ocr_text):
    """
    Detect document type using keyword-based matching
    Returns document type as string
    """
    text_lower = ocr_text.lower()
    
    # Banking document keywords
    banking_keywords = ['bank', 'ifsc', 'account', 'balance', 'deposit', 'withdrawal', 'cheque', 'passbook']
    banking_score = sum(1 for keyword in banking_keywords if keyword in text_lower)
    
    # Invoice keywords
    invoice_keywords = ['invoice', 'total', 'gst', 'tax', 'bill', 'amount due', 'payment']
    invoice_score = sum(1 for keyword in invoice_keywords if keyword in text_lower)
    
    # Certificate keywords
    certificate_keywords = ['certificate', 'issued', 'authority', 'certified', 'verification', 'official']
    certificate_score = sum(1 for keyword in certificate_keywords if keyword in text_lower)
    
    # Determine document type based on highest score
    if banking_score >= 2:
        return "Banking Document"
    elif invoice_score >= 2:
        return "Invoice"
    elif certificate_score >= 2:
        return "Certificate"
    else:
        return "Unknown Document"

def compute_authenticity_score(semantic_score, ocr_confidence, tamper_score, validation_result=None, entities=None):
    """
    Compute final authenticity score using rebalanced weighted combination and sigmoid function
    Enhanced with ML-based validation and entity coherence
    New weight distribution: Semantic 30%, OCR 15%, Tamper 20%, ML Classification 25%, Entity Coherence 10%
    """
    # Base weighted score calculation with new weights
    weighted_score = (
        0.30 * semantic_score +          # Semantic Validation: 30%
        0.15 * ocr_confidence +          # OCR Confidence: 15%
        0.20 * (1 - tamper_score)       # Tamper Analysis: 20%
    )
    
    # Add ML validation score with fixed logic for Unknown documents
    if validation_result and "overall_score" in validation_result:
        doc_type = validation_result.get("document_type", "Unknown")
        
        if doc_type == "Unknown":
            ml_validation_score = 0.4  # Fixed low score for Unknown documents
        else:
            ml_validation_score = validation_result["overall_score"]
        
        weighted_score += 0.25 * ml_validation_score  # ML Classification: 25%
    
    # Entity coherence bonus (reduced to 10%)
    if entities:
        entity_coherence = calculate_entity_coherence(entities)
        weighted_score += 0.10 * entity_coherence  # Entity Coherence: 10%
    
    # Ensure weighted_score stays within valid range
    weighted_score = max(0, min(1, weighted_score))
    
    # Apply sigmoid function
    final_score = 1 / (1 + math.exp(-5 * (weighted_score - 0.5)))
    
    return final_score, weighted_score

def calculate_entity_coherence(entities):
    """
    Calculate entity coherence score based on entity relationships
    """
    coherence_score = 0.0
    
    # Check for logical entity combinations
    has_persons = len(entities.get("persons", [])) > 0
    has_organizations = len(entities.get("organizations", [])) > 0
    has_dates = len(entities.get("dates", [])) > 0
    has_money = len(entities.get("money", [])) > 0
    
    # Logical combinations that make sense
    if has_persons and has_organizations:
        coherence_score += 0.3  # Person + Organization makes sense
    if has_dates and (has_persons or has_organizations):
        coherence_score += 0.3  # Dates with entities make sense
    if has_money and (has_persons or has_organizations):
        coherence_score += 0.2  # Money with entities makes sense
    if has_persons and has_dates and has_organizations:
        coherence_score += 0.2  # Complete entity set
    
    return min(coherence_score, 1.0)

def main():
    st.set_page_config(
        page_title="Automated Document Verification - Phase 1 Prototype",
        page_icon="🔍",
        layout="wide"
    )
    
    st.title("Automated Document Verification")
    st.markdown("---")
    
    # Initialize session state
    if 'processed' not in st.session_state:
        st.session_state.processed = False
    
    # 1️⃣ DOCUMENT UPLOAD SECTION
    st.header("1️⃣ Document Upload")
    uploaded_file = st.file_uploader(
        "Upload a document (JPG, JPEG, PNG, PDF)",
        type=['jpg', 'jpeg', 'png', 'pdf']
    )
    
    if uploaded_file is not None:
        # Check if uploaded file is PDF
        if uploaded_file.name.lower().endswith('.pdf'):
            st.info("📄 PDF detected - Processing multi-page document...")
            
            # Convert PDF to images in-memory
            images = convert_pdf_to_images(uploaded_file)
            
            if images:
                # Process PDF using pipeline
                try:
                    from ml_modules.pipeline import DocumentVerificationPipeline
                    pipeline = DocumentVerificationPipeline()
                    
                    # Display PDF results
                    st.subheader("📄 PDF Processing Results")
                    for page_number, image in enumerate(images):
                        st.write(f"**Page {page_number + 1}:**")
                        
                        # Process PIL image directly
                        result = pipeline.process_pil_image(image)
                        
                        # Display results
                        col1, col2, col3, col4 = st.columns(4)
                        with col1:
                            st.metric("Document Type", result.get('document_type', 'Unknown'))
                        with col2:
                            st.metric("Classification", f"{result.get('classification_confidence', 0):.2f}")
                        with col3:
                            st.metric("OCR Confidence", f"{result.get('ocr_confidence', 0):.2f}")
                        with col4:
                            st.metric("Final Score", f"{result.get('final_score', 0):.2f}")
                        
                        decision = result.get('decision', 'FLAGGED')
                        if decision == "VERIFIED":
                            st.success(f"Page {page_number + 1}: ✅ VERIFIED")
                        else:
                            st.error(f"Page {page_number + 1}: ⚠️ FLAGGED")
                        
                        st.markdown("---")
                    
                except Exception as e:
                    st.error(f"PDF processing failed: {str(e)}")
            else:
                st.error("Failed to convert PDF to images")
            
        else:
            # Handle image files
            image = Image.open(uploaded_file)
            st.image(image, caption="Original Uploaded Image", use_column_width=True)
            
            if st.button(" Process Document", type="primary"):
                st.session_state.processed = True
                st.session_state.image = image
                st.session_state.uploaded_file = uploaded_file
    
    if st.session_state.processed and 'image' in st.session_state:
        image = st.session_state.image
        
        # 2️⃣ IMAGE PREPROCESSING SECTION
        st.markdown("---")
        st.header("2️⃣ Image Preprocessing")
        
        gray, median_blur, sharpened = preprocess_image(image)
        
        col1, col2, col3 = st.columns(3)
        with col1:
            st.image(gray, caption="Grayscale", use_column_width=True)
        with col2:
            st.image(median_blur, caption="Median Blur", use_column_width=True)
        with col3:
            st.image(sharpened, caption="Sharpened", use_column_width=True)
        
        # 3️⃣ OCR SECTION
        st.markdown("---")
        st.header("3️⃣ OCR Text Extraction")
        
        with st.spinner("Extracting text..."):
            ocr_result = extract_text(image)
            extracted_text = ocr_result["text"]
            ocr_confidence = ocr_result["confidence"]
            boxes = ocr_result["detailed_boxes"]
        
        st.subheader("Extracted Text:")
        st.text_area("Full Text", extracted_text, height=200)
        st.info(f"OCR Confidence Score: {ocr_confidence:.2f}")
        
        # 4️⃣ PII MASKING SECTION
        st.markdown("---")
        st.header("4️⃣ PII Masking")
        
        masked_text = mask_pii(extracted_text)
        
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Original Text:")
            st.text(extracted_text)
        with col2:
            st.subheader("Masked Text:")
            st.text(masked_text)
        
        # 5️⃣ SEMANTIC VALIDATION SECTION
        st.markdown("---")
        st.header("5️⃣ Semantic Validation")
        
        semantic_score, checks = semantic_validation(extracted_text)
        
        st.write("Validation Checks:")
        st.write(f"- ✅ Date Format Found: {checks['has_date']}")
        st.write(f"- ✅ Numeric Fields Found: {checks['has_numeric']}")
        st.write(f"- ✅ Keywords Found: {checks['has_keywords']}")
        st.success(f"Semantic Score: {semantic_score:.2f}")
        
        # NEW: ML-POWERED STRUCTURED INFORMATION EXTRACTION SECTION
        st.markdown("---")
        st.header("🧠 ML-Powered Information Extraction (KIE)")
        
        with st.spinner("Performing ML-based analysis..."):
            # Ensure RGB image for CNN classifier
            rgb_image = ensure_rgb_image(image)
            
            # Image-based document type classification
            document_type, doc_confidence = predict_from_pil_image(rgb_image)
            
            # ML-based entity extraction
            entities = extract_entities(extracted_text)
            
            # Generate expected schema based on document type
            expected_schema = generate_expected_schema(document_type)
            
            # Document-aware semantic validation
            validation_result = validate_document(document_type, entities, extracted_text)
        
        # Display document type with confidence
        st.info(f"📄 **Document Type:** {document_type} (Confidence: {doc_confidence:.2f})")
        
        # Display extracted entities
        st.subheader("🔍 Extracted Entities:")
        
        # Create entity display
        entity_data = []
        for entity_type, entity_list in entities.items():
            if entity_list:  # Only show non-empty entities
                for entity in entity_list:
                    entity_data.append({
                        "Entity Type": entity_type.title(),
                        "Value": entity
                    })
        
        if entity_data:
            import pandas as pd
            entity_df = pd.DataFrame(entity_data)
            st.dataframe(entity_df, use_container_width=True, hide_index=True)
        else:
            st.warning("No entities detected")
        
        # Display validation results
        st.subheader("✅ Document Validation:")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Validation Score", f"{validation_result['overall_score']:.2f}")
            st.metric("Grade", validation_result['grade'])
        with col2:
            st.metric("Completeness", f"{validation_result['completeness_score']:.2f}")
            st.write(f"**Summary:** {validation_result['summary']}")
        
        # Show validation details
        with st.expander("🔧 Validation Details"):
            if validation_result['issues']:
                st.warning("Issues Found:")
                for issue in validation_result['issues']:
                    st.write(f"• {issue}")
            
            if validation_result['strengths']:
                st.success("Strengths:")
                for strength in validation_result['strengths']:
                    st.write(f"• {strength}")
        
        # Display expected schema
        with st.expander("📋 Expected Schema"):
            st.json(expected_schema)
        
        # 6️⃣ TAMPER DETECTION SECTION
        st.markdown("---")
        st.header("6️⃣ Tamper Detection (Error Level Analysis)")
        
        with st.spinner("Performing ELA analysis..."):
            ela_image, tamper_score = perform_ela(image)
        
        st.image(ela_image, caption="ELA Heatmap", use_column_width=True)
        st.warning(f"Tamper Score: {tamper_score:.2f} (Higher = More likely tampered)")
        
        # 7️⃣ ENHANCED AUTHENTICITY SCORING SECTION
        st.markdown("---")
        st.header("7️⃣ ML-Enhanced Authenticity Assessment")
        
        final_score, weighted_score = compute_authenticity_score(
            semantic_score, 
            ocr_confidence/100, 
            tamper_score, 
            validation_result, 
            entities
        )
        
        # Generate verdict
        verdict = "Likely Genuine" if final_score > 0.6 else "Potentially Tampered"
        
        # Create refined structured JSON output
        structured_output = {
            "document_type": document_type,
            "classification_confidence": round(doc_confidence, 2),
            "entities": {
                "persons": entities.get("persons", []),
                "organizations": entities.get("organizations", []),
                "dates": entities.get("dates", []),
                "locations": entities.get("locations", []),
                "money": entities.get("money", []),
                "tech": entities.get("tech", [])
            },
            "semantic_score": round(semantic_score, 2),
            "tamper_score": round(tamper_score, 2),
            "ml_validation_score": round(validation_result['overall_score'], 2),
            "final_score": round(final_score, 2),
            "verdict": "Likely Genuine" if final_score > 0.6 else "Suspicious"
        }
        
        # Display refined structured JSON output
        st.subheader("📋 Refined Structured Output (JSON)")
        st.json(structured_output)
        
        # Display comprehensive scores with new labels
        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Semantic (30%)", f"{semantic_score:.2f}")
        with col2:
            st.metric("OCR (15%)", f"{ocr_confidence/100:.2f}")
        with col3:
            st.metric("Tamper (20%)", f"{tamper_score:.2f}")
        with col4:
            st.metric("ML Valid (25%)", f"{validation_result['overall_score']:.2f}")
        with col5:
            st.metric("Final Score", f"{final_score:.2f}")
        
        # NEW: Enhanced Verification Breakdown using Pipeline
        st.markdown("---")
        st.header("🔍 Verification Breakdown")
        
        # Import and use the pipeline for comprehensive verification
        try:
            from ml_modules.pipeline import DocumentVerificationPipeline
            
            # Save current image temporarily for pipeline processing
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
                image.save(tmp_file.name)
                tmp_path = tmp_file.name
            
            # Process through pipeline
            pipeline = DocumentVerificationPipeline()
            result = pipeline.process_document(tmp_path)
            
            # Clean up temp file
            os.unlink(tmp_path)
            
            # Extract scores from pipeline result
            document_type = result.get("document_type", "Unknown")
            classification_conf = result.get("classification_confidence", 0.0)
            ocr_confidence = result.get("ocr_confidence", 0.0)
            tamper_score = result.get("tamper_score", 0.0)
            final_score = result.get("final_score", 0.0)
            decision = result.get("decision", "FLAGGED")
            reasoning = result.get("reasoning", {})
            
            # Display document type
            st.info(f"📄 **Document Type:** {document_type}")
            
            # Display metrics in columns
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Classification Confidence", f"{classification_conf:.2f}")
            with col2:
                st.metric("OCR Confidence", f"{ocr_confidence:.2f}")
            with col3:
                st.metric("Tamper Score", f"{tamper_score:.2f}")
            with col4:
                st.metric("Final Score", f"{final_score:.2f}")
            
            # Display decision prominently
            st.markdown("### 🎯 Final Decision")
            if decision == "VERIFIED":
                st.success("✅ VERIFIED")
            else:
                st.error("⚠️ FLAGGED")
            
            # Add decision interpretation
            if final_score >= 0.85:
                interpretation = "High confidence genuine document."
            elif final_score >= 0.7:
                interpretation = "Moderate confidence — review recommended."
            else:
                interpretation = "Flagged for further verification."
            
            st.info(f"🧠 Interpretation: {interpretation}")
            
            # Show reasoning breakdown
            with st.expander("🔍 Verification Reasoning"):
                st.write("**Weight Breakdown:**")
                for key, value in reasoning.items():
                    if key != "notes":
                        st.write(f"- {key.replace('_', ' ').title()}: {value}")
                st.write(f"**Notes:** {reasoning.get('notes', 'No additional notes')}")
                
        except Exception as e:
            st.error(f"Pipeline verification failed: {str(e)}")
            # Fallback to original display
            st.info(f"📄 **Document Type:** {document_type} (Confidence: {doc_confidence:.2f})")
        
        # Edge Case Testing Helper
        st.markdown("---")
        st.header("🧪 Edge Case Testing")
        
        st.write("Upload challenging images to test system robustness:")
        st.write("- Blurred images")
        st.write("- Rotated documents") 
        st.write("- Cropped or partial documents")
        st.write("- Low quality scans")
        
        # Allow additional upload for edge case testing
        edge_test_file = st.file_uploader(
            "Upload edge case image for testing",
            type=['png', 'jpg', 'jpeg'],
            key="edge_test"
        )
        
        if edge_test_file is not None:
            st.write("🔬 Edge Case Analysis:")
            edge_image = Image.open(edge_test_file).convert('RGB')
            
            # Quick classification test
            try:
                edge_rgb_image = ensure_rgb_image(edge_image)
                edge_doc_type, edge_confidence = predict_from_pil_image(edge_rgb_image)
                st.write(f"**Document Type:** {edge_doc_type}")
                st.write(f"**Classification Confidence:** {edge_confidence:.2f}")
                
                if edge_confidence < 0.45:
                    st.warning("⚠️ Low confidence classification - document may be challenging")
                else:
                    st.success("✅ Classification successful")
                    
            except Exception as e:
                st.error(f"❌ Edge case processing failed: {str(e)}")
                st.write("This helps identify system limitations for improvement.")

if __name__ == "__main__":
    main()
