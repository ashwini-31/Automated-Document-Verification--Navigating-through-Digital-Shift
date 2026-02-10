#  Automated Document Verification System

An end-to-end ML-powered document verification system that combines 
image classification, OCR, PII masking, tamper detection, and 
multi-signal decision fusion for intelligent document authentication.

---

##  Project Overview

This system verifies documents by combining multiple AI signals:

-  CNN-based Document Classification (Trained on Tobacco3482)
-  OCR Text Extraction (Tesseract)
-  Named Entity Recognition (spaCy)
-  PII Masking (Emails, Phones, Names)
-  Tamper Detection using Error Level Analysis (ELA)
-  Semantic Validation
-  Multi-signal Decision Fusion
-  Streamlit Interactive UI

The goal is to simulate a production-style document authentication pipeline.

---

##  Model Details

###  Document Classifier
- Architecture: ResNet18
- Dataset: Tobacco3482
- Classes: Letter, Resume, Report, Memo, Email, etc.
- Output: Document type + confidence score

###  OCR
- Engine: Tesseract OCR
- Confidence scoring based on extracted word confidence

###  PII Masking
- Regex-based detection for:
  - Emails
  - Phone numbers
- spaCy-based NER for PERSON entities
- Whitelist protection for technical terms

###  Tamper Detection
- Method: Error Level Analysis (ELA)
- Measures recompression inconsistencies
- Outputs tamper confidence score

###  Decision Fusion

Final score is computed using weighted fusion of:
- Classification confidence
- OCR confidence
- Tamper score
- Semantic validation score

---

##  Output Example

The system provides:

- Document Type
- Classification Confidence
- OCR Confidence
- Extracted Entities
- Semantic Score
- Tamper Score
- Final Authenticity Score
- Final Verdict (Verified / Flagged)

All intermediate scores are visible in the UI for transparency.

---

##  UI

Built using Streamlit.

Features:
- Single image upload
- Image preprocessing visualization
- OCR display
- PII masking comparison
- Entity extraction table
- Authenticity breakdown
- Edge case testing panel

---

##  Current Limitations

- PDF support is under debugging
- PII masking requires further refinement
- Model trained only on Tobacco3482 (limited generalization)
- Resume classification needs further calibration

---

##  Tech Stack

- Python
- PyTorch
- torchvision
- OpenCV
- Tesseract OCR
- spaCy
- Streamlit
- NumPy
- Pillow

---

##  Dataset

Tobacco3482 Dataset  
Used for document classification training.

---

##  How to Run

```bash
git clone <repo_link>
cd Document_Verification
pip install -r requirements.txt
streamlit run app.py
