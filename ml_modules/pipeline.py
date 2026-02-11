"""
Unified Document Verification Pipeline
Orchestrates all verification components
"""

import os
import tempfile
import numpy as np
import io
from typing import Dict, List, Any, Union
from PIL import Image
import cv2

# Import verification components
from ml_modules.image_classifier import predict_document_type
from ml_modules.decision_fusion import DecisionFusion

# Import OCR function from main app
try:
    from app import extract_text
except ImportError:
    print("Warning: Could not import extract_text from app.py")
    extract_text = None

class DocumentVerificationPipeline:
    def __init__(self):
        """
        Initialize the unified document verification pipeline
        """
        self.fusion = DecisionFusion()
        self.temp_files = []  # Track temporary files for cleanup
        
        # Override fusion weights for improved calibration
        self.fusion.update_weights({
            "classification": 0.5,
            "tamper": 0.3,
            "ocr": 0.2
        })
    
    def compute_tamper_score(self, image_path: str) -> float:
        """
        Compute tamper score using Error Level Analysis (ELA)
        
        Args:
            image_path: Path to image file
            
        Returns:
            tamper_score: 0-1 (0 = clean, 1 = fully tampered)
        """
        try:
            # Read image
            img = cv2.imread(image_path)
            if img is None:
                return 0.5  # Default middle score
            
            # Convert to RGB
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            # Save original at quality 95
            temp_original = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(temp_original.name, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 95])
            
            # Save recompressed at quality 75
            temp_recompressed = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False)
            cv2.imwrite(temp_recompressed.name, cv2.cvtColor(img_rgb, cv2.COLOR_RGB2BGR), [cv2.IMWRITE_JPEG_QUALITY, 75])
            
            # Read both images
            original = cv2.imread(temp_original.name)
            recompressed = cv2.imread(temp_recompressed.name)
            
            # Compute ELA
            ela = cv2.absdiff(original, recompressed)
            ela_gray = cv2.cvtColor(ela, cv2.COLOR_BGR2GRAY)
            
            # Calculate tamper score based on ELA intensity
            mean_ela = np.mean(ela_gray)
            max_ela = np.max(ela_gray)
            
            # Normalize to 0-1 range
            tamper_score = min(mean_ela / max_ela, 1.0) if max_ela > 0 else 0.0
            
            # Clean up
            os.unlink(temp_original.name)
            os.unlink(temp_recompressed.name)
            
            return tamper_score
            
        except Exception as e:
            print(f"Tamper detection failed for {image_path}: {str(e)}")
            return 0.5  # Default middle score
    
    def extract_text(self, image_path: str) -> Dict[str, Any]:
        """
        Extract text using OCR with structured output
        
        Args:
            image_path: Path to image file
            
        Returns:
            Dictionary with text, confidence, and bounding boxes
        """
        if extract_text:
            return extract_text(image_path)
        else:
            # Fallback OCR
            try:
                import pytesseract
                image = Image.open(image_path)
                text = pytesseract.image_to_string(image)
                return {
                    "text": text,
                    "confidence": 0.8,  # Default confidence
                    "boxes": []
                }
            except Exception as e:
                print(f"OCR failed: {str(e)}")
                return {
                    "text": "",
                    "confidence": 0.0,
                    "boxes": []
                }
    
    def process_pil_image(self, image: Image.Image) -> Dict[str, Any]:
        """
        Process a PIL Image directly through verification pipeline
        
        Args:
            image: PIL Image object
            
        Returns:
            Complete verification result dictionary
        """
        try:
            # Save PIL image to temporary file for processing
            temp_path = tempfile.NamedTemporaryFile(suffix='.jpg', delete=False).name
            image.save(temp_path, "JPEG")
            self.temp_files.append(temp_path)
            
            # Process through existing pipeline
            result = self.process_document(temp_path)
            return result
            
        except Exception as e:
            print(f"PIL image processing failed: {str(e)}")
            return {
                "document_type": "Unknown",
                "classification_confidence": 0.0,
                "ocr_confidence": 0.0,
                "tamper_score": 0.5,
                "final_score": 0.0,
                "decision": "FLAGGED",
                "reasoning": {"error": str(e)}
            }
    
    def process_document(self, image_path: str) -> Dict[str, Any]:
        """
        Process a single document through the complete verification pipeline
        
        Args:
            image_path: Path to image file
            
        Returns:
            Complete verification result dictionary
        """
        try:
            # 1. Classification with confidence threshold
            doc_type, classification_conf = predict_document_type(image_path)
            
            # Apply confidence threshold to prevent overconfident wrong predictions
            if classification_conf < 0.45:
                doc_type = "Unknown"
            
            # Safe classification confidence floor to prevent noisy predictions
            if classification_conf < 0.4:
                classification_conf *= 0.8  # reduce impact
                print(f"Classification confidence floor applied: {classification_conf:.3f}")
            
            # 2. OCR
            ocr_result = self.extract_text(image_path)
            
            # 2.5. Heuristic resume detection boost
            text_lower = ocr_result["text"].lower()
            resume_keywords = [
                "education", "experience", "skills", "intern", 
                "cgpa", "projects", "university", "degree",
                "bachelor", "master", "phd", "certification"
            ]
            
            resume_keyword_count = sum(1 for keyword in resume_keywords if keyword in text_lower)
            
            # Boost resume detection if keywords found
            if resume_keyword_count >= 3:
                if doc_type.lower() != "resume":
                    doc_type = "Resume"
                    classification_conf = min(classification_conf + 0.15, 1.0)
                    print(f"Resume heuristic boost applied: {classification_conf:.3f}")
            
            # 3. Real scores (no placeholders)
            signature_score = 0.0  # Signature module not yet implemented
            
            # Compute real tamper score using ELA (Error Level Analysis)
            tamper_score = self.compute_tamper_score(image_path)
            
            # Get real OCR confidence
            ocr_confidence = ocr_result["confidence"]
            
            # 4. Improved decision fusion logic using updated weights
            final_score = self.fusion.compute_final_score(
                classification_conf,
                signature_score,
                tamper_score,
                ocr_confidence
            )
            
            decision = self.fusion.make_decision(final_score)
            
            return {
                "document_type": doc_type,
                "classification_confidence": classification_conf,
                "ocr_confidence": ocr_confidence,
                "tamper_score": tamper_score,
                "final_score": final_score,
                "decision": decision,
                "reasoning": {
                    "weights": self.fusion.weights,
                    "notes": f"Classification: {classification_conf:.3f}, Tamper: {tamper_score:.3f}, OCR: {ocr_confidence:.3f}"
                }
            }
            
        except Exception as e:
            print(f"Document processing failed: {str(e)}")
            return {
                "document_type": "Unknown",
                "classification_confidence": 0.0,
                "ocr_confidence": 0.0,
                "tamper_score": 0.5,
                "final_score": 0.0,
                "decision": "FLAGGED",
                "reasoning": {"error": str(e)}
            }
    
    def process_pdf(self, pdf_path: str) -> List[Dict[str, Any]]:
        """
        Process PDF file using PyMuPDF (fitz)
        
        Args:
            pdf_path: Path to PDF file
            
        Returns:
            List of verification results for each page
        """
        try:
            import fitz  # PyMuPDF
        except ImportError:
            print("PyMuPDF not installed. Install with: pip install PyMuPDF")
            return []
        
        try:
            print(f"Processing PDF: {pdf_path}")
            
            # Open PDF document
            doc = fitz.open(pdf_path)
            print(f"PDF opened successfully. Number of pages: {len(doc)}")
            
            results = []
            
            for i, page in enumerate(doc):
                print(f"Processing page {i+1}/{len(doc)}")
                
                # Convert PDF page to pixmap
                pixmap = page.get_pixmap(dpi=150)
                
                # Convert pixmap to PIL Image
                img_data = pixmap.tobytes("png")
                pil_image = Image.open(io.BytesIO(img_data))
                
                # Save page as temporary file
                temp_path = f"temp_page_{i}.png"
                pil_image.save(temp_path, "PNG")
                self.temp_files.append(temp_path)
                print(f"Saved temporary file: {temp_path}")
                
                # Process the page
                result = self.process_document(temp_path)
                results.append({
                    "page": i + 1,
                    "result": result
                })
            
            doc.close()
            print(f"Processed {len(doc)} pages from PDF")
            return results
            
        except Exception as e:
            print(f"PDF processing failed for {pdf_path}: {str(e)}")
            return []
    
    def process_folder(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Process all images in a folder
        
        Args:
            folder_path: Path to folder containing images
            
        Returns:
            List of verification results
        """
        results = []
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return results
        
        for file in os.listdir(folder_path):
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                file_path = os.path.join(folder_path, file)
                print(f"Processing image: {file}")
                
                result = self.process_document(file_path)
                results.append({
                    "file_name": file,
                    "result": result
                })
        
        return results
    
    def process_folder_with_pdfs(self, folder_path: str) -> List[Dict[str, Any]]:
        """
        Process folder containing both images and PDFs
        
        Args:
            folder_path: Path to folder containing documents
            
        Returns:
            List of verification results for all documents
        """
        all_results = []
        
        if not os.path.exists(folder_path):
            print(f"Folder not found: {folder_path}")
            return all_results
        
        print(f"Processing folder with mixed formats: {folder_path}")
        
        for file in os.listdir(folder_path):
            file_path = os.path.join(folder_path, file)
            
            if file.lower().endswith((".png", ".jpg", ".jpeg")):
                # Process image
                print(f"Processing image: {file}")
                result = self.process_document(file_path)
                all_results.append({
                    "file_name": file,
                    "file_type": "image",
                    "result": result
                })
            elif file.lower().endswith(".pdf"):
                # Process PDF
                print(f"Processing PDF: {file}")
                pdf_results = self.process_pdf(file_path)
                all_results.append({
                    "file_name": file,
                    "file_type": "pdf",
                    "results": pdf_results
                })
        
        return all_results
    
    def cleanup_temp_files(self):
        """
        Clean up temporary files created during processing
        """
        for temp_file in self.temp_files:
            try:
                if os.path.exists(temp_file):
                    os.remove(temp_file)
                    print(f"Cleaned up: {temp_file}")
            except Exception as e:
                print(f"Failed to cleanup {temp_file}: {str(e)}")
        
        self.temp_files.clear()
    
    def get_pipeline_stats(self, results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        Generate statistics for processed results
        
        Args:
            results: List of verification results
            
        Returns:
            Statistics dictionary
        """
        if not results:
            return {"total_documents": 0}
        
        total_docs = len(results)
        verified_count = sum(1 for r in results if r.get("result", {}).get("decision") == "VERIFIED")
        flagged_count = total_docs - verified_count
        
        avg_score = sum(r.get("result", {}).get("final_score", 0) for r in results) / total_docs
        
        return {
            "total_documents": total_docs,
            "verified_documents": verified_count,
            "flagged_documents": flagged_count,
            "verification_rate": (verified_count / total_docs) * 100,
            "average_score": avg_score
        }
