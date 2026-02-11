"""
Test Script for Document Verification Pipeline
Tests all pipeline functionality
"""

import os
import sys
from ml_modules.pipeline import DocumentVerificationPipeline

def test_single_document():
    """Test single document processing"""
    print("🧪 Testing Single Document Processing")
    print("-" * 40)
    
    pipeline = DocumentVerificationPipeline()
    
    # Test with a sample image (create a dummy one if needed)
    sample_image = "sample.jpg"
    
    if not os.path.exists(sample_image):
        print(f"⚠️ Sample image {sample_image} not found")
        print("Creating a simple test image...")
        
        # Create a simple test image
        from PIL import Image, ImageDraw, ImageFont
        img = Image.new('RGB', (800, 600), color='white')
        draw = ImageDraw.Draw(img)
        
        # Draw some text
        try:
            font = ImageFont.truetype("arial.ttf", 32)
        except:
            font = ImageFont.load_default()
        
        draw.text((50, 50), "TEST DOCUMENT", fill='black', font=font)
        draw.text((50, 100), "This is a test document", fill='black', font=font)
        draw.text((50, 150), "Date: 2024-01-01", fill='black', font=font)
        draw.text((50, 200), "Reference: TEST-001", fill='black', font=font)
        
        img.save(sample_image)
        print(f"✅ Created test image: {sample_image}")
    
    # Process the document
    result = pipeline.process_document(sample_image)
    
    print("\n📋 Single Document Result:")
    print("=" * 40)
    for key, value in result.items():
        if key == "ocr_result":
            print(f"{key}:")
            for ocr_key, ocr_value in value.items():
                print(f"  {ocr_key}: {ocr_value}")
        else:
            print(f"{key}: {value}")
    
    return result

def test_folder_processing():
    """Test folder processing"""
    print("\n🧪 Testing Folder Processing")
    print("-" * 40)
    
    pipeline = DocumentVerificationPipeline()
    
    # Create test folder with sample images
    test_folder = "test_documents"
    if not os.path.exists(test_folder):
        os.makedirs(test_folder)
        print(f"✅ Created test folder: {test_folder}")
    
    # Create sample images in the folder
    from PIL import Image, ImageDraw, ImageFont
    
    sample_docs = [
        ("invoice.jpg", "INVOICE #123\nAmount: $500.00\nDue: 2024-01-15"),
        ("letter.jpg", "Dear Sir/Madam,\nThis is a formal letter.\nSincerely,\nJohn Doe"),
        ("form.jpg", "APPLICATION FORM\nName: ___________\nAddress: ___________")
    ]
    
    for filename, text_content in sample_docs:
        filepath = os.path.join(test_folder, filename)
        if not os.path.exists(filepath):
            img = Image.new('RGB', (800, 600), color='white')
            draw = ImageDraw.Draw(img)
            
            try:
                font = ImageFont.truetype("arial.ttf", 24)
            except:
                font = ImageFont.load_default()
            
            # Draw text
            lines = text_content.split('\n')
            for i, line in enumerate(lines):
                draw.text((50, 50 + i*40), line, fill='black', font=font)
            
            img.save(filepath)
            print(f"✅ Created sample document: {filename}")
    
    # Process the folder
    folder_results = pipeline.process_folder(test_folder)
    
    print(f"\n📋 Folder Results ({len(folder_results)} files):")
    print("=" * 40)
    
    for file_result in folder_results:
        filename = file_result['file_name']
        result = file_result['result']
        decision = result.get('decision', 'UNKNOWN')
        doc_type = result.get('document_type', 'Unknown')
        final_score = result.get('final_score', 0)
        
        print(f"📄 {filename}")
        print(f"   Type: {doc_type}")
        print(f"   Decision: {decision}")
        print(f"   Score: {final_score:.3f}")
        print()
    
    # Generate statistics
    stats = pipeline.get_pipeline_stats(folder_results)
    print("📊 Pipeline Statistics:")
    print("=" * 40)
    for key, value in stats.items():
        print(f"{key}: {value}")
    
    return folder_results

def test_pdf_processing():
    """Test PDF processing"""
    print("\n🧪 Testing PDF Processing")
    print("-" * 40)
    
    pipeline = DocumentVerificationPipeline()
    
    # Check if we have any PDF files
    test_pdf = "sample.pdf"
    
    if not os.path.exists(test_pdf):
        print(f"⚠️ Sample PDF {test_pdf} not found")
        print("PDF processing test skipped (requires actual PDF file)")
        return []
    
    # Process the PDF
    pdf_results = pipeline.process_pdf(test_pdf)
    
    print(f"\n📋 PDF Results ({len(pdf_results)} pages):")
    print("=" * 40)
    
    for page_result in pdf_results:
        page_num = page_result['page']
        result = page_result['result']
        decision = result.get('decision', 'UNKNOWN')
        doc_type = result.get('document_type', 'Unknown')
        
        print(f"📄 Page {page_num}")
        print(f"   Type: {doc_type}")
        print(f"   Decision: {decision}")
        print()
    
    return pdf_results

def test_decision_fusion():
    """Test decision fusion component"""
    print("\n🧪 Testing Decision Fusion")
    print("-" * 40)
    
    from ml_modules.decision_fusion import DecisionFusion
    
    fusion = DecisionFusion()
    
    # Test cases
    test_cases = [
        {
            "name": "High Confidence Document",
            "classification_conf": 0.95,
            "signature_score": 0.85,
            "tamper_score": 0.05
        },
        {
            "name": "Medium Confidence Document", 
            "classification_conf": 0.70,
            "signature_score": 0.60,
            "tamper_score": 0.30
        },
        {
            "name": "Low Confidence Document",
            "classification_conf": 0.40,
            "signature_score": 0.30,
            "tamper_score": 0.80
        }
    ]
    
    for test_case in test_cases:
        name = test_case["name"]
        class_conf = test_case["classification_conf"]
        sig_score = test_case["signature_score"]
        tamper_score = test_case["tamper_score"]
        
        final_score = fusion.compute_final_score(class_conf, sig_score, tamper_score)
        decision = fusion.make_decision(final_score)
        breakdown = fusion.get_score_breakdown(class_conf, sig_score, tamper_score)
        
        print(f"📋 {name}:")
        print(f"   Classification: {class_conf:.2f}")
        print(f"   Signature: {sig_score:.2f}")
        print(f"   Tamper: {tamper_score:.2f}")
        print(f"   Final Score: {final_score:.3f}")
        print(f"   Decision: {decision}")
        print(f"   Breakdown: {breakdown}")
        print()

def main():
    """Run all tests"""
    print("🚀 Document Verification Pipeline Test Suite")
    print("=" * 60)
    
    try:
        # Test individual components
        test_decision_fusion()
        
        # Test single document processing
        test_single_document()
        
        # Test folder processing
        test_folder_processing()
        
        # Test PDF processing
        test_pdf_processing()
        
        print("\n✅ All tests completed successfully!")
        print("=" * 60)
        
        # Cleanup
        print("\n🧹 Cleaning up test files...")
        cleanup_files = ["sample.jpg", "test_documents"]
        for cleanup_file in cleanup_files:
            if os.path.exists(cleanup_file):
                if os.path.isfile(cleanup_file):
                    os.remove(cleanup_file)
                    print(f"   Removed: {cleanup_file}")
                elif os.path.isdir(cleanup_file):
                    import shutil
                    shutil.rmtree(cleanup_file)
                    print(f"   Removed folder: {cleanup_file}")
        
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if not success:
        sys.exit(1)
