"""
Document Type Classifier Module
Uses TF-IDF vectorizer and Logistic Regression for document classification
Enhanced with Resume class and keyword override logic
"""

import os
import pickle
import re
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import numpy as np

class DocumentClassifier:
    def __init__(self):
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english', ngram_range=(1, 2))
        self.classifier = LogisticRegression(random_state=42, max_iter=1000)
        self.model_path = 'trained_models/doc_classifier.pkl'
        self.classes = ['Banking', 'Legal', 'Certificate', 'Invoice', 'Resume']
        self.is_trained = False
        
        # Keyword override patterns
        self.keyword_overrides = {
            'Resume': [
                'CGPA', 'Experience', 'Education', 'Skills', 'Summary',
                'Objective', 'Qualification', 'Projects', 'Certifications',
                'References', 'Portfolio', 'LinkedIn', 'GitHub'
            ],
            'Legal': [
                'Stamp', 'Non Judicial', 'Rs.', 'Serial No', 'Denomination',
                'Agreement', 'Deed', 'Notary', 'Witness', 'Jurisdiction',
                'Execution', 'Parties', 'Consideration', 'Counsel'
            ],
            'Banking': [
                'IFSC', 'Account No', 'Branch', 'Balance', 'Transaction',
                'Deposit', 'Withdrawal', 'MICR', 'Passbook', 'Cheque'
            ]
        }
        
    def load_training_data(self, training_dir='training_data'):
        """
        Load training data from text files
        Each file represents one document type
        """
        texts = []
        labels = []
        
        for filename in os.listdir(training_dir):
            if filename.endswith('.txt'):
                file_path = os.path.join(training_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f:
                    content = f.read().strip()
                    
                # Extract document type from filename
                filename_lower = filename.lower()
                if 'bank' in filename_lower:
                    label = 'Banking'
                elif 'legal' in filename_lower:
                    label = 'Legal'
                elif 'certificate' in filename_lower:
                    label = 'Certificate'
                elif 'invoice' in filename_lower:
                    label = 'Invoice'
                elif 'resume' in filename_lower:
                    label = 'Resume'
                else:
                    continue
                    
                # Split content into smaller chunks for better training
                sentences = re.split(r'[.!?]+', content)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if len(sentence) > 10:  # Only keep meaningful sentences
                        texts.append(sentence)
                        labels.append(label)
                        
                # Also add the full document
                texts.append(content)
                labels.append(label)
        
        return texts, labels
    
    def apply_keyword_override(self, text: str, ml_prediction: str, ml_confidence: float) -> tuple:
        """
        Apply keyword override logic after ML prediction
        """
        text_lower = text.lower()
        
        # Check for Resume keywords
        resume_matches = sum(1 for keyword in self.keyword_overrides['Resume'] 
                            if keyword.lower() in text_lower)
        if resume_matches >= 2:  # At least 2 resume keywords
            return 'Resume', 0.9  # High confidence override
        
        # Check for Legal keywords
        legal_matches = sum(1 for keyword in self.keyword_overrides['Legal'] 
                          if keyword.lower() in text_lower)
        if legal_matches >= 2:  # At least 2 legal keywords
            return 'Legal', 0.9  # High confidence override
        
        # Check for Banking keywords
        banking_matches = sum(1 for keyword in self.keyword_overrides['Banking'] 
                             if keyword.lower() in text_lower)
        if banking_matches >= 2:  # At least 2 banking keywords
            return 'Banking', 0.9  # High confidence override
        
        # Return ML prediction if no overrides match
        return ml_prediction, ml_confidence
    
    def train_classifier(self):
        """
        Train the document classifier
        """
        print("Loading training data...")
        texts, labels = self.load_training_data()
        
        if not texts:
            raise ValueError("No training data found!")
        
        print(f"Training with {len(texts)} samples...")
        print(f"Classes: {set(labels)}")
        
        # Split data for training
        X_train, X_test, y_train, y_test = train_test_split(
            texts, labels, test_size=0.2, random_state=42, stratify=labels
        )
        
        # Vectorize text
        print("Vectorizing text...")
        X_train_vec = self.vectorizer.fit_transform(X_train)
        X_test_vec = self.vectorizer.transform(X_test)
        
        # Train classifier
        print("Training classifier...")
        self.classifier.fit(X_train_vec, y_train)
        
        # Evaluate
        y_pred = self.classifier.predict(X_test_vec)
        print("Classification Report:")
        print(classification_report(y_test, y_pred))
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        print("Model saved successfully!")
        
    def save_model(self):
        """
        Save the trained model and vectorizer
        """
        os.makedirs(os.path.dirname(self.model_path), exist_ok=True)
        model_data = {
            'vectorizer': self.vectorizer,
            'classifier': self.classifier,
            'classes': self.classes
        }
        with open(self.model_path, 'wb') as f:
            pickle.dump(model_data, f)
    
    def load_model(self):
        """
        Load the trained model
        """
        if os.path.exists(self.model_path):
            with open(self.model_path, 'rb') as f:
                model_data = pickle.load(f)
                self.vectorizer = model_data['vectorizer']
                self.classifier = model_data['classifier']
                self.classes = model_data['classes']
                self.is_trained = True
            print("Model loaded successfully!")
            return True
        else:
            print("No trained model found!")
            return False
    
    def predict_document_type(self, text, confidence_threshold=0.35):
        """
        Predict document type with confidence and keyword overrides
        Returns (predicted_type, confidence_score)
        """
        if not self.is_trained:
            if not self.load_model():
                # Try to train if model doesn't exist
                self.train_classifier()
        
        # Preprocess text
        text = text.strip()
        if not text:
            return "Unknown", 0.0
        
        # Vectorize
        text_vec = self.vectorizer.transform([text])
        
        # Get prediction probabilities
        probabilities = self.classifier.predict_proba(text_vec)[0]
        predicted_class_idx = np.argmax(probabilities)
        confidence = probabilities[predicted_class_idx]
        
        # Get predicted class
        predicted_class = self.classifier.classes_[predicted_class_idx]
        
        # Apply keyword overrides
        final_prediction, final_confidence = self.apply_keyword_override(
            text, predicted_class, confidence
        )
        
        # Apply confidence threshold (reduced to 0.35)
        if final_confidence < confidence_threshold:
            return "Unknown", final_confidence
        
        return final_prediction, final_confidence
    
    def get_class_probabilities(self, text):
        """
        Get probability distribution for all classes
        """
        if not self.is_trained:
            if not self.load_model():
                return {}
        
        text_vec = self.vectorizer.transform([text])
        probabilities = self.classifier.predict_proba(text_vec)[0]
        
        return dict(zip(self.classifier.classes_, probabilities))

# Training function for standalone usage
def train_classifier():
    """
    Train and save the document classifier
    """
    classifier = DocumentClassifier()
    classifier.train_classifier()
    return classifier

# Prediction function for app usage
def predict_document_type(text):
    """
    Predict document type using trained model with keyword overrides
    """
    classifier = DocumentClassifier()
    return classifier.predict_document_type(text)

if __name__ == "__main__":
    # Train the classifier when run directly
    train_classifier()
