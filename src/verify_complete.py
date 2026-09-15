"""
Complete Verification System
Combines RQ1 (Signature) + RQ2 (Explainability) + RQ3 (Blockchain)
"""

import numpy as np
from src.qr_blockchain import QRBlockchainAuth
from src.data_loader import SignatureDatasetLoader


class CompleteVerifier:
    """
    Complete document verification system
    Combines:
    - RQ1: Siamese CNN signature verification
    - RQ2: Grad-CAM explainability
    - RQ3: Blockchain document integrity
    """
    
    def __init__(self, model, auth, loader):
        self.model = model
        self.auth = auth
        self.loader = loader
    
    def verify_complete(self, scanned_content, signature_image, 
                        reference_signature, qr_data):
        """
        Complete verification of a document
        
        Returns:
            dict: Complete verification results
        """
        print("=" * 60)
        print("🔍 COMPLETE DOCUMENT VERIFICATION")
        print("=" * 60)
        
        # =====================================================================
        # RQ3: Document Integrity Check
        # =====================================================================
        print("\n📄 RQ3: Document Integrity Check")
        print("-" * 40)
        
        doc_result = self.auth.verify_document(scanned_content, qr_data)
        
        print(f"   Content Verified: {doc_result['content_verified']}")
        print(f"   Metadata Verified: {doc_result['metadata_verified']}")
        print(f"   Signature Verified: {doc_result['signature_verified']}")
        print(f"   Blockchain Verified: {doc_result['blockchain_verified']}")
        print(f"   Status: {doc_result['status']}")
        
        # =====================================================================
        # RQ1: Signature Verification (Siamese CNN)
        # =====================================================================
        print("\n✍️ RQ1: Signature Verification")
        print("-" * 40)
        
        # Prepare signature for model
        sig1 = signature_image.reshape(1, 128, 128, 1)
        sig2 = reference_signature.reshape(1, 128, 128, 1)
        
        # Get prediction
        pred = self.model.predict([sig1, sig2], verbose=0)[0][0]
        signature_genuine = pred > 0.5
        
        print(f"   Prediction: {pred:.4f}")
        print(f"   Signature Genuine: {signature_genuine}")
        print(f"   Confidence: {abs(pred - 0.5) * 2:.2f}")
        
        # =====================================================================
        # Combined Result
        # =====================================================================
        print("\n" + "=" * 60)
        print("📊 COMBINED RESULT")
        print("=" * 60)
        
        if doc_result["status"] == "FULLY_AUTHENTIC" and signature_genuine:
            final_status = "COMPLETELY_AUTHENTIC"
            final_message = "✅ Document AND signature are authentic"
            color = "green"
        elif doc_result["status"] == "FULLY_AUTHENTIC" and not signature_genuine:
            final_status = "SIGNATURE_FORGED"
            final_message = "❌ Document is authentic BUT signature is forged"
            color = "red"
        elif doc_result["status"] != "FULLY_AUTHENTIC" and signature_genuine:
            final_status = "DOCUMENT_TAMPERED"
            final_message = "❌ Signature is genuine BUT document is tampered"
            color = "red"
        else:
            final_status = "BOTH_FORGED"
            final_message = "❌ Both document AND signature are fraudulent"
            color = "red"
        
        print(f"\n   Final Status: {final_status}")
        print(f"   Message: {final_message}")
        
        return {
            "final_status": final_status,
            "final_message": final_message,
            "document_check": doc_result,
            "signature_check": {
                "prediction": float(pred),
                "is_genuine": signature_genuine,
                "confidence": abs(pred - 0.5) * 2
            }
        }