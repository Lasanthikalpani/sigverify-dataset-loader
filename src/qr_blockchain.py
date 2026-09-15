"""
SigVerify - QR + Blockchain Integration
Combines dual-hashing with batch blockchain
"""

import hashlib
import json
from datetime import datetime
from Crypto.Signature import pkcs1_15
from Crypto.PublicKey import RSA
from Crypto.Hash import SHA256
from src.blockchain import BatchBlockchain


class QRBlockchainAuth:
    """
    Complete QR + Blockchain authentication system
    Based on supervisor's comments:
    - Batch documents (2000 per block)
    - Dual hashing
    - Blockchain verification
    """
    
    def __init__(self, private_key_path='keys/private.pem', 
                 public_key_path='keys/public.pem',
                 batch_size=2000):
        self.private_key = RSA.import_key(open(private_key_path).read())
        self.public_key = RSA.import_key(open(public_key_path).read())
        self.blockchain = BatchBlockchain(batch_size=batch_size)
    
    # =========================================================================
    # HASH GENERATION
    # =========================================================================
    
    def generate_content_hash(self, content):
        """Generate hash of document content ONLY"""
        return hashlib.sha256(content.encode()).hexdigest()
    
    def generate_metadata_hash(self, timestamp, issuer, document_id):
        """Generate hash of metadata ONLY"""
        metadata = {
            "timestamp": timestamp,
            "issuer": issuer,
            "document_id": document_id
        }
        metadata_string = json.dumps(metadata, sort_keys=True)
        return hashlib.sha256(metadata_string.encode()).hexdigest()
    
    # =========================================================================
    # DIGITAL SIGNATURE
    # =========================================================================
    
    def sign_hashes(self, content_hash, metadata_hash):
        """Create digital signature of both hashes"""
        combined = content_hash + metadata_hash
        h = SHA256.new(combined.encode())
        signature = pkcs1_15.new(self.private_key).sign(h)
        return signature.hex()
    
    def verify_signature(self, content_hash, metadata_hash, signature_hex):
        """Verify digital signature"""
        try:
            combined = content_hash + metadata_hash
            h = SHA256.new(combined.encode())
            signature = bytes.fromhex(signature_hex)
            pkcs1_15.new(self.public_key).verify(h, signature)
            return True
        except (ValueError, TypeError):
            return False
    
    # =========================================================================
    # QR CODE GENERATION
    # =========================================================================
    
    def create_document_qr(self, content, issuer, document_id):
        """
        Create QR code data for a document
        Also adds to blockchain batch
        """
        # 1. Generate timestamp
        timestamp = datetime.now().isoformat()
        
        # 2. Generate content hash (NO timestamp!)
        content_hash = self.generate_content_hash(content)
        
        # 3. Generate metadata hash
        metadata_hash = self.generate_metadata_hash(
            timestamp, issuer, document_id
        )
        
        # 4. Sign both hashes
        signature = self.sign_hashes(content_hash, metadata_hash)
        
        # 5. Create QR data
        qr_data = {
            "content_hash": content_hash,
            "metadata_hash": metadata_hash,
            "signature": signature,
            "timestamp": timestamp,
            "issuer": issuer,
            "document_id": document_id
        }
        
        # 6. Add to blockchain (batch)
        self.blockchain.add_document(
            document_hash=content_hash,
            document_type=document_id.split('-')[0] if '-' in document_id else "Document",
            issuer=issuer,
            timestamp=timestamp
        )
        
        return qr_data
    
    # =========================================================================
    # VERIFICATION
    # =========================================================================
    
    def verify_document(self, scanned_content, qr_data):
        """
        Verify a document against its QR code and blockchain
        """
        # 1. Calculate content hash of scanned content
        new_content_hash = self.generate_content_hash(scanned_content)
        
        # 2. Compare content hash
        content_matches = (new_content_hash == qr_data["content_hash"])
        
        # 3. Verify metadata hash
        new_metadata_hash = self.generate_metadata_hash(
            qr_data["timestamp"],
            qr_data["issuer"],
            qr_data["document_id"]
        )
        metadata_matches = (new_metadata_hash == qr_data["metadata_hash"])
        
        # 4. Verify digital signature
        signature_valid = self.verify_signature(
            qr_data["content_hash"],
            qr_data["metadata_hash"],
            qr_data["signature"]
        )
        
        # 5. Check blockchain
        blockchain_result = self.blockchain.verify_document(qr_data["content_hash"])
        
        # 6. Determine result
        if content_matches and signature_valid and blockchain_result["verified"]:
            status = "FULLY_AUTHENTIC"
            message = "✅ Document is fully authentic and verified in blockchain"
        elif content_matches and signature_valid:
            status = "AUTHENTIC_NOT_IN_BLOCKCHAIN"
            message = "⚠️ Document is authentic but not yet in blockchain"
        else:
            status = "TAMPERED"
            message = "❌ Document has been tampered with"
        
        return {
            "status": status,
            "message": message,
            "content_verified": content_matches,
            "metadata_verified": metadata_matches,
            "signature_verified": signature_valid,
            "blockchain_verified": blockchain_result["verified"],
            "blockchain_details": blockchain_result,
            "content_hash_from_qr": qr_data["content_hash"],
            "content_hash_calculated": new_content_hash
        }
    
    def get_blockchain_stats(self):
        """Get blockchain statistics"""
        return self.blockchain.get_stats()