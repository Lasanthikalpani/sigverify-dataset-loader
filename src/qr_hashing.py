"""
QR + Hashing Module for SigVerify
RQ3: Hybrid Cryptographic + AI Authentication
"""

import hashlib
import json
import os
from pathlib import Path
from datetime import datetime
import qrcode
from PIL import Image
from Crypto.PublicKey import RSA
from Crypto.Signature import pkcs1_15
from Crypto.Hash import SHA256


class DocumentAuthenticator:
    """
    Hybrid document authentication using QR + hashing
    """
    
    def __init__(self, key_dir='keys'):
        """
        Initialize authenticator
        
        Args:
            key_dir: Directory to store RSA keys
        """
        self.key_dir = Path(key_dir)
        self.key_dir.mkdir(exist_ok=True)
        
        self.private_key = None
        self.public_key = None
        
        # Load or generate keys
        self._load_or_generate_keys()
    
    def _load_or_generate_keys(self):
        """Load existing keys or generate new ones"""
        private_key_path = self.key_dir / 'private.pem'
        public_key_path = self.key_dir / 'public.pem'
        
        if private_key_path.exists() and public_key_path.exists():
            # Load existing keys
            with open(private_key_path, 'rb') as f:
                self.private_key = RSA.import_key(f.read())
            with open(public_key_path, 'rb') as f:
                self.public_key = RSA.import_key(f.read())
            print("✅ RSA keys loaded")
        else:
            # Generate new keys
            self.private_key = RSA.generate(2048)
            self.public_key = self.private_key.publickey()
            
            # Save keys
            with open(private_key_path, 'wb') as f:
                f.write(self.private_key.export_key())
            with open(public_key_path, 'wb') as f:
                f.write(self.public_key.export_key())
            print("✅ RSA keys generated and saved")
    
    def generate_document_hash(self, document_data):
        """
        Generate SHA-256 hash of document
        
        Args:
            document_data: Dictionary with document information
            
        Returns:
            str: SHA-256 hash
        """
        # Convert to JSON string (sorted for consistency)
        document_json = json.dumps(document_data, sort_keys=True)
        
        # Generate SHA-256 hash
        hash_object = hashlib.sha256(document_json.encode())
        return hash_object.hexdigest()
    
    def sign_hash(self, hash_value):
        """
        Sign hash with private key
        
        Args:
            hash_value: SHA-256 hash string
            
        Returns:
            bytes: Digital signature
        """
        hash_obj = SHA256.new(hash_value.encode())
        signature = pkcs1_15.new(self.private_key).sign(hash_obj)
        return signature
    
    def verify_signature(self, hash_value, signature):
        """
        Verify signature with public key
        
        Args:
            hash_value: SHA-256 hash string
            signature: Digital signature bytes
            
        Returns:
            bool: True if valid
        """
        try:
            hash_obj = SHA256.new(hash_value.encode())
            pkcs1_15.new(self.public_key).verify(hash_obj, signature)
            return True
        except (ValueError, TypeError):
            return False
    
    def create_qr_payload(self, document_data):
        """
        Create QR payload with hash and signature
        
        Args:
            document_data: Dictionary with document information
            
        Returns:
            str: QR payload string
        """
        # Generate hash
        doc_hash = self.generate_document_hash(document_data)
        
        # Sign hash
        signature = self.sign_hash(doc_hash)
        
        # Create payload
        payload = {
            'hash': doc_hash,
            'signature': signature.hex(),
            'timestamp': datetime.now().isoformat(),
            'version': '1.0'
        }
        
        return json.dumps(payload)
    
    def generate_qr_code(self, document_data, output_path):
        """
        Generate QR code with document hash and signature
        
        Args:
            document_data: Dictionary with document information
            output_path: Path to save QR code image
            
        Returns:
            str: Path to saved QR code
        """
        # Create payload
        payload = self.create_qr_payload(document_data)
        
        # Generate QR code
        qr = qrcode.QRCode(
            version=1,
            error_correction=qrcode.constants.ERROR_CORRECT_H,
            box_size=10,
            border=4
        )
        qr.add_data(payload)
        qr.make(fit=True)
        
        # Create image
        img = qr.make_image(fill_color="black", back_color="white")
        
        # Save
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        img.save(output_path)
        
        print(f"✅ QR code saved: {output_path}")
        return str(output_path)
    
    def verify_document(self, document_data, qr_payload):
        """
        Verify document against QR payload
        
        Args:
            document_data: Current document data
            qr_payload: QR payload string
            
        Returns:
            dict: Verification result
        """
        result = {
            'is_authentic': False,
            'hash_match': False,
            'signature_valid': False,
            'message': ''
        }
        
        try:
            # Parse payload
            payload = json.loads(qr_payload)
            
            # Generate hash of current document
            current_hash = self.generate_document_hash(document_data)
            
            # Check hash match
            result['hash_match'] = (current_hash == payload['hash'])
            
            if not result['hash_match']:
                result['message'] = '❌ Document has been TAMPERED with!'
                return result
            
            # Verify signature
            signature = bytes.fromhex(payload['signature'])
            result['signature_valid'] = self.verify_signature(payload['hash'], signature)
            
            if not result['signature_valid']:
                result['message'] = '❌ Invalid signature!'
                return result
            
            # Both checks passed
            result['is_authentic'] = True
            result['message'] = '✅ Document is AUTHENTIC!'
            
        except Exception as e:
            result['message'] = f'❌ Verification error: {str(e)}'
        
        return result


# Test function
if __name__ == "__main__":
    print("=" * 70)
    print("🔐 SIGVERIFY - QR + HASHING TEST")
    print("=" * 70)
    
    # Initialize authenticator
    auth = DocumentAuthenticator()
    
    # Sample document
    document = {
        'type': 'Land Deed',
        'owner': 'S.K.K.K.L. Chandrasekara',
        'property': '10 perches, Matara',
        'issued_date': '2026-09-15',
        'registration_number': 'LD-2026-001234'
    }
    
    print("\n📄 Original Document:")
    for key, value in document.items():
        print(f"   {key}: {value}")
    
    # Generate QR
    qr_path = auth.generate_qr_code(document, 'results/qr_codes/document_qr.png')
    
    # Simulate verification
    print("\n📋 Verification Test 1: Original Document")
    payload = auth.create_qr_payload(document)
    result = auth.verify_document(document, payload)
    print(f"   {result['message']}")
    print(f"   Hash Match: {result['hash_match']}")
    print(f"   Signature Valid: {result['signature_valid']}")
    
    # Simulate tampering
    print("\n📋 Verification Test 2: Tampered Document")
    tampered = document.copy()
    tampered['property'] = '100 perches, Matara'  # Changed!
    result = auth.verify_document(tampered, payload)
    print(f"   {result['message']}")
    print(f"   Hash Match: {result['hash_match']}")
    print(f"   Signature Valid: {result['signature_valid']}")
    
    print("\n" + "=" * 70)
    print("🎉 TEST COMPLETE!")
    print("=" * 70)