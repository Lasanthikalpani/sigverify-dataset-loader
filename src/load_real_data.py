"""
SigVerify - Load Real Data from CSV
"""

import os
import csv
import json
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_blockchain import QRBlockchainAuth


class RealDataLoader:
    def __init__(self, auth, output_dir='results/real_data'):
        self.auth = auth
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
    
    def load_from_csv(self, csv_file):
        print(f"📂 Loading: {csv_file}")
        
        if not os.path.exists(csv_file):
            print(f"❌ Not found: {csv_file}")
            return []
        
        documents = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                document_id = row.get('document_id', f'DOC-{i+1:04d}')
                
                content = (
                    f"Birth Certificate: {document_id}, "
                    f"Name: {row.get('name', 'Unknown')}, "
                    f"DOB: {row.get('dob', 'Unknown')}, "
                    f"Place: {row.get('place', 'Unknown')}"
                )
                
                qr_data = self.auth.create_document_qr(
                    content=content,
                    issuer="Registrar General's Office",
                    document_id=document_id
                )
                
                documents.append({
                    'document_id': document_id,
                    'content': content,
                    'qr_data': qr_data,
                    'source': 'real'
                })
        
        print(f"✅ Loaded: {len(documents)}")
        return documents
    
    def save_documents(self, documents, output_file='real_documents.json'):
        output_path = os.path.join(self.output_dir, output_file)
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(documents, f, indent=2, ensure_ascii=False, default=str)
        print(f"✅ Saved: {output_path}")


if __name__ == "__main__":
    auth = QRBlockchainAuth(batch_size=2000)
    loader = RealDataLoader(auth)
    
    # Create sample CSV
    os.makedirs('results/real_data', exist_ok=True)
    csv_file = 'results/real_data/sample_birth_certificates.csv'
    
    sample_data = [
        {'document_id': 'BC-REAL-0001', 'name': 'S.K.K.K.L. Chandrasekara', 
         'dob': '1990-01-15', 'place': 'Colombo'},
        {'document_id': 'BC-REAL-0002', 'name': 'A.B. Perera',
         'dob': '1985-05-20', 'place': 'Kandy'}
    ]
    
    with open(csv_file, 'w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=sample_data[0].keys())
        writer.writeheader()
        writer.writerows(sample_data)
    
    print(f"✅ Sample CSV: {csv_file}")
    
    # Load and save
    documents = loader.load_from_csv(csv_file)
    if documents:
        loader.save_documents(documents)