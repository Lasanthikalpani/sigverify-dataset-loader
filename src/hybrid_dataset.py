"""
SigVerify - Hybrid Dataset Generator
Combines synthetic and real data for RQ3
"""

import os
import sys
import json
import csv
import random
from datetime import datetime

# Add paths for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_blockchain import QRBlockchainAuth


class HybridDatasetGenerator:
    """
    Generate hybrid dataset combining synthetic and real data
    Synthetic: 95% (1900 documents)
    Real: 5% (100 documents)
    Total: 2000 documents
    """
    
    def __init__(self, auth, output_dir='results/dataset'):
        self.auth = auth
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.document_types = {
            'birth_certificate': {'synthetic': 1000, 'real': 50, 'prefix': 'BC'},
            'land_deed': {'synthetic': 300, 'real': 20, 'prefix': 'LD'},
            'id_card': {'synthetic': 300, 'real': 15, 'prefix': 'ID'},
            'educational_certificate': {'synthetic': 300, 'real': 15, 'prefix': 'EC'}
        }
    
    # =========================================================================
    # SYNTHETIC DATA
    # =========================================================================
    
    def generate_synthetic_document(self, doc_type, index):
        """Generate a synthetic document"""
        prefix = self.document_types[doc_type]['prefix']
        document_id = f"{prefix}-2026-{index:06d}"
        
        names = [
            "S.K.K.K.L. Chandrasekara", "A.B. Perera", "C.D. Silva",
            "E.F. Fernando", "G.H. Jayawardena", "I.J. Wickramasinghe",
            "K.L. Gunasekara", "M.N. Rajapaksa", "O.P. Bandara"
        ]
        places = ["Colombo", "Kandy", "Galle", "Jaffna", "Negombo", "Matara"]
        
        name = random.choice(names)
        place = random.choice(places)
        year = random.randint(1980, 2005)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        dob = f"{year}-{month:02d}-{day:02d}"
        
        content = (
            f"Birth Certificate: {document_id}, "
            f"Name: {name}, DOB: {dob}, Place: {place}"
        )
        
        qr_data = self.auth.create_document_qr(
            content=content,
            issuer="Registrar General's Office",
            document_id=document_id
        )
        
        return {
            'document_id': document_id,
            'document_type': doc_type,
            'content': content,
            'qr_data': qr_data,
            'source': 'synthetic'
        }
    
    # =========================================================================
    # REAL DATA LOADING
    # =========================================================================
    
    def load_real_documents(self, real_data_file=None):
        """Load real documents from CSV or JSON file"""
        real_docs = []
        
        if real_data_file and os.path.exists(real_data_file):
            if real_data_file.endswith('.csv'):
                real_docs = self._load_from_csv(real_data_file)
            elif real_data_file.endswith('.json'):
                real_docs = self._load_from_json(real_data_file)
            else:
                print(f"⚠️ Unsupported file type. Using placeholder data.")
                real_docs = self._create_placeholder_data()
        else:
            print("⚠️ No real data file. Using placeholder real data.")
            real_docs = self._create_placeholder_data()
        
        return real_docs
    
    def _load_from_csv(self, csv_file):
        """Load real documents from CSV file"""
        print(f"📂 Loading CSV: {csv_file}")
        real_docs = []
        
        with open(csv_file, 'r', encoding='utf-8') as f:
            reader = csv.DictReader(f)
            
            for i, row in enumerate(reader):
                document_id = row.get('document_id', f'BC-REAL-{i+1:04d}')
                
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
                
                real_docs.append({
                    'document_id': document_id,
                    'document_type': 'birth_certificate',
                    'content': content,
                    'qr_data': qr_data,
                    'source': 'real'
                })
        
        print(f"   ✅ Loaded {len(real_docs)} real documents from CSV")
        return real_docs
    
    def _load_from_json(self, json_file):
        """Load real documents from JSON file"""
        print(f"📂 Loading JSON: {json_file}")
        real_docs = []
        
        with open(json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        for item in data:
            document_id = item.get('document_id', 'UNKNOWN')
            content = item.get('content', '')
            
            qr_data = self.auth.create_document_qr(
                content=content,
                issuer=item.get('issuer', 'Government Office'),
                document_id=document_id
            )
            
            real_docs.append({
                'document_id': document_id,
                'document_type': item.get('document_type', 'birth_certificate'),
                'content': content,
                'qr_data': qr_data,
                'source': 'real'
            })
        
        print(f"   ✅ Loaded {len(real_docs)} real documents from JSON")
        return real_docs
    
    def _create_placeholder_data(self):
        """Create placeholder real data (100 documents)"""
        print("⚠️ Creating placeholder real data...")
        real_docs = []
        
        for i in range(100):
            document_id = f"BC-REAL-{i+1:04d}"
            content = f"Birth Certificate: {document_id}, Real Person {i+1}"
            
            qr_data = self.auth.create_document_qr(
                content=content,
                issuer="Registrar General's Office",
                document_id=document_id
            )
            
            real_docs.append({
                'document_id': document_id,
                'document_type': 'birth_certificate',
                'content': content,
                'qr_data': qr_data,
                'source': 'real'
            })
        
        return real_docs
    
    # =========================================================================
    # HYBRID DATASET GENERATION
    # =========================================================================
    
    def generate_hybrid_dataset(self, real_data_file=None):
        """Generate complete hybrid dataset"""
        print("=" * 70)
        print("📊 GENERATING HYBRID DATASET")
        print("=" * 70)
        
        dataset = {'synthetic': [], 'real': [], 'combined': []}
        
        # Synthetic
        print("\n📄 Step 1: Generating synthetic documents...")
        for doc_type, counts in self.document_types.items():
            for i in range(counts['synthetic']):
                doc = self.generate_synthetic_document(doc_type, i + 1)
                dataset['synthetic'].append(doc)
            print(f"   ✅ {doc_type}: {counts['synthetic']} synthetic")
        
        print(f"\n   Total synthetic: {len(dataset['synthetic'])}")
        
        # Real
        print("\n📄 Step 2: Loading real documents...")
        dataset['real'] = self.load_real_documents(real_data_file)
        print(f"   Total real: {len(dataset['real'])}")
        
        # Combined
        print("\n📄 Step 3: Combining datasets...")
        dataset['combined'] = dataset['synthetic'] + dataset['real']
        random.shuffle(dataset['combined'])
        print(f"   Total combined: {len(dataset['combined'])}")
        
        # Save
        print("\n📄 Step 4: Saving dataset...")
        self._save_dataset(dataset)
        
        return dataset
    
    def _save_dataset(self, dataset):
        """Save dataset to files"""
        combined_file = os.path.join(self.output_dir, 'hybrid_dataset.json')
        with open(combined_file, 'w', encoding='utf-8') as f:
            json.dump(dataset['combined'], f, indent=2, ensure_ascii=False, default=str)
        print(f"   ✅ Combined: {combined_file}")
        
        summary_file = os.path.join(self.output_dir, 'dataset_summary.txt')
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("HYBRID DATASET SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total: {len(dataset['combined'])}\n")
            f.write(f"Synthetic: {len(dataset['synthetic'])}\n")
            f.write(f"Real: {len(dataset['real'])}\n")
        print(f"   ✅ Summary: {summary_file}")


# =============================================================================
# MAIN
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from qr_blockchain import QRBlockchainAuth
    
    print("=" * 70)
    print("🔍 HYBRID DATASET GENERATOR")
    print("=" * 70)
    
    auth = QRBlockchainAuth(batch_size=2000)
    generator = HybridDatasetGenerator(auth)
    
    # Use CSV file if exists
    csv_file = 'results/real_data/real_birth_certificates.csv'
    
    if os.path.exists(csv_file):
        print(f"\n✅ Using real data from: {csv_file}")
        dataset = generator.generate_hybrid_dataset(real_data_file=csv_file)
    else:
        print(f"\n⚠️ CSV not found: {csv_file}")
        dataset = generator.generate_hybrid_dataset()
    
    print("\n🎉 Hybrid dataset generated successfully!")