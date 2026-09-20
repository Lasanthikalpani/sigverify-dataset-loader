"""
RQ3 Training - Enhanced Version
Builds blockchain ledger with 5000 documents
"""

import os
import sys
import json
import time
import random
import hashlib
from datetime import datetime

sys.path.insert(0, 'src')
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from qr_blockchain import QRBlockchainAuth


class EnhancedRQ3Trainer:
    """
    Enhanced RQ3 Trainer
    - 5000 documents
    - 4 document types
    - Advanced metrics
    - Visualization
    """
    
    def __init__(self, batch_size=2000, output_dir='results/training_enhanced'):
        self.batch_size = batch_size
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        self.auth = QRBlockchainAuth(batch_size=batch_size)
        
        # Document type distribution
        self.document_types = {
            'birth_certificate': {
                'count': 2000,
                'prefix': 'BC',
                'issuer': "Registrar General's Office"
            },
            'land_deed': {
                'count': 1000,
                'prefix': 'LD',
                'issuer': "Land Registry"
            },
            'id_card': {
                'count': 1000,
                'prefix': 'ID',
                'issuer': "Department of Registration of Persons"
            },
            'educational_certificate': {
                'count': 1000,
                'prefix': 'EC',
                'issuer': "University Grants Commission"
            }
        }
        
        # Random data
        self.names = [
            "S.K.K.K.L. Chandrasekara", "A.B. Perera", "C.D. Silva",
            "E.F. Fernando", "G.H. Jayawardena", "I.J. Wickramasinghe",
            "K.L. Gunasekara", "M.N. Rajapaksa", "O.P. Bandara"
        ]
        self.places = ["Colombo", "Kandy", "Galle", "Jaffna", "Negombo", "Matara"]
    
    def generate_content(self, doc_type, doc_id):
        """Generate content based on document type"""
        
        name = random.choice(self.names)
        place = random.choice(self.places)
        year = random.randint(1980, 2005)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        dob = f"{year}-{month:02d}-{day:02d}"
        
        if doc_type == 'birth_certificate':
            return f"Birth Certificate: {doc_id}, Name: {name}, DOB: {dob}, Place: {place}"
        
        elif doc_type == 'land_deed':
            perches = random.choice([5, 10, 15, 20, 50, 100])
            return f"Land Deed: {doc_id}, Owner: {name}, Property: {perches} perches, Location: {place}"
        
        elif doc_type == 'id_card':
            return f"National ID: {doc_id}, Name: {name}, DOB: {dob}, Address: {place}"
        
        else:  # educational_certificate
            degree = random.choice(["BSc", "BA", "BCom", "BBA", "MSc", "MA"])
            university = random.choice(["Colombo", "Peradeniya", "Jayewardenepura", "Kelaniya", "Moratuwa"])
            return f"Educational Certificate: {doc_id}, Name: {name}, Degree: {degree}, University: {university}"
    
    def train(self):
        """Run the enhanced training"""
        
        print("=" * 70)
        print("🎓 RQ3 ENHANCED TRAINING - 5000 DOCUMENTS")
        print("=" * 70)
        
        all_documents = []
        start_time = time.time()
        
        # =====================================================================
        # Generate documents for each type
        # =====================================================================
        print("\n📄 Step 1: Generating documents...")
        
        for doc_type, config in self.document_types.items():
            print(f"\n   Generating {config['count']} {doc_type}...")
            
            for i in range(config['count']):
                doc_id = f"{config['prefix']}-2026-{i+1:06d}"
                content = self.generate_content(doc_type, doc_id)
                
                qr_data = self.auth.create_document_qr(
                    content=content,
                    issuer=config['issuer'],
                    document_id=doc_id
                )
                
                all_documents.append({
                    'document_id': doc_id,
                    'document_type': doc_type,
                    'content': content,
                    'qr_data': qr_data,
                    'source': 'synthetic'
                })
                
                if (i + 1) % 500 == 0:
                    print(f"      ✅ {i+1}/{config['count']}")
        
        # Mine remaining transactions
        if self.auth.blockchain.pending_transactions:
            self.auth.blockchain.mine_block()
        
        end_time = time.time()
        training_time = end_time - start_time
        
        # =====================================================================
        # Statistics
        # =====================================================================
        print("\n" + "=" * 70)
        print("📊 TRAINING STATISTICS")
        print("=" * 70)
        
        stats = self.auth.get_blockchain_stats()
        
        print(f"\n   Documents: {len(all_documents)}")
        print(f"   Total blocks: {stats['total_blocks']}")
        print(f"   Batch size: {stats['batch_size']}")
        print(f"   Chain valid: {stats['chain_valid']}")
        print(f"   Storage saved: {stats['storage_saved']}")
        print(f"   Training time: {training_time:.2f} seconds")
        
        # By document type
        print("\n   By document type:")
        type_counts = {}
        for doc in all_documents:
            t = doc['document_type']
            type_counts[t] = type_counts.get(t, 0) + 1
        for t, c in type_counts.items():
            print(f"      {t}: {c}")
        
        # =====================================================================
        # Save results
        # =====================================================================
        print("\n💾 Step 2: Saving results...")
        
        # Save documents
        with open(f'{self.output_dir}/all_documents.json', 'w', encoding='utf-8') as f:
            json.dump(all_documents, f, indent=2, ensure_ascii=False, default=str)
        print(f"   ✅ Documents saved")
        
        # Save summary
        with open(f'{self.output_dir}/summary.txt', 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("RQ3 ENHANCED TRAINING SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            f.write(f"Total documents: {len(all_documents)}\n")
            f.write(f"Total blocks: {stats['total_blocks']}\n")
            f.write(f"Batch size: {stats['batch_size']}\n")
            f.write(f"Storage saved: {stats['storage_saved']}\n")
            f.write(f"Training time: {training_time:.2f}s\n\n")
            f.write("By document type:\n")
            for t, c in type_counts.items():
                f.write(f"  {t}: {c}\n")
        print(f"   ✅ Summary saved")
        
        # =====================================================================
        # Performance Metrics
        # =====================================================================
        print("\n📈 Performance Metrics:")
        print(f"   Documents per second: {len(all_documents)/training_time:.0f}")
        print(f"   Documents per block: {len(all_documents)/stats['total_blocks']:.0f}")
        print(f"   Storage per document: {100/len(all_documents):.4f}%")
        
        print("\n" + "=" * 70)
        print("🎉 ENHANCED TRAINING COMPLETE!")
        print("=" * 70)
        
        return all_documents, stats


if __name__ == "__main__":
    trainer = EnhancedRQ3Trainer(batch_size=2000)
    documents, stats = trainer.train()