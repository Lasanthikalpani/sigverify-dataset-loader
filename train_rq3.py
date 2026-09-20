"""
RQ3 Training - Building the Blockchain Ledger
"""

import os
import sys
import hashlib
import json
from datetime import datetime

sys.path.insert(0, 'src')

from qr_blockchain import QRBlockchainAuth


def train_rq3():
    """
    Train RQ3 = Build the blockchain ledger
    """
    
    print("=" * 70)
    print("🎓 RQ3 TRAINING - BUILDING BLOCKCHAIN LEDGER")
    print("=" * 70)
    
    # Initialize
    auth = QRBlockchainAuth(batch_size=2000)
    
    # =========================================================================
    # STEP 1: Generate 2000 Documents
    # =========================================================================
    print("\n📄 Step 1: Generating 2000 documents...")
    
    documents = []
    
    for i in range(2000):
        document_id = f"BC-2026-{i+1:06d}"
        content = (
            f"Birth Certificate: {document_id}, "
            f"Person {i+1}, "
            f"DOB: 199{i%10}-0{(i%9)+1}-15, "
            f"Place: Colombo"
        )
        
        documents.append({
            'document_id': document_id,
            'content': content
        })
        
        if (i + 1) % 500 == 0:
            print(f"   ✅ Generated {i+1}/2000 documents")
    
    # =========================================================================
    # STEP 2: Create Transactions (Hash + QR)
    # =========================================================================
    print("\n🔐 Step 2: Creating transactions (hashing + QR)...")
    
    transactions = []
    
    for i, doc in enumerate(documents):
        qr_data = auth.create_document_qr(
            content=doc['content'],
            issuer="Registrar General's Office",
            document_id=doc['document_id']
        )
        
        transaction = {
            'document_id': doc['document_id'],
            'content_hash': qr_data['content_hash'],
            'metadata_hash': qr_data['metadata_hash'],
            'signature': qr_data['signature'],
            'timestamp': qr_data['timestamp']
        }
        
        transactions.append(transaction)
        
        if (i + 1) % 500 == 0:
            print(f"   ✅ Created {i+1}/2000 transactions")
    
    # =========================================================================
    # STEP 3: Check Blockchain State
    # =========================================================================
    print("\n📦 Step 3: Checking blockchain state...")
    
    stats = auth.get_blockchain_stats()
    
    print(f"\n   Blockchain Statistics:")
    print(f"   • Total blocks: {stats['total_blocks']}")
    print(f"   • Total documents: {stats['total_documents']}")
    print(f"   • Batch size: {stats['batch_size']}")
    print(f"   • Pending: {stats['pending_documents']}")
    print(f"   • Chain valid: {stats['chain_valid']}")
    
    # =========================================================================
    # STEP 4: Save Results
    # =========================================================================
    print("\n💾 Step 4: Saving training results...")
    
    os.makedirs('results/training', exist_ok=True)
    
    # Save transactions
    output_file = 'results/training/rq3_training.json'
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump({
            'total_documents': len(documents),
            'total_transactions': len(transactions),
            'blockchain_stats': stats,
            'sample_transactions': transactions[:5]
        }, f, indent=2, ensure_ascii=False)
    
    print(f"   ✅ Saved: {output_file}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎉 RQ3 TRAINING COMPLETE!")
    print("=" * 70)
    print(f"✅ Documents generated: 2000")
    print(f"✅ Transactions created: 2000")
    print(f"✅ Blocks mined: {stats['total_blocks']}")
    print(f"✅ Blockchain valid: {stats['chain_valid']}")
    print(f"✅ Storage reduction: {stats['storage_saved']}")
    print("=" * 70)


if __name__ == "__main__":
    train_rq3()