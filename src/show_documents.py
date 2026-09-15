"""
Show and save 100 original documents
"""

import sys
import os
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from qr_blockchain import QRBlockchainAuth


def show_and_save_documents():
    """Generate and save 100 original documents"""
    
    print("=" * 70)
    print("📄 GENERATING 100 ORIGINAL DOCUMENTS")
    print("=" * 70)
    
    # Initialize
    auth = QRBlockchainAuth(batch_size=2000)
    
    # Create output directory
    os.makedirs('results/original_documents', exist_ok=True)
    
    # Generate 100 documents
    documents = []
    
    for i in range(100):
        document_id = f"BC-2026-{i+1:06d}"
        content = f"Birth Certificate: {document_id}, Person {i+1}, DOB: 199{i%10}-0{(i%9)+1}-15, Place: Colombo"
        
        qr_data = auth.create_document_qr(content, "Registrar General's Office", document_id)
        
        documents.append({
            'number': i + 1,
            'document_id': document_id,
            'content': content,
            'qr_data': qr_data
        })
        
        # Print every 10th document
        if (i + 1) % 10 == 0:
            print(f"   ✅ Generated {i+1}/100 documents")
    
    # Save to JSON file
    output_file = 'results/original_documents/all_documents.json'
    
    with open(output_file, 'w', encoding='utf-8') as f:

        json.dump(documents, f, indent=2)
    
    print(f"\n✅ All 100 documents saved to: {output_file}")
    
    # Save a summary text file
    summary_file = 'results/original_documents/summary.txt'
    
    with open(summary_file, 'w', encoding='utf-8') as f:

        f.write("=" * 70 + "\n")
        f.write("📄 ORIGINAL DOCUMENTS SUMMARY\n")
        f.write("=" * 70 + "\n\n")
        
        for doc in documents:
            f.write(f"Document #{doc['number']}\n")
            f.write("-" * 50 + "\n")
            f.write(f"  Document ID: {doc['document_id']}\n")
            f.write(f"  Content: {doc['content']}\n")
            f.write(f"  Content Hash: {doc['qr_data']['content_hash']}\n")
            f.write(f"  Metadata Hash: {doc['qr_data']['metadata_hash']}\n")
            f.write(f"  Signature: {doc['qr_data']['signature'][:64]}...\n")
            f.write(f"  Timestamp: {doc['qr_data']['timestamp']}\n")
            f.write(f"  Issuer: {doc['qr_data']['issuer']}\n")
            f.write("\n")
    
    print(f"✅ Summary saved to: {summary_file}")
    
    # Show first 5 documents in console
    print("\n" + "=" * 70)
    print("📄 FIRST 5 DOCUMENTS")
    print("=" * 70)
    
    for doc in documents[:5]:
        print(f"\nDocument #{doc['number']}")
        print("-" * 50)
        print(f"  Content: {doc['content']}")
        print(f"  Document ID: {doc['document_id']}")
        print(f"  Content Hash: {doc['qr_data']['content_hash'][:32]}...")
        print(f"  Metadata Hash: {doc['qr_data']['metadata_hash'][:32]}...")
        print(f"  Timestamp: {doc['qr_data']['timestamp']}")
    
    print("\n🎉 Done!")
    print(f"\n📁 Files saved in: results/original_documents/")


if __name__ == "__main__":
    show_and_save_documents()