"""
SigVerify - Tampered Document Generator
Creates realistic tampered documents for testing
Based on latest research (2020-2026)
"""

import hashlib
import random
import re
import json
import os
from PIL import Image, ImageDraw, ImageFont


class TamperGenerator:
    """
    Generate realistic tampered documents for testing
    """
    
    def __init__(self, output_dir='results/tampered'):
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Tampering strategies
        self.strategies = [
            'change_date',
            'change_place',
            'change_name',
            'change_id',
            'swap_fields',
            'multiple_changes'
        ]
    
    # =========================================================================
    # TEXT-BASED TAMPERING
    # =========================================================================
    
    def tamper_content(self, content, strategy='change_date'):
        """
        Create tampered content based on strategy
        
        Args:
            content: Original document content
            strategy: Tampering strategy
        
        Returns:
            str: Tampered content
        """
        original = content  # Keep original for safety check
        
        # =====================================================================
        # STRATEGY: CHANGE DATE
        # =====================================================================
        if strategy == 'change_date':
            match = re.search(r'DOB: (\d{4})', content)
            if match:
                current_year = int(match.group(1))
                possible_years = [y for y in range(1980, 2005) if y != current_year]
                new_year = random.choice(possible_years)
                content = re.sub(r'DOB: \d{4}', f'DOB: {new_year}', content)
        
        # =====================================================================
        # STRATEGY: CHANGE PLACE
        # =====================================================================
        elif strategy == 'change_place':
            places = ['Kandy', 'Galle', 'Jaffna', 'Negombo', 'Matara', 'Anuradhapura']
            match = re.search(r'Place: (\w+)', content)
            if match:
                current_place = match.group(1)
                available_places = [p for p in places if p != current_place]
                new_place = random.choice(available_places)
                content = re.sub(r'Place: \w+', f'Place: {new_place}', content)
        
        # =====================================================================
        # STRATEGY: CHANGE NAME
        # =====================================================================
        elif strategy == 'change_name':
            match = re.search(r'Person (\d+)', content)
            if match:
                current_num = int(match.group(1))
                new_num = current_num + 900
                content = re.sub(r'Person \d+', f'Person {new_num}', content)
        
        # =====================================================================
        # STRATEGY: CHANGE ID
        # =====================================================================
        elif strategy == 'change_id':
            match = re.search(r'BC-\d{4}-\d{6}', content)
            if match:
                content = re.sub(r'BC-\d{4}-\d{6}', 'BC-2026-999999', content)
            else:
                content = content + " BC-2026-999999"
        
        # =====================================================================
        # STRATEGY: SWAP FIELDS
        # =====================================================================
        elif strategy == 'swap_fields':
            content = content.replace('DOB:', 'TEMP:')
            content = content.replace('Place:', 'DOB:')
            content = content.replace('TEMP:', 'Place:')
        
        # =====================================================================
        # STRATEGY: MULTIPLE CHANGES
        # =====================================================================
        elif strategy == 'multiple_changes':
            match = re.search(r'DOB: (\d{4})', content)
            if match:
                current_year = int(match.group(1))
                new_year = current_year + 5 if current_year < 1995 else current_year - 5
                content = re.sub(r'DOB: \d{4}', f'DOB: {new_year}', content)
            
            match = re.search(r'Place: (\w+)', content)
            if match:
                current_place = match.group(1)
                new_place = 'Kandy' if current_place != 'Kandy' else 'Galle'
                content = re.sub(r'Place: \w+', f'Place: {new_place}', content)
        
        # =====================================================================
        # SAFETY CHECK
        # =====================================================================
        if content == original:
            content = original + " [TAMPERED]"
        
        return content
    
    def generate_tampered_documents(self, original_documents, num_tampered=100):
        """
        Generate multiple tampered documents
        """
        tampered_docs = []
        
        for i in range(num_tampered):
            original = random.choice(original_documents)
            strategy = random.choice(self.strategies)
            tampered_content = self.tamper_content(original['content'], strategy)
            
            tampered_docs.append({
                'original_content': original['content'],
                'tampered_content': tampered_content,
                'strategy': strategy,
                'original_id': original.get('document_id', 'unknown'),
                'original_qr': original['qr_data']
            })
        
        return tampered_docs
    
    # =========================================================================
    # SAVE TO FILES
    # =========================================================================
    
    def save_tampered_to_files(self, tampered_docs):
        """
        Save tampered documents to JSON and text files
        """
        os.makedirs(self.output_dir, exist_ok=True)
        
        # Save JSON
        json_file = f"{self.output_dir}/tampered_documents.json"
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(tampered_docs, f, indent=2, ensure_ascii=False, default=str)
        print(f"   ✅ Saved JSON: {json_file}")
        
        # Save summary text
        summary_file = f"{self.output_dir}/tampered_summary.txt"
        with open(summary_file, 'w', encoding='utf-8') as f:
            f.write("=" * 70 + "\n")
            f.write("TAMPERED DOCUMENTS SUMMARY\n")
            f.write("=" * 70 + "\n\n")
            
            for i, doc in enumerate(tampered_docs, 1):
                f.write(f"Tampered Document #{i}\n")
                f.write("-" * 50 + "\n")
                f.write(f"  Strategy:         {doc['strategy']}\n")
                f.write(f"  Original ID:      {doc['original_id']}\n")
                f.write(f"  Original Content: {doc['original_content']}\n")
                f.write(f"  Tampered Content: {doc['tampered_content']}\n")
                f.write(f"  Original QR Hash: {doc['original_qr']['content_hash'][:32]}...\n")
                f.write("\n")
        print(f"   ✅ Saved Summary: {summary_file}")
        
        return json_file, summary_file
    
    # =========================================================================
    # IMAGE-BASED TAMPERING
    # =========================================================================
    
    def create_tampered_image(self, original_cert_path, tampered_content, output_name):
        """Create tampered image version of certificate"""
        original = Image.open(original_cert_path)
        tampered = original.copy()
        draw = ImageDraw.Draw(tampered)
        
        try:
            font = ImageFont.truetype("arial.ttf", 18)
        except:
            font = ImageFont.load_default()
        
        draw.rectangle([(80, 270), (600, 500)], fill='white')
        
        lines = tampered_content.split(', ')
        y_position = 280
        for line in lines:
            draw.text((80, y_position), line, fill='black', font=font)
            y_position += 40
        
        output_path = f"{self.output_dir}/{output_name}"
        tampered.save(output_path)
        return output_path
    
    # =========================================================================
    # VERIFICATION
    # =========================================================================
    
    def verify_tampered(self, auth, tampered_docs):
        """Verify tampered documents and check detection rate"""
        results = []
        
        for doc in tampered_docs:
            result = auth.verify_document(
                doc['tampered_content'],
                doc['original_qr']
            )
            
            results.append({
                'strategy': doc['strategy'],
                'status': result['status'],
                'detected': result['status'] == 'TAMPERED',
                'content_verified': result['content_verified']
            })
        
        return results


# =============================================================================
# DEMO: Create, Save, and Verify Tampered Documents
# =============================================================================

if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    
    from qr_blockchain import QRBlockchainAuth
    
    print("=" * 70)
    print("🔍 TAMPERED DOCUMENT GENERATOR DEMO")
    print("=" * 70)
    
    # Initialize
    auth = QRBlockchainAuth(batch_size=2000)
    tamperer = TamperGenerator()
    
    # =========================================================================
    # STEP 1: Generate Original Documents
    # =========================================================================
    print("\n📄 Generating 100 original documents...")
    originals = []
    for i in range(100):
        document_id = f"BC-2026-{i+1:06d}"
        content = f"Birth Certificate: {document_id}, Person {i+1}, DOB: 199{i%10}-0{(i%9)+1}-15, Place: Colombo"
        
        qr_data = auth.create_document_qr(content, "Registrar General's Office", document_id)
        
        originals.append({
            'content': content,
            'document_id': document_id,
            'qr_data': qr_data
        })
    
    # =========================================================================
    # STEP 2: Generate Tampered Documents
    # =========================================================================
    print("\n⚠️ Generating 100 tampered documents...")
    tampered_docs = tamperer.generate_tampered_documents(originals, num_tampered=100)
    
    # =========================================================================
    # STEP 3: Save to Files
    # =========================================================================
    print("\n💾 Saving tampered documents to files...")
    json_file, summary_file = tamperer.save_tampered_to_files(tampered_docs)
    
    # =========================================================================
    # STEP 4: Verify Tampered Documents
    # =========================================================================
    print("\n🔍 Verifying tampered documents...")
    results = tamperer.verify_tampered(auth, tampered_docs)
    
    # Statistics
    detected = sum(1 for r in results if r['detected'])
    total = len(results)
    
    print(f"\n" + "=" * 70)
    print("📊 TAMPER DETECTION STATISTICS")
    print("=" * 70)
    print(f"Total tampered documents: {total}")
    print(f"Detected as tampered: {detected}")
    print(f"Detection rate: {(detected/total)*100:.1f}%")
    
    # Strategy breakdown
    print("\n📋 Detection by Strategy:")
    strategies = {}
    for r in results:
        s = r['strategy']
        if s not in strategies:
            strategies[s] = {'total': 0, 'detected': 0}
        strategies[s]['total'] += 1
        if r['detected']:
            strategies[s]['detected'] += 1
    
    for strategy, stats in strategies.items():
        rate = (stats['detected'] / stats['total']) * 100
        print(f"   {strategy}: {stats['detected']}/{stats['total']} ({rate:.1f}%)")
    
    # =========================================================================
    # STEP 5: Show First 5 Tampered Documents
    # =========================================================================
    print("\n" + "=" * 70)
    print("📄 FIRST 5 TAMPERED DOCUMENTS")
    print("=" * 70)
    
    for i, doc in enumerate(tampered_docs[:5], 1):
        print(f"\nTampered Document #{i}")
        print("-" * 50)
        print(f"  Strategy: {doc['strategy']}")
        print(f"  Original:  {doc['original_content']}")
        print(f"  Tampered:  {doc['tampered_content']}")
    
    # =========================================================================
    # SUMMARY
    # =========================================================================
    print("\n" + "=" * 70)
    print("🎉 TAMPER DETECTION TEST COMPLETE!")
    print("=" * 70)
    print(f"\n📁 Files saved in: {tamperer.output_dir}/")
    print(f"   - {json_file}")
    print(f"   - {summary_file}")
    print(f"\n✅ Detection rate: {(detected/total)*100:.1f}%")
    print("📌 Supervisor's idea successfully implemented!")