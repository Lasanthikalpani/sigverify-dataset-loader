"""
SigVerify - Blockchain Batch Ledger
Optimized for 2000 documents per block (Supervisor's idea)
"""

import hashlib
import json
from datetime import datetime
import time


class Block:
    """
    A block in the blockchain
    Contains multiple document records (batch of 2000)
    """
    
    def __init__(self, index, transactions, previous_hash, timestamp=None):
        self.index = index
        self.transactions = transactions  # List of 2000 document records
        self.previous_hash = previous_hash
        self.timestamp = timestamp or datetime.now().isoformat()
        self.nonce = 0
        self.merkle_root = self.compute_merkle_root()
        self.hash = self.compute_hash()
    
    def compute_merkle_root(self):
        """Compute Merkle root of all transactions"""
        if not self.transactions:
            return hashlib.sha256("empty".encode()).hexdigest()
        
        # Get hashes of all transactions
        hashes = [
            hashlib.sha256(json.dumps(tx, sort_keys=True).encode()).hexdigest()
            for tx in self.transactions
        ]
        
        # Build Merkle tree
        while len(hashes) > 1:
            if len(hashes) % 2 != 0:
                hashes.append(hashes[-1])
            
            new_hashes = []
            for i in range(0, len(hashes), 2):
                combined = hashes[i] + hashes[i + 1]
                new_hashes.append(hashlib.sha256(combined.encode()).hexdigest())
            hashes = new_hashes
        
        return hashes[0]
    
    def compute_hash(self):
        """Compute block hash"""
        block_data = {
            "index": self.index,
            "merkle_root": self.merkle_root,
            "previous_hash": self.previous_hash,
            "timestamp": self.timestamp,
            "nonce": self.nonce
        }
        block_string = json.dumps(block_data, sort_keys=True)
        return hashlib.sha256(block_string.encode()).hexdigest()
    
    def mine(self, difficulty=2):
        """Simple proof-of-work mining"""
        target = "0" * difficulty
        while self.hash[:difficulty] != target:
            self.nonce += 1
            self.hash = self.compute_hash()
        return self.hash
    
    def to_dict(self):
        return {
            "index": self.index,
            "timestamp": self.timestamp,
            "merkle_root": self.merkle_root,
            "previous_hash": self.previous_hash,
            "nonce": self.nonce,
            "hash": self.hash,
            "transaction_count": len(self.transactions),
            "transactions": self.transactions[:5]  # Show first 5 only
        }


class BatchBlockchain:
    """
    Blockchain with batch processing
    Optimized for 2000 documents per block
    """
    
    def __init__(self, batch_size=2000, difficulty=2):
        self.chain = []
        self.pending_transactions = []
        self.batch_size = batch_size
        self.difficulty = difficulty
        
        # Create genesis block
        self.create_genesis_block()
    
    def create_genesis_block(self):
        """Create the first block"""
        genesis = Block(0, [], "0" * 64)
        genesis.mine(self.difficulty)
        self.chain.append(genesis)
        print(f"✅ Genesis block created: {genesis.hash[:16]}...")
    
    def add_document(self, document_hash, document_type, issuer, timestamp):
        """
        Add a document to pending transactions
        
        Args:
            document_hash: SHA-256 hash of document content
            document_type: Type of document
            issuer: Issuing authority
            timestamp: Issue timestamp
        """
        transaction = {
            "document_hash": document_hash,
            "document_type": document_type,
            "issuer": issuer,
            "timestamp": timestamp,
            "added_at": datetime.now().isoformat()
        }
        self.pending_transactions.append(transaction)
        
        # Check if batch is full
        if len(self.pending_transactions) >= self.batch_size:
            self.mine_block()
    
    def mine_block(self):
        """Mine a new block with pending transactions"""
        if not self.pending_transactions:
            print("⚠️ No pending transactions to mine!")
            return None
        
        # Get previous hash
        previous_hash = self.chain[-1].hash
        
        # Create new block
        new_block = Block(
            index=len(self.chain),
            transactions=self.pending_transactions.copy(),
            previous_hash=previous_hash
        )
        
        # Mine the block
        print(f"⛏️ Mining block {new_block.index} with {len(new_block.transactions)} transactions...")
        start_time = time.time()
        new_block.mine(self.difficulty)
        end_time = time.time()
        
        # Add to chain
        self.chain.append(new_block)
        self.pending_transactions = []
        
        print(f"✅ Block {new_block.index} mined in {end_time - start_time:.2f}s")
        print(f"   Merkle Root: {new_block.merkle_root[:16]}...")
        print(f"   Block Hash: {new_block.hash[:16]}...")
        print(f"   Transactions: {len(new_block.transactions)}")
        
        return new_block
    
    def verify_document(self, document_hash):
        """
        Verify a document is in the blockchain
        
        Returns:
            dict: Verification result
        """
        for block in self.chain:
            for tx in block.transactions:
                if tx["document_hash"] == document_hash:
                    if self.is_chain_valid():
                        return {
                            "verified": True,
                            "block_index": block.index,
                            "merkle_root": block.merkle_root,
                            "timestamp": tx["timestamp"],
                            "issuer": tx["issuer"],
                            "document_type": tx["document_type"],
                            "message": "✅ Document verified in blockchain"
                        }
        
        return {
            "verified": False,
            "message": "❌ Document not found in blockchain"
        }
    
    def is_chain_valid(self):
        """Verify the entire blockchain is valid"""
        for i in range(1, len(self.chain)):
            current = self.chain[i]
            previous = self.chain[i - 1]
            
            if current.hash != current.compute_hash():
                return False
            
            if current.previous_hash != previous.hash:
                return False
        
        return True
    
    def get_stats(self):
        """Get blockchain statistics"""
        total_docs = sum(len(block.transactions) for block in self.chain)
        return {
            "total_blocks": len(self.chain),
            "total_documents": total_docs,
            "batch_size": self.batch_size,
            "pending_documents": len(self.pending_transactions),
            "chain_valid": self.is_chain_valid(),
            "storage_saved": f"{(1 - len(self.chain)/max(total_docs, 1)) * 100:.2f}%"
        }
    
    def get_block(self, index):
        """Get a specific block"""
        if 0 <= index < len(self.chain):
            return self.chain[index].to_dict()
        return None