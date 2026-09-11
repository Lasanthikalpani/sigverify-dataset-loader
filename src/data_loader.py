"""
Signature Dataset Loader for SigVerify
Supports CEDAR, GPDS, and Kaggle datasets
"""

import os
import cv2
import numpy as np
import requests
import zipfile
import pickle
from tqdm import tqdm
from sklearn.model_selection import train_test_split
import urllib.request
import shutil
from pathlib import Path


class SignatureDatasetLoader:
    """
    Load and preprocess signature datasets for Siamese networks
    """
    
    def __init__(self, data_dir='data', img_size=(128, 128), verbose=True):
        """
        Initialize the dataset loader
        
        Args:
            data_dir: Root directory for data storage
            img_size: Target image size (height, width)
            verbose: Print progress messages
        """
        self.data_dir = Path(data_dir)
        self.img_size = img_size
        self.verbose = verbose
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self._create_dirs()
        
        # Data storage
        self.genuine_images = []
        self.forged_images = []
        self.genuine_labels = []
        self.forged_labels = []
    
    def _create_dirs(self):
        """Create required directories"""
        for dir_path in [self.raw_dir, self.processed_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)
            (dir_path / '.gitkeep').touch(exist_ok=True)
    
    def _log(self, message):
        """Print message if verbose"""
        if self.verbose:
            print(f"[SigVerify] {message}")
    
    def download_cedar(self, force_download=False):
        """
        Download CEDAR signature dataset
        
        Args:
            force_download: Force download even if exists
        """
        cedar_dir = self.raw_dir / 'cedar'
        
        if cedar_dir.exists() and not force_download:
            self._log(f"CEDAR dataset already exists at {cedar_dir}")
            return
        
        self._log("Downloading CEDAR dataset...")
        
        # Note: You may need to update this URL
        urls = [
            "https://www.dropbox.com/s/xxxxxxxxx/cedar.zip?dl=1",
            "https://www.cedar.buffalo.edu/NIJ/data/signatures.rar",
        ]
        
        for url in urls:
            try:
                zip_path = self.raw_dir / 'cedar.zip'
                
                response = requests.get(url, stream=True)
                total_size = int(response.headers.get('content-length', 0))
                
                with open(zip_path, 'wb') as f:
                    with tqdm(total=total_size, unit='B', unit_scale=True, 
                              desc='Downloading') as pbar:
                        for data in response.iter_content(chunk_size=1024):
                            f.write(data)
                            pbar.update(len(data))
                
                self._log("Extracting CEDAR dataset...")
                with zipfile.ZipFile(zip_path, 'r') as zip_ref:
                    zip_ref.extractall(self.raw_dir)
                
                extracted = self.raw_dir / 'cedar'
                if not extracted.exists():
                    for item in self.raw_dir.iterdir():
                        if item.is_dir() and item.name != 'cedar':
                            if 'cedar' in item.name.lower():
                                item.rename(extracted)
                                break
                
                zip_path.unlink()
                self._log(f"CEDAR dataset downloaded to {cedar_dir}")
                return
                
            except Exception as e:
                self._log(f"Failed to download from {url}: {e}")
                continue
        
        self._log("=" * 60)
        self._log("⚠️  Could not automatically download CEDAR dataset.")
        self._log("Please manually download from:")
        self._log("  - https://www.cedar.buffalo.edu/NIJ/data/signatures.rar")
        self._log("=" * 60)
    
    def load_cedar(self):
        """
        Load CEDAR dataset from disk
        
        Returns:
            tuple: (genuine_images, forged_images)
        """
        self._log("Loading CEDAR dataset...")
        
        cedar_dir = self.raw_dir / 'cedar'
        
        if not cedar_dir.exists():
            self._log(f"CEDAR dataset not found at {cedar_dir}")
            self._log("Please run download_cedar() first or manually place the dataset.")
            return None, None
        
        self.genuine_images = []
        self.forged_images = []
        
        person_dirs = [d for d in cedar_dir.iterdir() if d.is_dir()]
        
        for person_dir in tqdm(person_dirs, desc="Loading signatures"):
            for img_file in person_dir.iterdir():
                if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                    img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                    if img is not None:
                        img = cv2.resize(img, self.img_size)
                        img = img / 255.0
                        
                        if 'F' in img_file.stem:
                            self.forged_images.append(img)
                        else:
                            self.genuine_images.append(img)
        
        self._log(f"Loaded {len(self.genuine_images)} genuine signatures")
        self._log(f"Loaded {len(self.forged_images)} forged signatures")
        
        return np.array(self.genuine_images), np.array(self.forged_images)
    
    def load_gpds(self, gpds_path=None):
        """Load GPDS dataset"""
        if gpds_path:
            gpds_dir = Path(gpds_path)
        else:
            gpds_dir = self.raw_dir / 'gpds'
        
        if not gpds_dir.exists():
            self._log(f"GPDS dataset not found at {gpds_dir}")
            self._log("Please download from: https://gpds.ulpgc.es/")
            return None, None
        
        self.genuine_images = []
        self.forged_images = []
        
        genuine_dir = gpds_dir / 'Reference'
        forged_dir = gpds_dir / 'Forged'
        
        if genuine_dir.exists():
            for img_file in tqdm(list(genuine_dir.glob('*.png')), desc="Loading genuine"):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0
                    self.genuine_images.append(img)
        
        if forged_dir.exists():
            for img_file in tqdm(list(forged_dir.glob('*.png')), desc="Loading forged"):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, self.img_size)
                    img = img / 255.0
                    self.forged_images.append(img)
        
        self._log(f"Loaded {len(self.genuine_images)} genuine signatures")
        self._log(f"Loaded {len(self.forged_images)} forged signatures")
        
        return np.array(self.genuine_images), np.array(self.forged_images)
    
    def load_from_kaggle(self, kaggle_path=None):
        """Load dataset from Kaggle"""
        if kaggle_path:
            dataset_dir = Path(kaggle_path)
        else:
            dataset_dir = self.raw_dir / 'kaggle'
        
        if not dataset_dir.exists():
            self._log(f"Kaggle dataset not found at {dataset_dir}")
            self._log("Please download from Kaggle and place in data/raw/kaggle/")
            return None, None
        
        self.genuine_images = []
        self.forged_images = []
        
        for pattern in ['genuine', 'real', 'original', 'gen']:
            genuine_dir = dataset_dir / pattern
            if genuine_dir.exists():
                for img_file in genuine_dir.glob('*.*'):
                    img = self._load_image(img_file)
                    if img is not None:
                        self.genuine_images.append(img)
        
        for pattern in ['forged', 'fake', 'forge', 'fraud']:
            forged_dir = dataset_dir / pattern
            if forged_dir.exists():
                for img_file in forged_dir.glob('*.*'):
                    img = self._load_image(img_file)
                    if img is not None:
                        self.forged_images.append(img)
        
        self._log(f"Loaded {len(self.genuine_images)} genuine signatures")
        self._log(f"Loaded {len(self.forged_images)} forged signatures")
        
        return np.array(self.genuine_images), np.array(self.forged_images)
    
    def _load_image(self, img_path):
        """Helper function to load and preprocess a single image"""
        img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
        if img is not None:
            img = cv2.resize(img, self.img_size)
            img = img / 255.0
            return img
        return None
    
    def create_pairs(self, genuine=None, forged=None, num_pairs=10000):
        """
        Create training pairs for Siamese network
        """
        if genuine is None:
            genuine = np.array(self.genuine_images)
        if forged is None:
            forged = np.array(self.forged_images)
        
        # Check if data exists
        if len(genuine) == 0 or len(forged) == 0:
            print("⚠️ No data loaded! Please run load_cedar() first.")
            return np.array([]), np.array([]), np.array([])
        
        # Add channel dimension if needed
        if len(genuine.shape) == 3:
            genuine = genuine.reshape(-1, self.img_size[0], self.img_size[1], 1)
            forged = forged.reshape(-1, self.img_size[0], self.img_size[1], 1)
        
        X1 = []
        X2 = []
        y = []
        
        # Positive pairs (genuine vs genuine)
        for _ in range(num_pairs // 2):
            idx1 = np.random.randint(0, len(genuine))
            idx2 = np.random.randint(0, len(genuine))
            X1.append(genuine[idx1])
            X2.append(genuine[idx2])
            y.append(1)
        
        # Negative pairs (genuine vs forged)
        for _ in range(num_pairs // 2):
            idx1 = np.random.randint(0, len(genuine))
            idx2 = np.random.randint(0, len(forged))
            X1.append(genuine[idx1])
            X2.append(forged[idx2])
            y.append(0)
        
        return np.array(X1), np.array(X2), np.array(y)
    
    def save_processed_data(self, data, name='sigverify_data'):
        """Save processed data for later use"""
        save_dir = self.processed_dir / name
        save_dir.mkdir(exist_ok=True)
        
        for key, value in data.items():
            if isinstance(value, np.ndarray):
                np.save(save_dir / f"{key}.npy", value)
        
        self._log(f"Data saved to {save_dir}")
    
    def load_processed_data(self, name='sigverify_data'):
        """Load processed data"""
        data = {}
        load_dir = self.processed_dir / name
        
        if not load_dir.exists():
            self._log(f"No processed data found at {load_dir}")
            return None
        
        for file in load_dir.glob('*.npy'):
            key = file.stem
            data[key] = np.load(file)
        
        self._log(f"Loaded data from {load_dir}")
        return data
    
    def get_train_test_data(self, test_size=0.2, num_pairs=10000):
        """Get train and test data splits"""
        if len(self.genuine_images) == 0:
            self.load_cedar()
        
        X1, X2, y = self.create_pairs(num_pairs=num_pairs)
        
        X1_train, X1_test, X2_train, X2_test, y_train, y_test = train_test_split(
            X1, X2, y, test_size=test_size, random_state=42
        )
        
        self._log(f"Train pairs: {len(X1_train)}, Test pairs: {len(X1_test)}")
        
        return {
            'train': (X1_train, X2_train, y_train),
            'test': (X1_test, X2_test, y_test)
        }
    
    def visualize_samples(self, num_samples=5, save_path=None):
        """Visualize sample signatures"""
        import matplotlib.pyplot as plt
        
        if len(self.genuine_images) == 0:
            self.load_cedar()
        
        fig, axes = plt.subplots(2, num_samples, figsize=(15, 6))
        
        for i in range(num_samples):
            idx = np.random.randint(0, len(self.genuine_images))
            axes[0, i].imshow(self.genuine_images[idx], cmap='gray')
            axes[0, i].set_title('Genuine')
            axes[0, i].axis('off')
        
        for i in range(num_samples):
            idx = np.random.randint(0, len(self.forged_images))
            axes[1, i].imshow(self.forged_images[idx], cmap='gray')
            axes[1, i].set_title('Forged')
            axes[1, i].axis('off')
        
        plt.suptitle('Sample Signatures - Genuine (Top) vs Forged (Bottom)')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300)
        
        plt.show()
    
    def get_dataset_stats(self):
        """Get statistics about the loaded dataset"""
        return {
            'num_genuine': len(self.genuine_images),
            'num_forged': len(self.forged_images),
            'image_shape': self.img_size,
            'total_signatures': len(self.genuine_images) + len(self.forged_images)
        }


# Example usage
if __name__ == "__main__":
    loader = SignatureDatasetLoader(data_dir='data')
    loader.load_cedar()
    print(loader.get_dataset_stats())
    
    X1, X2, y = loader.create_pairs(num_pairs=5000)
    print(f"Created {len(X1)} pairs")
    
    loader.visualize_samples(save_path='results/figures/sample_signatures.png')
    
    data = loader.get_train_test_data()
    print(f"Train data: {data['train'][0].shape}")
    print(f"Test data: {data['test'][0].shape}")