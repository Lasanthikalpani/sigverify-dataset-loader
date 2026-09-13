"""
Data Collection Methods for SigVerify
Achieving 1000+ Original Signatures and 5000+ Augmented Images
"""

import os
import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm


class SignatureDataCollector:
    """
    Collect and combine multiple signature datasets
    Target: 1000+ original signatures
    """
    
    def __init__(self, data_dir='data'):
        self.data_dir = Path(data_dir)
        self.raw_dir = self.data_dir / 'raw'
        self.processed_dir = self.data_dir / 'processed'
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        
        # Data storage
        self.genuine_images = []
        self.forged_images = []
    
    def load_cedar(self):
        """Load CEDAR dataset"""
        cedar_dir = self.raw_dir / 'cedar'
        
        if not cedar_dir.exists():
            print("⚠️ CEDAR dataset not found")
            return 0, 0
        
        initial_genuine = len(self.genuine_images)
        initial_forged = len(self.forged_images)
        
        for person_dir in cedar_dir.iterdir():
            if person_dir.is_dir():
                for img_file in person_dir.iterdir():
                    if img_file.suffix.lower() in ['.png', '.jpg', '.jpeg']:
                        img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            img = cv2.resize(img, (128, 128)) / 255.0
                            if 'F' in img_file.stem:
                                self.forged_images.append(img)
                            else:
                                self.genuine_images.append(img)
        
        genuine_loaded = len(self.genuine_images) - initial_genuine
        forged_loaded = len(self.forged_images) - initial_forged
        
        print(f"✅ CEDAR: {genuine_loaded} genuine, {forged_loaded} forged")
        return genuine_loaded, forged_loaded
    
    def load_gpds(self):
        """Load GPDS dataset"""
        gpds_dir = self.raw_dir / 'gpds'
        
        if not gpds_dir.exists():
            print("⚠️ GPDS dataset not found")
            return 0, 0
        
        initial_genuine = len(self.genuine_images)
        initial_forged = len(self.forged_images)
        
        # Load genuine
        genuine_dir = gpds_dir / 'Reference'
        if genuine_dir.exists():
            for img_file in genuine_dir.glob('*.png'):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (128, 128)) / 255.0
                    self.genuine_images.append(img)
        
        # Load forged
        forged_dir = gpds_dir / 'Forged'
        if forged_dir.exists():
            for img_file in forged_dir.glob('*.png'):
                img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    img = cv2.resize(img, (128, 128)) / 255.0
                    self.forged_images.append(img)
        
        genuine_loaded = len(self.genuine_images) - initial_genuine
        forged_loaded = len(self.forged_images) - initial_forged
        
        print(f"✅ GPDS: {genuine_loaded} genuine, {forged_loaded} forged")
        return genuine_loaded, forged_loaded
    
    def load_bhsig(self):
        """Load BHSig dataset"""
        bhsig_dir = self.raw_dir / 'bhsig'
        
        if not bhsig_dir.exists():
            print("⚠️ BHSig dataset not found")
            return 0, 0
        
        initial_genuine = len(self.genuine_images)
        initial_forged = len(self.forged_images)
        
        for img_file in bhsig_dir.rglob('*.png'):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128)) / 255.0
                if 'forged' in str(img_file).lower():
                    self.forged_images.append(img)
                else:
                    self.genuine_images.append(img)
        
        genuine_loaded = len(self.genuine_images) - initial_genuine
        forged_loaded = len(self.forged_images) - initial_forged
        
        print(f"✅ BHSig: {genuine_loaded} genuine, {forged_loaded} forged")
        return genuine_loaded, forged_loaded
    
    def load_kaggle(self, max_images=10000):
        """Load Kaggle dataset (manishvem/signatures-dataset)
        
        Structure:
        - Folder without _forg = genuine (e.g., 1472/)
        - Folder with _forg = forged (e.g., 1472_forg/)
        """
        kaggle_dir = self.raw_dir / 'kaggle'
        
        if not kaggle_dir.exists():
            print("⚠️ Kaggle dataset not found")
            return 0, 0
        
        initial_genuine = len(self.genuine_images)
        initial_forged = len(self.forged_images)
        
        # Get all image files
        all_files = list(kaggle_dir.rglob('*.jpg')) + list(kaggle_dir.rglob('*.png'))
        print(f"   Found {len(all_files)} total images")
        
        # Limit for speed
        if len(all_files) > max_images:
            np.random.shuffle(all_files)
            all_files = all_files[:max_images]
            print(f"   Limited to {max_images} images for speed")
        
        # Process
        for img_file in tqdm(all_files, desc="Loading Kaggle"):
            img = cv2.imread(str(img_file), cv2.IMREAD_GRAYSCALE)
            if img is not None:
                img = cv2.resize(img, (128, 128)) / 255.0
                
                # Check if parent folder has '_forg' in name
                parent_folder = img_file.parent.name
                if '_forg' in parent_folder.lower() or 'forged' in parent_folder.lower():
                    self.forged_images.append(img)
                else:
                    self.genuine_images.append(img)
        
        genuine_loaded = len(self.genuine_images) - initial_genuine
        forged_loaded = len(self.forged_images) - initial_forged
        
        print(f"✅ Kaggle: {genuine_loaded} genuine, {forged_loaded} forged")
        return genuine_loaded, forged_loaded
    
    def load_all(self):
        """Load all available datasets"""
        print("=" * 60)
        print("📂 LOADING ALL DATASETS")
        print("=" * 60)
        
        self.load_cedar()
        self.load_gpds()
        self.load_bhsig()
        self.load_kaggle()
        
        print("\n" + "=" * 60)
        print(f"📊 TOTAL ORIGINAL:")
        print(f"   Genuine: {len(self.genuine_images)}")
        print(f"   Forged: {len(self.forged_images)}")
        print(f"   Total: {len(self.genuine_images) + len(self.forged_images)}")
        print("=" * 60)
        
        return self.genuine_images, self.forged_images


class SignatureAugmenter:
    """
    Apply data augmentation to signatures
    Target: 5000+ augmented images
    """
    
    def __init__(self):
        self.augmented_genuine = []
        self.augmented_forged = []
    
    def augment_image(self, img):
        """
        Apply multiple augmentations to a single image
        Returns list of augmented images
        """
        augmented = [img]  # Original
        
        # 1. Rotation (+/- 15, 10, 5 degrees)
        for angle in [-15, -10, -5, 5, 10, 15]:
            M = cv2.getRotationMatrix2D((64, 64), angle, 1)
            rotated = cv2.warpAffine(img, M, (128, 128))
            augmented.append(rotated)
        
        # 2. Horizontal Flip
        augmented.append(cv2.flip(img, 1))
        
        # 3. Vertical Flip
        augmented.append(cv2.flip(img, 0))
        
        # 4. Scale (0.85, 0.9, 0.95, 1.05, 1.1, 1.15)
        for scale in [0.85, 0.9, 0.95, 1.05, 1.1, 1.15]:
            new_size = int(128 * scale)
            scaled = cv2.resize(img, (new_size, new_size))
            
            if new_size > 128:
                # Crop center
                start = (new_size - 128) // 2
                scaled = scaled[start:start+128, start:start+128]
            else:
                # Pad
                padded = np.zeros((128, 128))
                start = (128 - new_size) // 2
                padded[start:start+new_size, start:start+new_size] = scaled
                scaled = padded
            
            augmented.append(scaled)
        
        # 5. Translation
        for dx, dy in [(5,0), (-5,0), (0,5), (0,-5)]:
            M = np.float32([[1, 0, dx], [0, 1, dy]])
            translated = cv2.warpAffine(img, M, (128, 128))
            augmented.append(translated)
        
        # 6. Gaussian Noise
        noise = np.random.normal(0, 0.05, img.shape)
        augmented.append(np.clip(img + noise, 0, 1))
        
        # 7. Blur
        augmented.append(cv2.GaussianBlur(img, (3, 3), 0))
        
        return augmented
    
    def augment_dataset(self, images, target_count):
        """Augment dataset to target count"""
        print(f"\n🔄 Augmenting {len(images)} images to {target_count}...")
        
        augmented = []
        images = list(images)
        
        while len(augmented) < target_count:
            for img in tqdm(images, desc="Augmenting"):
                aug_versions = self.augment_image(img)
                augmented.extend(aug_versions)
                
                if len(augmented) >= target_count:
                    break
        
        augmented = augmented[:target_count]
        np.random.shuffle(augmented)
        
        print(f"   ✅ Augmented to: {len(augmented)} images")
        
        return np.array(augmented)
    
    def augment_all(self, genuine, forged, target_total=5000):
        """Augment both genuine and forged to target total"""
        target_each = target_total // 2
        
        print("=" * 60)
        print("🔄 DATA AUGMENTATION")
        print("=" * 60)
        
        self.augmented_genuine = self.augment_dataset(genuine, target_each)
        self.augmented_forged = self.augment_dataset(forged, target_each)
        
        print("\n" + "=" * 60)
        print("📊 AUGMENTED SUMMARY")
        print("=" * 60)
        print(f"   Genuine: {len(self.augmented_genuine)}")
        print(f"   Forged: {len(self.augmented_forged)}")
        print(f"   Total: {len(self.augmented_genuine) + len(self.augmented_forged)}")
        print("=" * 60)
        
        return self.augmented_genuine, self.augmented_forged


def run_data_pipeline(target_original=1000, target_augmented=5000):
    """
    Run complete data collection and augmentation pipeline
    """
    print("=" * 60)
    print("🎯 SIGVERIFY DATA PIPELINE")
    print("=" * 60)
    print(f"   Target original: {target_original}")
    print(f"   Target augmented: {target_augmented}")
    print("=" * 60)
    
    # Step 1: Collect data
    collector = SignatureDataCollector(data_dir='data')
    genuine, forged = collector.load_all()
    
    original_total = len(genuine) + len(forged)
    print(f"\n📊 Original total: {original_total}")
    
    if original_total < target_original:
        print(f"\n⚠️ Need {target_original - original_total} more original signatures")
        print("   Please download more datasets:")
        print("   - CEDAR: https://www.cedar.buffalo.edu/NIJ/data/signatures.rar")
        print("   - GPDS: https://gpds.ulpgc.es/")
        print("   - Kaggle: kagglehub.dataset_download('divyanshi/signature-verification')")
    
    # Step 2: Augment data
    augmenter = SignatureAugmenter()
    genuine_aug, forged_aug = augmenter.augment_all(
        genuine, forged, target_total=target_augmented
    )
    
    # Step 3: Save
    print("\n" + "=" * 60)
    print("💾 SAVING AUGMENTED DATA")
    print("=" * 60)
    
    os.makedirs('data/processed', exist_ok=True)
    np.save('data/processed/genuine_augmented.npy', genuine_aug)
    np.save('data/processed/forged_augmented.npy', forged_aug)
    
    print(f"   ✅ Saved genuine: {genuine_aug.shape}")
    print(f"   ✅ Saved forged: {forged_aug.shape}")
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 FINAL SUMMARY")
    print("=" * 60)
    print(f"   Original genuine: {len(genuine)}")
    print(f"   Original forged: {len(forged)}")
    print(f"   Original total: {original_total}")
    print(f"   Augmented genuine: {len(genuine_aug)}")
    print(f"   Augmented forged: {len(forged_aug)}")
    print(f"   Augmented total: {len(genuine_aug) + len(forged_aug)}")
    print("=" * 60)
    print("✅ PIPELINE COMPLETE!")
    
    return genuine_aug, forged_aug


if __name__ == "__main__":
    run_data_pipeline(target_original=1000, target_augmented=5000)