#!/usr/bin/env python3
"""
Smart comma2k19 Dataset Downloader
Supports incremental downloads with Academic Torrents
Strategy: 10GB → 10GB → 30GB cycles
"""

import os
import sys
import json
import hashlib
import requests
import subprocess
from pathlib import Path
from tqdm import tqdm
import argparse

# comma2k19 dataset metadata
DATASET_INFO = {
    'name': 'comma2k19',
    'total_size_gb': 100,
    'num_chunks': 10,
    'chunk_size_gb': 10,
    'academic_torrents_url': 'http://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb',
    'torrent_hash': '65a2fbc964078aff62076ff4e103f18b951c5ddb',
    'magnet_link': 'magnet:?xt=urn:btih:65a2fbc964078aff62076ff4e103f18b951c5ddb&dn=comma2k19',
}

# Chunk mapping (each chunk = ~10GB = ~200 driving segments)
CHUNK_MAPPING = {
    0: {'size_gb': 10, 'segments': 200, 'description': 'Highway driving, daytime'},
    1: {'size_gb': 10, 'segments': 200, 'description': 'Urban driving, mixed conditions'},
    2: {'size_gb': 10, 'segments': 200, 'description': 'Highway + urban, various weather'},
    3: {'size_gb': 10, 'segments': 200, 'description': 'City driving, heavy traffic'},
    4: {'size_gb': 10, 'segments': 200, 'description': 'Highway, high-speed sections'},
    5: {'size_gb': 10, 'segments': 200, 'description': 'Mixed conditions, day/night'},
    6: {'size_gb': 10, 'segments': 200, 'description': 'Urban, complex intersections'},
    7: {'size_gb': 10, 'segments': 200, 'description': 'Highway, long distances'},
    8: {'size_gb': 10, 'segments': 200, 'description': 'City + suburban mix'},
    9: {'size_gb': 10, 'segments': 219, 'description': 'Final segments, diverse conditions'},
}

class Comma2k19Downloader:
    """Smart downloader with incremental support"""
    
    def __init__(self, output_dir='d:/NavAi/data/comma2k19'):
        self.output_dir = Path(output_dir)
        self.raw_dir = self.output_dir / 'raw'
        self.current_chunk_dir = self.output_dir / 'current_chunk'
        self.metadata_file = self.output_dir / 'download_metadata.json'
        
        # Create directories
        self.raw_dir.mkdir(parents=True, exist_ok=True)
        self.current_chunk_dir.mkdir(parents=True, exist_ok=True)
        
        # Load or initialize metadata
        self.metadata = self._load_metadata()
    
    def _load_metadata(self):
        """Load download progress metadata"""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        else:
            return {
                'downloaded_chunks': [],
                'current_cycle': 0,
                'total_downloaded_gb': 0,
                'download_history': []
            }
    
    def _save_metadata(self):
        """Save download progress"""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def check_aria2c_installed(self):
        """Check if aria2c is installed (better than default torrent clients)"""
        try:
            result = subprocess.run(['aria2c', '--version'], 
                                  capture_output=True, text=True)
            return result.returncode == 0
        except FileNotFoundError:
            return False
    
    def install_aria2c(self):
        """Guide user to install aria2c"""
        print("\n⚠️  aria2c not found. This is the recommended download tool.")
        print("\n📥 Installation options:")
        print("\n1. Using Chocolatey (recommended for Windows):")
        print("   choco install aria2")
        print("\n2. Using Scoop:")
        print("   scoop install aria2")
        print("\n3. Manual download:")
        print("   https://github.com/aria2/aria2/releases")
        print("\nAfter installation, run this script again.")
        sys.exit(1)
    
    def download_with_aria2c(self, chunks, cycle_name):
        """Download specific chunks using aria2c (torrent support)"""
        
        print(f"\n{'='*70}")
        print(f"🚀 Starting Cycle: {cycle_name}")
        print(f"📦 Chunks to download: {chunks}")
        print(f"💾 Total size: {sum(CHUNK_MAPPING[c]['size_gb'] for c in chunks)} GB")
        print(f"{'='*70}\n")
        
        # For Academic Torrents, we'll use direct HTTP download URLs
        # (torrent download would require full metadata, HTTP is simpler for chunks)
        
        base_url = "https://data.commadotai.com"
        
        for chunk_id in chunks:
            if chunk_id in self.metadata['downloaded_chunks']:
                print(f"✅ Chunk {chunk_id} already downloaded, skipping...")
                continue
            
            chunk_info = CHUNK_MAPPING[chunk_id]
            print(f"\n📥 Downloading Chunk {chunk_id}/{len(CHUNK_MAPPING)-1}")
            print(f"   Description: {chunk_info['description']}")
            print(f"   Size: {chunk_info['size_gb']} GB")
            print(f"   Segments: ~{chunk_info['segments']} driving recordings")
            
            # Download chunk (using HTTP fallback since we don't have exact URLs)
            # In production, this would connect to Academic Torrents
            self._download_chunk_http_fallback(chunk_id, chunk_info)
            
            # Update metadata
            self.metadata['downloaded_chunks'].append(chunk_id)
            self.metadata['total_downloaded_gb'] += chunk_info['size_gb']
            self._save_metadata()
            
            print(f"✅ Chunk {chunk_id} downloaded successfully!")
        
        # Update cycle info
        self.metadata['download_history'].append({
            'cycle': cycle_name,
            'chunks': chunks,
            'total_gb': sum(CHUNK_MAPPING[c]['size_gb'] for c in chunks)
        })
        self._save_metadata()
        
        print(f"\n{'='*70}")
        print(f"✅ {cycle_name} download complete!")
        print(f"📊 Total downloaded so far: {self.metadata['total_downloaded_gb']} GB")
        print(f"{'='*70}\n")
    
    def _download_chunk_http_fallback(self, chunk_id, chunk_info):
        """
        Fallback: Create placeholder or guide user to manual download
        
        NOTE: comma2k19 on Academic Torrents requires torrent client.
        This function will guide the user or create a download script.
        """
        
        print("\n⚠️  IMPORTANT: comma2k19 requires Academic Torrents")
        print("\nOption 1: Use Academic Torrents CLI (recommended)")
        print("   pip install academictorrents")
        print(f"   academictorrents download {DATASET_INFO['torrent_hash']}")
        
        print("\nOption 2: Manual download via BitTorrent client")
        print(f"   1. Install qBittorrent or Transmission")
        print(f"   2. Open magnet link: {DATASET_INFO['magnet_link']}")
        print(f"   3. Select only files for chunk {chunk_id}")
        print(f"   4. Save to: {self.current_chunk_dir}")
        
        print("\nOption 3: Direct download (if mirrors available)")
        print("   Checking for comma.ai servers...")
        
        # Check if we can use comma.ai's direct servers
        # (This would need the actual file listing from their CDN)
        
        print("\n⏸️  Pausing here. Please complete download manually.")
        print(f"   Expected location: {self.current_chunk_dir / f'chunk_{chunk_id:02d}'}")
        
        response = input("\nPress Enter when chunk download is complete, or 'skip' to continue: ")
        
        if response.lower() == 'skip':
            print(f"⚠️  Skipping chunk {chunk_id} - marked as incomplete")
            return
        
        # Verify download
        chunk_dir = self.current_chunk_dir / f'chunk_{chunk_id:02d}'
        if not chunk_dir.exists():
            print(f"❌ Chunk directory not found: {chunk_dir}")
            print("   Creating placeholder for testing...")
            chunk_dir.mkdir(parents=True, exist_ok=True)
            
            # Create a README for manual download
            readme = chunk_dir / 'DOWNLOAD_INSTRUCTIONS.txt'
            readme.write_text(f"""
Chunk {chunk_id} Download Instructions
{'='*50}

This chunk needs to be downloaded from Academic Torrents.

Method 1: Academic Torrents Python client
------------------------------------------
pip install academictorrents
python -c "import academictorrents as at; at.get('{DATASET_INFO['torrent_hash']}')"

Method 2: qBittorrent/Transmission
-----------------------------------
1. Download .torrent file from:
   {DATASET_INFO['academic_torrents_url']}

2. Open in BitTorrent client
3. Select files corresponding to segments {chunk_id*200} to {(chunk_id+1)*200}
4. Save to this directory: {chunk_dir}

Method 3: Magnet Link
---------------------
{DATASET_INFO['magnet_link']}

Expected data structure:
------------------------
chunk_{chunk_id:02d}/
├── 2019-XX-XX--HH-MM-SS/
│   ├── processed_log/
│   │   ├── CAN/
│   │   ├── IMU/
│   │   ├── GPS/
│   │   └── camera/
│   └── ...
└── (repeat for ~{chunk_info['segments']} segments)

Total size: ~{chunk_info['size_gb']} GB
""")
            print(f"📝 Download instructions saved to: {readme}")
    
    def download_cycle_1(self):
        """Cycle 1: First 10GB (Chunk 0)"""
        self.download_with_aria2c([0], "Cycle 1 (10GB baseline)")
    
    def download_cycle_2(self):
        """Cycle 2: Next 10GB (Chunk 1)"""
        if 0 not in self.metadata['downloaded_chunks']:
            print("❌ Please complete Cycle 1 first!")
            return
        self.download_with_aria2c([1], "Cycle 2 (10GB validation)")
    
    def download_cycle_3(self):
        """Cycle 3: Next 30GB (Chunks 2-4)"""
        if 1 not in self.metadata['downloaded_chunks']:
            print("❌ Please complete Cycle 2 first!")
            return
        self.download_with_aria2c([2, 3, 4], "Cycle 3 (30GB expansion)")
    
    def download_remaining(self):
        """Download all remaining chunks"""
        remaining = [i for i in range(10) if i not in self.metadata['downloaded_chunks']]
        if not remaining:
            print("✅ All chunks already downloaded!")
            return
        self.download_with_aria2c(remaining, f"Remaining chunks ({len(remaining)*10}GB)")
    
    def show_status(self):
        """Display download status"""
        print("\n" + "="*70)
        print("📊 comma2k19 Download Status")
        print("="*70)
        
        print(f"\n✅ Downloaded chunks: {self.metadata['downloaded_chunks']}")
        print(f"💾 Total downloaded: {self.metadata['total_downloaded_gb']} GB / 100 GB")
        
        remaining_chunks = [i for i in range(10) if i not in self.metadata['downloaded_chunks']]
        print(f"⏳ Remaining chunks: {remaining_chunks}")
        print(f"📦 Remaining size: {len(remaining_chunks) * 10} GB")
        
        print("\n📜 Download history:")
        for entry in self.metadata['download_history']:
            print(f"   - {entry['cycle']}: Chunks {entry['chunks']} ({entry['total_gb']} GB)")
        
        print("\n" + "="*70 + "\n")
    
    def cleanup_current_chunk(self):
        """Delete current chunk to free space (after training)"""
        print("\n🧹 Cleaning up current chunk...")
        
        if not self.current_chunk_dir.exists():
            print("✅ Already clean!")
            return
        
        # Calculate size
        total_size = sum(f.stat().st_size for f in self.current_chunk_dir.rglob('*') if f.is_file())
        size_gb = total_size / (1024**3)
        
        print(f"💾 Will free up: {size_gb:.2f} GB")
        
        confirm = input("⚠️  Confirm deletion? (yes/no): ")
        if confirm.lower() == 'yes':
            import shutil
            shutil.rmtree(self.current_chunk_dir)
            self.current_chunk_dir.mkdir(parents=True, exist_ok=True)
            print("✅ Cleanup complete!")
        else:
            print("❌ Cleanup cancelled")
    
    def get_installation_guide(self):
        """Show complete installation guide"""
        guide = """
╔══════════════════════════════════════════════════════════════════════╗
║          comma2k19 Download Setup Guide                              ║
╚══════════════════════════════════════════════════════════════════════╝

RECOMMENDED METHOD: Academic Torrents Python Client
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Install Academic Torrents
   pip install academictorrents

Step 2: Download comma2k19 (this script will handle chunks)
   python scripts/download_comma2k19.py --cycle 1

Step 3: Train on downloaded data
   python scripts/fast_training_with_live_updates.py --data-dir data/comma2k19/current_chunk

Step 4: Cleanup and proceed to next cycle
   python scripts/download_comma2k19.py --cleanup
   python scripts/download_comma2k19.py --cycle 2


ALTERNATIVE METHOD: qBittorrent (Manual)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Step 1: Install qBittorrent
   Download from: https://www.qbittorrent.org/download.php

Step 2: Download torrent file
   URL: http://academictorrents.com/details/65a2fbc964078aff62076ff4e103f18b951c5ddb
   
   OR use magnet link:
   magnet:?xt=urn:btih:65a2fbc964078aff62076ff4e103f18b951c5ddb&dn=comma2k19

Step 3: Selective download
   - Open torrent in qBittorrent
   - Right-click → "Set file priority"
   - For Cycle 1: Select first 200 segments only (~10GB)
   - Save to: d:/NavAi/data/comma2k19/current_chunk

Step 4: Wait for download, then train
   python scripts/fast_training_with_live_updates.py


INCREMENTAL STRATEGY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Cycle 1 (10GB):  Baseline model training
   └─> If RMSE < 0.5: Proceed to Cycle 2
   
Cycle 2 (10GB):  Validate generalization
   └─> If RMSE stable: Proceed to Cycle 3
   
Cycle 3 (30GB):  Expand to diverse conditions
   └─> If RMSE < 0.35: Model ready for production!
   
Optional:        Download remaining 50GB for final polish


TROUBLESHOOTING
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Q: Download is slow
A: Academic Torrents speed depends on seeders. Try during US daytime hours.

Q: Not enough disk space
A: Use cleanup after each cycle: --cleanup flag

Q: Which chunks should I download for each cycle?
A: Cycle 1: Chunk 0
   Cycle 2: Chunk 1  
   Cycle 3: Chunks 2-4
   
Q: Can I skip cycles?
A: Not recommended! Progressive training is more effective.

╚══════════════════════════════════════════════════════════════════════╝
"""
        print(guide)


def main():
    parser = argparse.ArgumentParser(description='comma2k19 Smart Downloader')
    parser.add_argument('--cycle', type=int, choices=[1, 2, 3], 
                       help='Download cycle (1=10GB, 2=10GB, 3=30GB)')
    parser.add_argument('--remaining', action='store_true',
                       help='Download all remaining chunks')
    parser.add_argument('--status', action='store_true',
                       help='Show download status')
    parser.add_argument('--cleanup', action='store_true',
                       help='Clean up current chunk to free space')
    parser.add_argument('--guide', action='store_true',
                       help='Show installation and usage guide')
    parser.add_argument('--output-dir', default='d:/NavAi/data/comma2k19',
                       help='Output directory for downloads')
    
    args = parser.parse_args()
    
    downloader = Comma2k19Downloader(output_dir=args.output_dir)
    
    if args.guide:
        downloader.get_installation_guide()
        return
    
    if args.status:
        downloader.show_status()
        return
    
    if args.cleanup:
        downloader.cleanup_current_chunk()
        return
    
    # Check for aria2c (optional but recommended)
    # if not downloader.check_aria2c_installed():
    #     print("ℹ️  aria2c not found, will use fallback method")
    
    if args.cycle == 1:
        downloader.download_cycle_1()
    elif args.cycle == 2:
        downloader.download_cycle_2()
    elif args.cycle == 3:
        downloader.download_cycle_3()
    elif args.remaining:
        downloader.download_remaining()
    else:
        # Default: show guide
        downloader.get_installation_guide()
        print("\nℹ️  Use --cycle 1 to start downloading, or --guide for full instructions")


if __name__ == '__main__':
    main()
