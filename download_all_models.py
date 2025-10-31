#!/usr/bin/env python3
"""
Consolidated script to download all AI models with SSL verification disabled
for corporate environments with certificate issues.

This script downloads:
1. InsightFace buffalo_l model for face detection/recognition
2. BLIP-2 model for image description generation  
3. CLIP model for text-searchable embeddings

Run this script on a network where Hugging Face and GitHub are accessible.
"""

import os
import ssl
import urllib.request
import zipfile
import urllib3
import requests
from pathlib import Path
from transformers import BlipProcessor, BlipForConditionalGeneration, CLIPProcessor, CLIPModel

def setup_ssl_bypass():
    """Setup SSL bypass for corporate environments"""
    # Disable SSL warnings and verification
    urllib3.disable_warnings(urllib3.exceptions.InsecureRequestWarning)
    os.environ['CURL_CA_BUNDLE'] = ''
    os.environ['REQUESTS_CA_BUNDLE'] = ''
    os.environ['HF_HUB_DISABLE_TELEMETRY'] = '1'
    ssl._create_default_https_context = ssl._create_unverified_context
    
    # Patch requests globally to disable SSL verification
    original_request = requests.Session.request
    def patched_request(self, method, url, **kwargs):
        kwargs['verify'] = False
        return original_request(self, method, url, **kwargs)
    requests.Session.request = patched_request
    
    print("🔧 SSL verification disabled for model downloads")

def download_insightface_model():
    """Download InsightFace buffalo_l model manually"""
    print("\n📥 Downloading InsightFace buffalo_l model...")
    
    # Create SSL context that doesn't verify certificates
    ssl_context = ssl.create_default_context()
    ssl_context.check_hostname = False
    ssl_context.verify_mode = ssl.CERT_NONE
    
    # Model URL and paths
    model_url = "https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_l.zip"
    models_dir = Path.home() / ".insightface" / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    zip_path = models_dir / "buffalo_l.zip"
    extract_path = models_dir / "buffalo_l"
    
    # Check if already downloaded
    if extract_path.exists() and any(extract_path.iterdir()):
        print("✅ InsightFace buffalo_l model already exists")
        return True
    
    try:
        print(f"   Downloading from: {model_url}")
        print("   This may take a few minutes...")
        
        # Download with SSL verification disabled
        with urllib.request.urlopen(model_url, context=ssl_context) as response:
            with open(zip_path, 'wb') as f:
                f.write(response.read())
        
        print("   Extracting...")
        
        # Extract the zip file
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(models_dir)
        
        # Clean up zip file
        zip_path.unlink()
        
        print(f"✅ InsightFace model extracted to: {extract_path}")
        return True
        
    except Exception as e:
        print(f"❌ Failed to download InsightFace model: {e}")
        return False

def download_huggingface_models():
    """Download BLIP-2 and CLIP models from Hugging Face"""
    print("\n📥 Downloading Hugging Face models...")
    
    models_downloaded = []
    
    try:
        print("   Downloading BLIP-2 model (Salesforce/blip2-opt-2.7b)...")
        print("   This is a large model and may take 10-15 minutes...")
        
        blip_processor = BlipProcessor.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            trust_remote_code=True,
            token=False
        )
        blip_model = BlipForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b",
            trust_remote_code=True,
            token=False
        )
        print("✅ BLIP-2 model downloaded successfully")
        models_downloaded.append("BLIP-2")
        
    except Exception as e:
        print(f"❌ Failed to download BLIP-2 model: {e}")
    
    try:
        print("   Downloading CLIP model (openai/clip-vit-base-patch32)...")
        
        clip_processor = CLIPProcessor.from_pretrained(
            "openai/clip-vit-base-patch32",
            trust_remote_code=True,
            token=False
        )
        clip_model = CLIPModel.from_pretrained(
            "openai/clip-vit-base-patch32",
            trust_remote_code=True,
            token=False
        )
        print("✅ CLIP model downloaded successfully")
        models_downloaded.append("CLIP")
        
    except Exception as e:
        print(f"❌ Failed to download CLIP model: {e}")
    
    return models_downloaded

def main():
    """Main function to download all models"""
    print("🚀 AI Photo Analysis Service - Model Downloader")
    print("=" * 60)
    print("This script will download all required AI models:")
    print("• InsightFace buffalo_l (face detection/recognition)")
    print("• BLIP-2 opt-2.7b (image description generation)")
    print("• CLIP vit-base-patch32 (text-searchable embeddings)")
    print("=" * 60)
    
    # Setup SSL bypass
    setup_ssl_bypass()
    
    # Download InsightFace model
    insightface_success = download_insightface_model()
    
    # Download Hugging Face models
    hf_models = download_huggingface_models()
    
    # Summary
    print("\n" + "=" * 60)
    print("📊 Download Summary:")
    print(f"   InsightFace: {'✅ SUCCESS' if insightface_success else '❌ FAILED'}")
    print(f"   Hugging Face models: {len(hf_models)} of 2 downloaded")
    for model in hf_models:
        print(f"     • {model}: ✅ SUCCESS")
    
    if not hf_models:
        print("     • BLIP-2: ❌ FAILED")
        print("     • CLIP: ❌ FAILED")
    elif len(hf_models) == 1:
        missing = "CLIP" if "BLIP-2" in hf_models else "BLIP-2"
        print(f"     • {missing}: ❌ FAILED")
    
    print("\n💡 Notes:")
    print("   • Face detection will work with just InsightFace")
    print("   • Image descriptions need BLIP-2 model")
    print("   • Text-searchable embeddings need CLIP model")
    print("   • Service has fallback mechanisms for missing models")
    
    if insightface_success:
        print("\n🎉 Core functionality ready! You can start the service.")
        print("   Run: python main.py")
    else:
        print("\n⚠️  Face detection may not work without InsightFace model.")
    
    print("\n🌐 If downloads failed due to network restrictions:")
    print("   • Try running this script on a different network")
    print("   • Corporate firewalls may block GitHub/Hugging Face")
    print("   • The service will use fallback mechanisms for missing models")

if __name__ == "__main__":
    main()