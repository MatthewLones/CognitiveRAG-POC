#!/usr/bin/env python3
"""
Startup script for Cognitive RAG POC
This script starts both the FastAPI backend and Streamlit frontend
"""

import subprocess
import time
import sys
import os
from pathlib import Path

# Set tokenizers parallelism to avoid warnings
os.environ["TOKENIZERS_PARALLELISM"] = "false"


def check_data_directory():
    """Check if data directory has documents"""
    data_path = Path("data")
    if not data_path.exists():
        print("❌ Data directory not found")
        return False
    
    pdf_files = list(data_path.glob("*.pdf"))
    if not pdf_files:
        print("⚠️  No PDF files found in data/ directory")
        print("📄 Add PDF files to data/ for the RAG system to process")
        return False
    
    print(f"✅ Found {len(pdf_files)} PDF files in data/ directory")
    return True

def start_backend():
    """Start the FastAPI backend"""
    print("\n🚀 Starting FastAPI backend...")
    print("📍 Backend will be available at: http://localhost:8000")
    print("📖 API docs at: http://localhost:8000/docs")
    
    try:
        # Start uvicorn server
        subprocess.run([
            sys.executable, "-m", "uvicorn", 
            "app.api:app", 
            "--host", "0.0.0.0", 
            "--port", "8000", 
            "--reload"
        ], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Backend stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting backend: {e}")

def start_frontend():
    """Start the Streamlit frontend"""
    print("\n🎨 Starting Streamlit frontend...")
    print("📍 Frontend will be available at: http://localhost:8501")
    
    try:
        # Start streamlit
        subprocess.run([
            sys.executable, "-m", "streamlit", "run", 
            "app/ui.py",
            "--server.port", "8501",
            "--server.address", "0.0.0.0"
        ], check=True)
    except KeyboardInterrupt:
        print("\n⏹️  Frontend stopped by user")
    except subprocess.CalledProcessError as e:
        print(f"❌ Error starting frontend: {e}")

def main():
    """Main startup function"""
    print("🧠 Cognitive RAG POC - Startup Script")
    print("=" * 50)
    
    if not check_data_directory():
        sys.exit(1)
    
    print("\n✅ All checks passed! Starting the system...")
    
    # Ask user what to start
    print("\nWhat would you like to start?")
    print("1. Backend only (FastAPI)")
    print("2. Frontend only (Streamlit)")
    print("3. Both (recommended)")
    
    choice = input("\nEnter your choice (1-3): ").strip()
    
    if choice == "1":
        start_backend()
    elif choice == "2":
        start_frontend()
    elif choice == "3":
        print("\n🔄 Starting both services...")
        print("💡 Tip: Start backend first, then frontend in separate terminals")
        print("   Or use a process manager like PM2 for production")
        
        # For demo purposes, start backend first
        print("\n Starting backend first...")
        print("   Press Ctrl+C to stop backend, then start frontend separately")
        start_backend()
    else:
        print("❌ Invalid choice. Please run the script again.")

if __name__ == "__main__":
    main()
