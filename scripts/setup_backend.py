# scripts/setup_backend.py

import subprocess
import sys
import os
import venv
import time
import argparse
from pathlib import Path

# --- Helper for Colored Console Output ---
class Colors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNING = '\033[93m'
    FAIL = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'

def print_color(text, color):
    """Prints text in a given color."""
    print(f"{color}{text}{Colors.ENDC}")

# --- Core Setup Logic ---

def get_venv_executable(venv_path: Path, executable_name: str) -> str:
    """Gets the path to an executable inside the venv."""
    if sys.platform == "win32":
        return str(venv_path / "Scripts" / executable_name)
    else:
        return str(venv_path / "bin" / executable_name)

def main():
    parser = argparse.ArgumentParser(description="Automated setup for the Prometheus backend.")
    parser.add_argument("--force", action="store_true", help="Force re-installation and re-download of all components.")
    args = parser.parse_args()

    print_color("--- Starting Prometheus Backend Smart Setup ---", Colors.BOLD + Colors.HEADER)
    total_start_time = time.monotonic()

    project_root = Path(__file__).resolve().parent.parent
    venv_path = project_root / "venv"
    requirements_file = project_root / 'requirements' / 'backend.txt'
    
    # --- Step 1: Create or Verify Virtual Environment ---
    if not venv_path.exists():
        print_color(f"\n- Creating virtual environment at: {venv_path}", Colors.HEADER)
        venv.create(venv_path, with_pip=True)
        print_color("[SUCCESS] Virtual environment created.", Colors.OKGREEN)
    else:
        print_color(f"\n- Virtual environment found at: {venv_path}", Colors.OKBLUE)

    pip_executable = get_venv_executable(venv_path, "pip")
    python_executable = get_venv_executable(venv_path, "python")

    # --- Step 2: Install Dependencies ---
    print_color("\n- Installing/Verifying Python dependencies...", Colors.HEADER)
    install_start_time = time.monotonic()
    install_command = [pip_executable, 'install']
    if args.force:
        print("  > Force flag detected. Re-installing all packages.")
        install_command.extend(['--force-reinstall', '--no-cache-dir'])
    install_command.extend(['-r', str(requirements_file)])
    
    result = subprocess.run(install_command, capture_output=True, text=True, encoding='utf-8')
    if result.returncode != 0:
        print_color("[ERROR] Pip installation failed.", Colors.FAIL)
        print(result.stdout)
        print(result.stderr)
        sys.exit(1)
    print_color(f"[SUCCESS] Dependencies installed in {time.monotonic() - install_start_time:.2f}s.", Colors.OKGREEN)

    # --- Step 3: Download spaCy Model ---
    SPACY_MODEL = "en_core_web_lg"
    print_color(f"\n- Verifying spaCy model: {SPACY_MODEL}", Colors.HEADER)
    spacy_start_time = time.monotonic()
    
    # *** FIX IS HERE: Use spacy.util.is_model_installed for the check. ***
    # This requires running a small python script.
    check_spacy_script = f"""
import spacy, sys
sys.exit(0) if spacy.util.is_model_installed('{SPACY_MODEL}') else sys.exit(1)
"""
    # The script will exit with code 0 if installed, and 1 if not.
    is_installed = subprocess.run([python_executable, "-c", check_spacy_script]).returncode == 0
    
    if not args.force and is_installed:
        print_color("  > Model already installed. Skipping download.", Colors.OKGREEN)
    else:
        if args.force:
            print_color("  > Force flag detected. Re-downloading model...", Colors.OKBLUE)
        else:
            print_color("  > Model not found. Downloading...", Colors.OKBLUE)
        
        download_spacy_command = [python_executable, "-m", "spacy", "download", SPACY_MODEL]
        download_result = subprocess.run(download_spacy_command, capture_output=True, text=True, encoding='utf-8')
        if download_result.returncode != 0:
            print_color("[ERROR] spaCy model download failed.", Colors.FAIL)
            print(download_result.stdout)
            print(download_result.stderr)
            sys.exit(1)
    print_color(f"[SUCCESS] spaCy model verified in {time.monotonic() - spacy_start_time:.2f}s.", Colors.OKGREEN)

    # --- Step 4: Download Hugging Face Models ---
    print_color("\n- Verifying Hugging Face models...", Colors.HEADER)
    hf_start_time = time.monotonic()
    
    # We use a simple script to leverage the library's caching and progress bars
    hf_downloader_script = f"""
import sys
from huggingface_hub import hf_hub_download
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
# Use parse_known_args to avoid conflicts with the main script's args
script_args, _ = parser.parse_known_args()

models = [
    "sentence-transformers/all-mpnet-base-v2", 
    "TinyLlama/TinyLlama-1.1B-Chat-v1.0", 
    "j-hartmann/emotion-english-distilroberta-base"
]
print("  > Checking status of required Hugging Face models...")
for model in models:
    try:
        # This function checks for the file and downloads it only if missing or if force=True
        hf_hub_download(repo_id=model, filename="config.json", force_download=script_args.force)
        print(f"    - Verified: {{model}}")
    except Exception as e:
        print(f"  [WARN] Could not verify model '{{model}}'. It may need to be downloaded on first run. Error: {{e}}")
"""
    hf_helper_path = project_root / "scripts" / "_hf_helper.py"
    with open(hf_helper_path, "w", encoding="utf-8") as f:
        f.write(hf_downloader_script)
    
    hf_command = [python_executable, str(hf_helper_path)]
    if args.force:
        hf_command.append("--force")
        
    hf_result = subprocess.run(hf_command, text=True, encoding='utf-8')
    os.remove(hf_helper_path)
    if hf_result.returncode != 0:
        print_color("[ERROR] Hugging Face model verification failed.", Colors.FAIL)
        sys.exit(1)
    print_color(f"[SUCCESS] Hugging Face models verified in {time.monotonic() - hf_start_time:.2f}s.", Colors.OKGREEN)

    total_elapsed = time.monotonic() - total_start_time
    print_color(f"\n--- Prometheus Backend Setup Complete! (Total Time: {total_elapsed:.2f}s) ---", Colors.BOLD + Colors.OKGREEN)
    print("The necessary environment has been created/verified.")
    print_color("To run the application, you MUST activate the virtual environment first:", Colors.HEADER)
    if sys.platform == "win32":
        print_color(r"  In PowerShell/CMD:  .\venv\Scripts\activate", Colors.OKCYAN + Colors.BOLD)
    else:
        print_color(r"  In bash/zsh:      source venv/bin/activate", Colors.OKCYAN + Colors.BOLD)
    print("Once activated, you can run the application with the master launcher:")
    print_color("  python run_prometheus.py", Colors.OKCYAN + Colors.BOLD)

if __name__ == "__main__":
    main()