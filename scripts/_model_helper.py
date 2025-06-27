
import sys, os, time, argparse
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

parser = argparse.ArgumentParser()
parser.add_argument("--force", action="store_true")
script_args = parser.parse_args()

SPACY_MODEL = "en_core_web_lg"
HF_MODELS = ["sentence-transformers/all-mpnet-base-v2", "gpt2", "j-hartmann/emotion-english-distilroberta-base"]

class C:
    H, OK, W, F, E = '\033[95m', '\033[92m', '\033[93m', '\033[91m', '\033[0m'

def download_spacy():
    try:
        import spacy
        print(f"{C.H}- Checking spaCy model: {SPACY_MODEL}{C.E}")
        if not script_args.force and spacy.util.is_model_installed(SPACY_MODEL):
            print(f"  {C.OK}> Model already installed. Skipping.{C.E}")
        else:
            print(f"  > Downloading... This may take a moment.")
            subprocess.check_call([sys.executable, '-m', 'spacy', 'download', SPACY_MODEL])
    except Exception as e:
        print(f"{C.F}[ERROR] Failed to download spaCy model: {e}{C.E}")
        sys.exit(1)

def download_hf():
    try:
        from huggingface_hub import hf_hub_download
        print(f"{C.H}- Checking Hugging Face models...{C.E}")
        for model in HF_MODELS:
            print(f"  > Verifying: {model}")
            try:
                # The hf_hub_download function with force_download=False will check before downloading.
                hf_hub_download(repo_id=model, filename="config.json", force_download=script_args.force)
            except Exception as e:
                print(f"    {C.F}! Error checking model '{model}': {e}{C.E}")
    except Exception as e:
        print(f"{C.F}[ERROR] huggingface_hub check failed: {e}{C.E}")

if __name__ == "__main__":
    download_spacy()
    download_hf()
