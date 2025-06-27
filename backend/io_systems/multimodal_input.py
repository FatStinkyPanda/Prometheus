# backend/io_systems/multimodal_input.py

import io
import wave
import json
from pathlib import Path
from typing import Dict, Any, Optional

from backend.utils.logger import Logger
from backend.io_systems.io_types import InputPayload, InputType

# --- Optional Dependencies ---
# These are handled gracefully to allow the system to run without full multimodal capabilities.

# Image Processing
try:
    from PIL import Image
    PIL_AVAILABLE = True
except ImportError:
    Image = None
    PIL_AVAILABLE = False

# Document Processing
try:
    import fitz  # PyMuPDF
    PYMUPDF_AVAILABLE = True
except ImportError:
    fitz = None
    PYMUPDF_AVAILABLE = False

# OFFLINE Speech Recognition
try:
    from vosk import Model, KaldiRecognizer
    VOSK_AVAILABLE = True
except ImportError:
    Model, KaldiRecognizer = None, None
    VOSK_AVAILABLE = False

class MultimodalInput:
    """
    Handles the processing of non-textual inputs like images, audio (offline), and documents.

    This class routes input payloads to the appropriate handler based on their type
    and extracts textual content and metadata for further processing by the NLP
    and cognitive minds.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the MultimodalInput handler.

        Args:
            config (Dict[str, Any]): The I/O systems configuration dictionary.
        """
        self.logger = Logger(__name__)
        self.config = config
        self.vosk_model = None
        self._load_vosk_model()
        self.log_dependency_status()

    def _load_vosk_model(self):
        """Loads the offline Vosk speech recognition model."""
        if not VOSK_AVAILABLE:
            return
        
        # The model path should be configurable, but we'll use a default location for now.
        # This assumes the model was placed in 'models/vosk' in the project root.
        model_path = Path(__file__).resolve().parents[2] / 'models' / 'vosk'
        model_name = "vosk-model-small-en-us-0.15" # A good starting model
        full_model_path = model_path / model_name

        if full_model_path.is_dir():
            try:
                self.logger.info(f"Loading offline Vosk model from: {full_model_path}")
                self.vosk_model = Model(str(full_model_path))
                self.logger.info("Offline Vosk model loaded successfully.")
            except Exception as e:
                self.logger.error(f"Failed to load Vosk model from '{full_model_path}'. Error: {e}", exc_info=True)
                self.vosk_model = None
        else:
            self.logger.warning(f"Vosk model directory not found at '{full_model_path}'. Offline speech recognition will be unavailable.")
            self.logger.warning("Please download a Vosk model and place it there. See https://alphacephei.com/vosk/models")


    def log_dependency_status(self):
        """Logs the availability of optional multimodal libraries."""
        self.logger.info("--- Multimodal Dependency Status ---")
        self.logger.info(f"Image Processing (Pillow): {'Available' if PIL_AVAILABLE else 'Not Available'}")
        self.logger.info(f"PDF Processing (PyMuPDF): {'Available' if PYMUPDF_AVAILABLE else 'Not Available'}")
        self.logger.info(f"OFFLINE Speech Recognition (Vosk): {'Available' if VOSK_AVAILABLE else 'Not Available'}")
        if VOSK_AVAILABLE:
            self.logger.info(f"  > Vosk Model Loaded: {'Yes' if self.vosk_model else 'No'}")
        self.logger.info("------------------------------------")
        if not all([PIL_AVAILABLE, PYMUPDF_AVAILABLE, VOSK_AVAILABLE, self.vosk_model]):
            self.logger.warning("One or more multimodal dependencies/models are not available. Related functionality will be disabled.")

    async def process_input(self, payload: InputPayload) -> Optional[Dict[str, Any]]:
        """
        Asynchronously routes an input payload to the correct processing function.
        """
        self.logger.info(f"Processing multimodal input of type: {payload.type.value}")
        
        processing_function_map = {
            InputType.IMAGE: self.process_image,
            InputType.DOCUMENT: self.process_document,
            InputType.AUDIO: self.process_audio,
            InputType.VOICE: self.process_audio,
        }

        handler = processing_function_map.get(payload.type)

        if handler:
            return await handler(payload)
        else:
            self.logger.warning(f"No multimodal handler found for input type '{payload.type}'.")
            return None

    async def process_image(self, payload: InputPayload) -> Optional[Dict[str, Any]]:
        """Processes an image input."""
        if not PIL_AVAILABLE:
            self.logger.error("Cannot process image: Pillow library is not installed.")
            return {"error": "Image processing functionality is not available."}
        try:
            image_bytes = payload.content
            if not isinstance(image_bytes, bytes): raise TypeError("Image payload content must be bytes.")
            with Image.open(io.BytesIO(image_bytes)) as img:
                metadata = {"format": img.format, "mode": img.mode, "width": img.width, "height": img.height}
                extracted_text = f"[Image detected: A {metadata['format']} file of size {metadata['width']}x{metadata['height']}. Image content analysis is not yet implemented.]"
            return {"text_content": extracted_text, "metadata": metadata}
        except Exception as e:
            self.logger.error(f"Failed to process image input. Error: {e}", exc_info=True)
            return {"error": f"Failed to process image: {e}"}

    async def process_document(self, payload: InputPayload) -> Optional[Dict[str, Any]]:
        """Processes a document input (PDFs)."""
        if not PYMUPDF_AVAILABLE:
            self.logger.error("Cannot process document: PyMuPDF (fitz) library is not installed.")
            return {"error": "Document processing functionality is not available."}
        try:
            doc_bytes = payload.content
            if not isinstance(doc_bytes, bytes): raise TypeError("Document payload content must be bytes.")
            with fitz.open(stream=doc_bytes, filetype="pdf") as doc:
                text = "".join(page.get_text() for page in doc)
                metadata = {"page_count": doc.page_count, "author": doc.metadata.get('author'), "title": doc.metadata.get('title')}
            return {"text_content": text, "metadata": metadata}
        except Exception as e:
            self.logger.error(f"Failed to process document input. Error: {e}", exc_info=True)
            return {"error": f"Failed to process document: {e}"}

    async def process_audio(self, payload: InputPayload) -> Optional[Dict[str, Any]]:
        """
        Processes an audio input using the OFFLINE Vosk engine for transcription.
        """
        if not VOSK_AVAILABLE or not self.vosk_model:
            self.logger.error("Cannot process audio: Vosk library or model is not available.")
            return {"error": "Offline audio processing functionality is not available."}
        
        try:
            audio_bytes = payload.content
            if not isinstance(audio_bytes, bytes): raise TypeError("Audio payload content must be bytes.")
            
            # Vosk requires audio in WAV format. We'll use the wave library to read it.
            with io.BytesIO(audio_bytes) as audio_buffer:
                with wave.open(audio_buffer, 'rb') as wf:
                    if wf.getnchannels() != 1 or wf.getsampwidth() != 2 or wf.getcomptype() != "NONE":
                        return {"error": "Audio file must be WAV format mono 16-bit."}
                    
                    # Initialize recognizer with the correct sample rate
                    rec = KaldiRecognizer(self.vosk_model, wf.getframerate())
                    rec.SetWords(True)
                    
                    # Process the audio in chunks
                    while True:
                        data = wf.readframes(4000)
                        if len(data) == 0:
                            break
                        rec.AcceptWaveform(data)
            
            # Get the final result
            result = json.loads(rec.FinalResult())
            extracted_text = result.get('text', '')
            
            metadata = {
                "duration_seconds": len(audio_bytes) / (wf.getframerate() * wf.getsampwidth() * wf.getnchannels()),
                "confidence_per_word": result.get('result', [])
            }
            
            if not extracted_text:
                self.logger.warning("Vosk transcribed the audio but found no text.")
            
            self.logger.info("Successfully transcribed audio to text using offline Vosk model.")
            return {"text_content": extracted_text, "metadata": metadata}

        except Exception as e:
            self.logger.error(f"Failed to process audio input with Vosk. Error: {e}", exc_info=True)
            return {"error": f"Failed to process audio offline: {e}"}