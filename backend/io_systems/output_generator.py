# backend/io_systems/output_generator.py

import asyncio
import json
import io
import os
import tempfile
import uuid
from typing import Dict, Any, Optional, AsyncGenerator

import torch # For type checking if internal_state contains tensors
from backend.io_systems.io_types import OutputPayload, OutputType
from backend.utils.logger import Logger
from backend.io_systems.natural_language_processor import NaturalLanguageProcessor

# --- Optional Dependency for OFFLINE Text-to-Speech ---
try:
    import pyttsx3
    PYTTSX3_AVAILABLE = True
except ImportError:
    pyttsx3 = None
    PYTTSX3_AVAILABLE = False


class OutputGenerator:
    """
    Generates the final user-facing output based on the integrated state of the consciousness.
    This class can format responses as text, audio (offline), structured data, and can now
    stream text-based outputs.
    """

    def __init__(self, config: Dict[str, Any], nlp_processor: NaturalLanguageProcessor):
        """
        Initializes the OutputGenerator.

        Args:
            config (Dict[str, Any]): The I/O systems configuration dictionary.
            nlp_processor (NaturalLanguageProcessor): An instance of the NLP processor for tasks like summarization.
        """
        self.logger = Logger(__name__)
        self.config = config.get('output_generator', {})
        self.nlp_processor = nlp_processor
        self.log_dependency_status()

    def log_dependency_status(self):
        """Logs the availability of optional output generation libraries."""
        self.logger.info("--- Output Generation Dependency Status ---")
        self.logger.info(f"OFFLINE Audio Generation (pyttsx3): {'Available' if PYTTSX3_AVAILABLE else 'Not Available'}")
        self.logger.info("-----------------------------------------")
        if not PYTTSX3_AVAILABLE:
            self.logger.warning("pyttsx3 library not installed. Offline text-to-speech functionality will be disabled.")
            self.logger.warning("Please run: pip install pyttsx3")
            
    async def generate_output_stream(self, internal_state: Dict[str, Any], preferred_output_type: OutputType) -> AsyncGenerator[OutputPayload, None]:
        """
        The main router for generating a stream of output payloads.

        Args:
            internal_state (Dict[str, Any]): The final, integrated state from the UnifiedConsciousness.
            preferred_output_type (OutputType): The user's desired output format.

        Yields:
            OutputPayload: Chunks of the response as they are generated.
        """
        self.logger.info(f"Generating output stream with preferred type: {preferred_output_type.value}")
        
        try:
            # Extract the primary text content, usually from the creative mind's output
            # This needs to be robust to missing keys.
            final_states = internal_state.get('final_states', {})
            creative_state = final_states.get('creative', {})
            creative_payload_dict = creative_state.get('payload', {})
            text_content = creative_payload_dict.get('generated_text', "I have processed your input. However, I do not have a specific textual response to provide at this moment.")

            if preferred_output_type == OutputType.TEXT:
                async for chunk_payload in self._stream_text_output(text_content):
                    yield chunk_payload
            elif preferred_output_type == OutputType.AUDIO:
                # For audio, we might yield a "generating..." message first, then the audio data.
                yield OutputPayload(type=OutputType.TEXT, content="Generating audio response...", metadata={"status": "generating_audio"})
                audio_payload = await self._generate_audio_output(text_content) # This is a single payload
                yield audio_payload
            elif preferred_output_type in [OutputType.STRUCTURED, OutputType.ANALYSIS, OutputType.COMPLEX]:
                # These types are typically not streamed chunk by chunk.
                # We generate them fully and yield as a single payload.
                yield OutputPayload(type=OutputType.TEXT, content=f"Preparing {preferred_output_type.value} data...", metadata={"status": f"generating_{preferred_output_type.value}"})
                structured_payload = self._generate_structured_output(internal_state, output_type=preferred_output_type)
                yield structured_payload
            elif preferred_output_type == OutputType.ERROR: # Should be handled by caller, but as a fallback
                 yield self._generate_error_output(text_content or "An unspecified error occurred.")
            else:
                # Default to a text stream for any other/unknown type
                self.logger.warning(f"Unsupported preferred_output_type '{preferred_output_type}'. Defaulting to TEXT stream.")
                async for chunk_payload in self._stream_text_output(text_content):
                    yield chunk_payload

        except Exception as e:
            self.logger.error(f"Failed to generate output stream. Error: {e}", exc_info=True)
            yield self.generate_error_output(f"An error occurred during output generation: {e}")

    async def _stream_text_output(self, text_content: str) -> AsyncGenerator[OutputPayload, None]:
        """
        Generates a stream of text chunks from the full text content.
        This makes the AI appear more responsive by sending parts of the message as they are "thought of".
        """
        self.logger.debug(f"Streaming text content of length {len(text_content)}")
        if not text_content or not isinstance(text_content, str):
            self.logger.warning("Stream text output called with empty or invalid content.")
            yield OutputPayload(type=OutputType.TEXT, content="", metadata={"end_of_stream": True}) # Send an empty final chunk
            return

        # Use the NLP processor to split the text into sentences for more natural chunking.
        try:
            doc = self.nlp_processor.nlp(text_content)
            sentences = [sent.text.strip() for sent in doc.sents]
            if not sentences: # Handle cases where text might not form sentences (e.g., code snippets, short phrases)
                sentences = [text_content.strip()]
        except Exception as e:
            self.logger.error(f"Failed to split text into sentences for streaming, falling back to simple space split. Error: {e}")
            sentences = text_content.split('. ') # Fallback, less ideal

        total_sentences = len(sentences)
        for i, sentence in enumerate(sentences):
            if not sentence: continue
            
            # Add appropriate punctuation if it was split off, unless it's the last sentence and already has it.
            is_last_sentence = (i == total_sentences - 1)
            ends_with_punct = sentence.endswith(('.', '?', '!'))
            
            # Add a space after each sentence chunk for better client-side concatenation.
            content_to_send = sentence + " " 
            
            yield OutputPayload(
                type=OutputType.TEXT,
                content=content_to_send,
                metadata={
                    "chunk_index": i, 
                    "total_chunks": total_sentences, 
                    "end_of_stream": is_last_sentence
                }
            )
            # A small sleep yields control to the event loop, allowing the chunk to be sent.
            await asyncio.sleep(0.05) # Slightly increased for smoother perceived streaming

    def _generate_text_output(self, text_content: str) -> OutputPayload:
        """Generates a simple text output payload (non-streaming)."""
        summary = self.nlp_processor.summarize(text_content, ratio=0.1) or text_content[:150] # Ensure summary has content
        return OutputPayload(type=OutputType.TEXT, content=text_content, summary=summary)

    async def _generate_audio_output(self, text_content: str) -> OutputPayload:
        """Generates an audio output payload using an OFFLINE text-to-speech engine."""
        if not PYTTSX3_AVAILABLE:
            self.logger.warning("Audio output requested but pyttsx3 is not available. Falling back to text output.")
            text_payload = self._generate_text_output(text_content)
            text_payload.metadata['fallback_reason'] = "Offline Text-to-speech library (pyttsx3) not installed on server."
            text_payload.metadata['status'] = "completed_fallback"
            return text_payload
            
        self.logger.info("Generating offline text-to-speech audio for: '%s...'", text_content[:50])
        loop = asyncio.get_running_loop()
        try:
            # Run the blocking TTS generation in an executor thread
            audio_bytes = await loop.run_in_executor(None, self._blocking_tts_generation, text_content)
            if audio_bytes:
                self.logger.info(f"Successfully generated {len(audio_bytes)} bytes of audio.")
                return OutputPayload(type=OutputType.AUDIO, content=audio_bytes, summary=text_content[:150], metadata={"format": "WAV", "status": "completed"})
            else:
                raise RuntimeError("TTS generation returned empty audio bytes.")
        except Exception as e:
            self.logger.error(f"pyttsx3 failed to generate audio. Falling back to text output. Error: {e}", exc_info=True)
            text_payload = self._generate_text_output(text_content)
            text_payload.metadata['fallback_reason'] = f"Offline TTS generation failed: {e}"
            text_payload.metadata['status'] = "completed_fallback"
            return text_payload
    
    def _blocking_tts_generation(self, text: str) -> Optional[bytes]:
        """Synchronous helper function that generates TTS audio and returns the bytes."""
        # Ensure text is not empty
        if not text.strip():
            self.logger.warning("TTS generation called with empty text.")
            return None

        temp_dir = tempfile.gettempdir()
        # Generate a unique filename to avoid conflicts if multiple TTS ops run "concurrently" via executor
        temp_filename = os.path.join(temp_dir, f"prometheus_tts_{uuid.uuid4().hex}.wav")
        engine = None # Ensure engine is defined in this scope
        try:
            engine = pyttsx3.init()
            if engine is None: # Check if init() failed
                self.logger.error("pyttsx3.init() failed, engine is None.")
                return None
                
            # Optional: Configure voice, rate, volume if needed
            # voices = engine.getProperty('voices')
            # engine.setProperty('voice', voices[0].id) # Example: use the first available voice
            # engine.setProperty('rate', 150)
            
            engine.save_to_file(text, temp_filename)
            engine.runAndWait() # Blocks until speaking is complete

            # Check if the file was actually created and has content
            if os.path.exists(temp_filename) and os.path.getsize(temp_filename) > 0:
                with open(temp_filename, 'rb') as f:
                    audio_bytes = f.read()
                return audio_bytes
            else:
                self.logger.error(f"TTS engine did not create a valid audio file at {temp_filename}")
                return None
        except Exception as e:
            self.logger.error(f"Exception in _blocking_tts_generation: {e}", exc_info=True)
            return None # Return None on any exception
        finally:
            # Clean up the temporary file
            if os.path.exists(temp_filename):
                try:
                    os.remove(temp_filename)
                except Exception as e_rem:
                    self.logger.warning(f"Could not remove temporary TTS file {temp_filename}: {e_rem}")
            # It's good practice to stop the engine if it was started by this instance of the method.
            # However, pyttsx3's init() might be a singleton internally, so frequent init/stop might be inefficient
            # or problematic. For now, let's assume the engine handles its lifecycle okay across calls.


    def _generate_structured_output(self, internal_state: Dict[str, Any], output_type: OutputType = OutputType.STRUCTURED) -> OutputPayload:
        """Generates a structured JSON output of the AI's final internal state."""
        self.logger.info(f"Generating {output_type.value} JSON output.")
        
        content_to_serialize = internal_state # Default to the whole state
        
        if output_type == OutputType.ANALYSIS: # Be more selective for analysis
            content_to_serialize = {
                "summary_of_input_processing": internal_state.get("input_summary", "N/A"),
                "final_mind_states": internal_state.get("final_states", {}),
                "truth_evaluation": internal_state.get("truth_evaluation", {}),
                "ethical_assessment": internal_state.get("ethical_framework_output", {}) # Assuming this key exists
            }
        
        try:
            # Custom serializer to handle non-standard JSON types like Tensors
            def custom_serializer(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.cpu().numpy().tolist() # Convert tensor to list
                if isinstance(obj, (set, frozenset)):
                    return list(obj) # Convert set to list
                if hasattr(obj, 'isoformat'): # For datetime objects
                    return obj.isoformat()
                try:
                    # Fallback for other complex objects, convert to string representation
                    return str(obj)
                except Exception:
                    return f"<Object of type {type(obj).__name__} not directly serializable>"

            json_content_str = json.dumps(content_to_serialize, indent=2, default=custom_serializer)
            # The content for OutputPayload should ideally be the serializable dict itself,
            # but for direct display/copy, a string might be fine too.
            # Let's pass the dict.
            json_content_dict = json.loads(json_content_str) # Re-parse to ensure it's a valid dict

            summary = f"A {output_type.value} representation of the AI's cognitive state."
            if output_type == OutputType.ANALYSIS:
                summary = "Analysis of the AI's processing cycle."
                
            return OutputPayload(type=output_type, content=json_content_dict, summary=summary, metadata={"status": "completed"})
        except Exception as e:
            self.logger.error(f"Failed to serialize internal state to JSON for {output_type.value}. Error: {e}", exc_info=True)
            return self.generate_error_output(f"Failed to generate {output_type.value} output: {e}")
            
    def generate_error_output(self, error_message: str) -> OutputPayload: # Made public for direct use
        """Creates a standardized error output payload."""
        self.logger.error(f"Generating error output: {error_message}")
        return OutputPayload(type=OutputType.ERROR, content=error_message, summary="An error occurred.", metadata={"status": "error"})