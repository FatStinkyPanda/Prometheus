# backend/io_systems/stream_manager.py

from typing import Dict, Any, Optional, TYPE_CHECKING, AsyncGenerator

# Use TYPE_CHECKING to import UnifiedConsciousness only for type hinting.
if TYPE_CHECKING:
    from backend.core.consciousness.unified_consciousness import UnifiedConsciousness

from backend.io_systems.io_types import InputPayload, OutputPayload, InputType, OutputType
from backend.io_systems.multimodal_input import MultimodalInput
from backend.io_systems.natural_language_processor import NaturalLanguageProcessor
from backend.io_systems.output_generator import OutputGenerator
from backend.utils.logger import Logger


class StreamManager:
    """
    Manages the end-to-end flow of data into and out of the UnifiedConsciousness.

    This class acts as the primary I/O controller, orchestrating the preprocessing
    of input, the invocation of the core consciousness, and the generation of the
    final output. This version is updated to support streaming output generation.
    """

    def __init__(self,
                 consciousness: 'UnifiedConsciousness',
                 nlp_processor: NaturalLanguageProcessor,
                 multimodal_handler: MultimodalInput,
                 output_generator: OutputGenerator):
        """
        Initializes the StreamManager.

        Args:
            consciousness (UnifiedConsciousness): The main consciousness engine.
            nlp_processor (NaturalLanguageProcessor): The NLP handler.
            multimodal_handler (MultimodalInput): The handler for non-text inputs.
            output_generator (OutputGenerator): The handler for generating final output.
        """
        self.logger = Logger(__name__)
        self.consciousness = consciousness
        self.nlp_processor = nlp_processor
        self.multimodal_handler = multimodal_handler
        self.output_generator = output_generator
        self.logger.info("StreamManager initialized and linked to I/O components and Consciousness.")

    async def handle_stream(self, raw_request: Dict[str, Any]) -> AsyncGenerator[OutputPayload, None]:
        """
        Processes a raw input request through the entire system, yielding a stream of
        output payloads as they are generated.

        Args:
            raw_request (Dict[str, Any]): A dictionary containing the input details.
                                          Expected keys: 'type', 'content', 'session_id',
                                          'metadata' (optional), 'options' (optional).

        Yields:
            OutputPayload: Chunks of the final response as they become available.
                           This could be text chunks, status updates, or final structured data.
        """
        session_id = raw_request.get('session_id', 'unknown_session')
        self.logger.info(f"Handling new stream request for session: {session_id}, type: {raw_request.get('type')}")
        
        input_payload: Optional[InputPayload] = None
        try:
            # 1. Standardize the input into a structured payload
            try:
                input_payload = InputPayload(
                    type=InputType(raw_request.get('type', InputType.TEXT.value)), # Default to TEXT if not provided
                    content=raw_request.get('content'), # Content can be None for some types initially
                    session_id=session_id,
                    metadata=raw_request.get('metadata', {}),
                    options=raw_request.get('options', {})
                )
                if input_payload.content is None and input_payload.type == InputType.TEXT:
                    self.logger.warning(f"InputPayload TEXT type has None content for session {session_id}. Yielding error.")
                    yield self.output_generator.generate_error_output("Input content is missing for TEXT type.")
                    return

            except (ValueError, TypeError) as e:
                self.logger.error(f"Failed to create valid InputPayload from raw request for session {session_id}: {e}", exc_info=True)
                yield self.output_generator.generate_error_output(f"Invalid input request structure: {e}")
                return

            # 2. Pre-process the input to extract text
            text_for_consciousness = ""
            if input_payload.type == InputType.TEXT:
                if isinstance(input_payload.content, str):
                    text_for_consciousness = input_payload.content
                else:
                    self.logger.error(f"InputType.TEXT received non-string content for session {session_id}.")
                    yield self.output_generator.generate_error_output("TEXT input must have string content.")
                    return
            elif input_payload.content is not None: # Only process multimodal if content exists
                multimodal_result = await self.multimodal_handler.process_input(input_payload)
                if multimodal_result and 'text_content' in multimodal_result:
                    text_for_consciousness = multimodal_result['text_content']
                elif multimodal_result and 'error' in multimodal_result:
                     self.logger.warning(f"Multimodal input processing failed for session {session_id}: {multimodal_result['error']}")
                     yield self.output_generator.generate_error_output(f"Input processing failed: {multimodal_result['error']}")
                     return
                else: # Multimodal processed but no text or error (e.g., image without OCR capability)
                    text_for_consciousness = f"[{input_payload.type.value} input received, content analysis pending further development]"
            else: # Non-TEXT type with no content
                self.logger.warning(f"Input type {input_payload.type.value} received with no content for session {session_id}.")
                text_for_consciousness = f"[{input_payload.type.value} input received without content]"


            if not text_for_consciousness.strip(): # Check if it's empty or just whitespace
                self.logger.warning(f"Input processing resulted in effectively no text for consciousness for session {session_id}.")
                # Yield a mild informational message rather than a harsh error if it's just empty.
                yield OutputPayload(type=OutputType.TEXT, content="Input received, but no specific textual content to process.", metadata={"status": "empty_input"})
                return
            
            # 3. Send the processed text to the core consciousness engine
            self.logger.debug(f"Sending text to UnifiedConsciousness for session {session_id}: '{text_for_consciousness[:100]}...'")
            consciousness_response = await self.consciousness.process_input(
                input_text=text_for_consciousness,
                session_id=input_payload.session_id
            )

            # 4. Generate the final output payload as a stream
            if not consciousness_response: # Should not happen if process_input is robust
                self.logger.error(f"UnifiedConsciousness.process_input returned None for session {session_id}.")
                yield self.output_generator.generate_error_output("Core consciousness processing failed to return a response.")
                return

            if consciousness_response.get("status") == "error":
                error_msg = consciousness_response.get('output', "An unknown error occurred in the consciousness engine.")
                self.logger.error(f"Consciousness engine returned an error for session {session_id}: {error_msg}")
                yield self.output_generator.generate_error_output(error_msg)
                return
                
            # Determine preferred output type from options, default to TEXT
            preferred_output_type_str = input_payload.options.get('preferred_output_type', OutputType.TEXT.value)
            try:
                preferred_output_type = OutputType(preferred_output_type_str)
            except ValueError:
                self.logger.warning(f"Invalid preferred_output_type '{preferred_output_type_str}' for session {session_id}. Defaulting to TEXT.")
                preferred_output_type = OutputType.TEXT
            
            self.logger.debug(f"Piping consciousness response to OutputGenerator stream for session {session_id} with preferred type '{preferred_output_type}'.")
            
            stream_had_content = False
            async for output_chunk in self.output_generator.generate_output_stream(
                internal_state=consciousness_response,
                preferred_output_type=preferred_output_type
            ):
                yield output_chunk
                if output_chunk.content and output_chunk.type != OutputType.ERROR:
                    stream_had_content = True
            
            if not stream_had_content:
                self.logger.info(f"OutputGenerator stream completed for session {session_id} but yielded no content payloads (excluding errors).")
                # Optionally, yield a final "Process complete, no further output" message if desired.
                # For now, if there's no content, the stream just ends.

        except Exception as e:
            self.logger.critical(f"An unhandled exception occurred in StreamManager.handle_stream for session {session_id}: {e}", exc_info=True)
            yield self.output_generator.generate_error_output(f"A critical internal error occurred in the stream manager: {e}")