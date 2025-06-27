# backend/io_systems/infinite_stream_processor.py

import asyncio
import json
from typing import AsyncGenerator, Dict, Any, Optional, List, Callable
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from backend.core.consciousness.infinite_context_manager import InfiniteContextManager
from backend.io_systems.io_types import InputPayload, OutputPayload, InputType, OutputType
from backend.utils.logger import Logger

@dataclass
class StreamingSession:
    """Represents an active streaming session."""
    session_id: str
    stream_id: str
    start_time: datetime
    total_input_tokens: int = 0
    total_output_tokens: int = 0
    active: bool = True
    metadata: Dict[str, Any] = field(default_factory=dict)
    input_buffer: str = ""
    output_buffer: str = ""
    context_manager_session: Optional[Any] = None

class InfiniteStreamProcessor:
    """
    Handles unlimited streaming input/output with intelligent chunking,
    parallel processing, and continuous learning.
    """
    
    def __init__(self, consciousness, context_manager: InfiniteContextManager):
        self.logger = Logger(__name__)
        self.consciousness = consciousness
        self.context_manager = context_manager
        
        # Active streaming sessions
        self.streaming_sessions: Dict[str, StreamingSession] = {}
        
        # Configuration
        self.chunk_size = 1000  # Base chunk size in characters
        self.parallel_streams = 4  # Number of parallel processing streams
        self.buffer_timeout = 2.0  # Seconds to wait before processing incomplete buffer
        
        # Learning components
        self.stream_patterns: Dict[str, List[Dict]] = {}
        self.processing_stats: Dict[str, Dict] = {}
        
        self.logger.info("InfiniteStreamProcessor initialized")
        
    async def create_streaming_session(self, session_id: str, metadata: Optional[Dict] = None) -> StreamingSession:
        """Create a new streaming session."""
        stream_id = f"stream_{uuid.uuid4().hex[:8]}"
        
        session = StreamingSession(
            session_id=session_id,
            stream_id=stream_id,
            start_time=datetime.utcnow(),
            metadata=metadata or {}
        )
        
        # Create context manager session
        session.context_manager_session = await self.context_manager.create_streaming_session(session_id)
        
        self.streaming_sessions[stream_id] = session
        
        self.logger.info(f"Created streaming session {stream_id} for {session_id}")
        return session
        
    async def stream_unlimited_input(
        self,
        stream_id: str,
        input_stream: AsyncGenerator[str, None],
        process_callback: Optional[Callable[[Dict], None]] = None
    ) -> AsyncGenerator[OutputPayload, None]:
        """
        Process unlimited streaming input and generate streaming output.
        
        This method can handle billions of tokens by:
        1. Chunking input intelligently
        2. Processing in parallel streams
        3. Maintaining context across chunks
        4. Generating continuous output
        """
        session = self.streaming_sessions.get(stream_id)
        if not session:
            yield OutputPayload(
                type=OutputType.ERROR,
                content="Invalid stream ID"
            )
            return
            
        try:
            # Create parallel processing tasks
            input_queue = asyncio.Queue(maxsize=100)
            output_queue = asyncio.Queue(maxsize=100)
            
            # Start parallel processors
            processors = []
            for i in range(self.parallel_streams):
                processor = asyncio.create_task(
                    self._process_stream_worker(
                        session, i, input_queue, output_queue, process_callback
                    )
                )
                processors.append(processor)
                
            # Start output generator
            output_task = asyncio.create_task(
                self._generate_streaming_output(session, output_queue)
            )
            
            # Process input stream
            chunk_buffer = ""
            chunk_count = 0
            
            async for input_chunk in input_stream:
                chunk_buffer += input_chunk
                session.total_input_tokens += len(input_chunk.split())
                
                # Check if we should process this chunk
                if self._should_process_chunk(chunk_buffer, chunk_count):
                    # Find sentence boundaries for clean chunking
                    sentences = self._extract_complete_sentences(chunk_buffer)
                    
                    if sentences['complete']:
                        # Queue complete sentences for processing
                        await input_queue.put({
                            'chunk_id': chunk_count,
                            'content': sentences['complete'],
                            'timestamp': datetime.utcnow()
                        })
                        
                        chunk_buffer = sentences['incomplete']
                        chunk_count += 1
                        
                    # Yield progress update
                    yield OutputPayload(
                        type=OutputType.TEXT,
                        content="",
                        metadata={
                            'progress': True,
                            'input_tokens': session.total_input_tokens,
                            'chunks_processed': chunk_count
                        }
                    )
                    
            # Process final buffer
            if chunk_buffer.strip():
                await input_queue.put({
                    'chunk_id': chunk_count,
                    'content': chunk_buffer,
                    'timestamp': datetime.utcnow(),
                    'final': True
                })
                
            # Signal end of input
            for _ in range(self.parallel_streams):
                await input_queue.put(None)
                
            # Wait for processors to complete
            await asyncio.gather(*processors)
            
            # Signal end of processing
            await output_queue.put(None)
            
            # Get final output chunks
            final_chunks = []
            while True:
                try:
                    chunk = output_queue.get_nowait()
                    if chunk is None:
                        break
                    final_chunks.append(chunk)
                except asyncio.QueueEmpty:
                    break
                    
            # Yield final output chunks
            for chunk in final_chunks:
                if isinstance(chunk, OutputPayload):
                    yield chunk
                    
            # Yield completion summary
            yield OutputPayload(
                type=OutputType.TEXT,
                content="\n\n[Processing Complete]",
                metadata={
                    'summary': await self._generate_session_summary(session),
                    'total_input_tokens': session.total_input_tokens,
                    'total_output_tokens': session.total_output_tokens,
                    'duration': (datetime.utcnow() - session.start_time).total_seconds()
                }
            )
            
        except Exception as e:
            self.logger.error(f"Error in streaming processing: {e}", exc_info=True)
            yield OutputPayload(
                type=OutputType.ERROR,
                content=f"Streaming error: {str(e)}"
            )
        finally:
            # Cleanup
            session.active = False
            
    async def _process_stream_worker(
        self,
        session: StreamingSession,
        worker_id: int,
        input_queue: asyncio.Queue,
        output_queue: asyncio.Queue,
        process_callback: Optional[Callable]
    ):
        """Worker task for parallel stream processing."""
        self.logger.debug(f"Stream worker {worker_id} started for session {session.stream_id}")
        
        while True:
            try:
                # Get chunk from queue
                chunk_data = await input_queue.get()
                if chunk_data is None:
                    break
                    
                # Process chunk through consciousness
                context = await self.context_manager.get_relevant_context(
                    session.session_id,
                    chunk_data['content'],
                    max_context_tokens=4096
                )
                
                # Build processing context
                processing_context = {
                    'chunk': chunk_data,
                    'context': context,
                    'session': session,
                    'worker_id': worker_id
                }
                
                # Process through consciousness with streaming context
                result = await self._process_with_context(processing_context)
                
                # Generate output for this chunk
                output_chunks = await self._generate_chunk_output(result, chunk_data)
                
                # Queue output chunks
                for output_chunk in output_chunks:
                    await output_queue.put(output_chunk)
                    session.total_output_tokens += len(output_chunk.content.split())
                    
                # Callback for monitoring
                if process_callback:
                    process_callback({
                        'worker_id': worker_id,
                        'chunk_id': chunk_data['chunk_id'],
                        'status': 'processed'
                    })
                    
            except Exception as e:
                self.logger.error(f"Worker {worker_id} error: {e}", exc_info=True)
                await output_queue.put(OutputPayload(
                    type=OutputType.ERROR,
                    content=f"Processing error in worker {worker_id}"
                ))
                
        self.logger.debug(f"Stream worker {worker_id} completed")
        
    async def _generate_streaming_output(
        self,
        session: StreamingSession,
        output_queue: asyncio.Queue
    ) -> AsyncGenerator[OutputPayload, None]:
        """Generate streaming output from the output queue."""
        buffer = []
        last_output_time = datetime.utcnow()
        
        while True:
            try:
                # Try to get output with timeout
                output = await asyncio.wait_for(
                    output_queue.get(),
                    timeout=0.1
                )
                
                if output is None:
                    # End of stream
                    if buffer:
                        # Flush remaining buffer
                        yield self._combine_output_chunks(buffer)
                    break
                    
                buffer.append(output)
                
                # Check if we should yield buffered output
                if len(buffer) >= 5 or (datetime.utcnow() - last_output_time).total_seconds() > 0.5:
                    if buffer:
                        yield self._combine_output_chunks(buffer)
                        buffer = []
                        last_output_time = datetime.utcnow()
                        
            except asyncio.TimeoutError:
                # Check if we should flush buffer due to timeout
                if buffer and (datetime.utcnow() - last_output_time).total_seconds() > 1.0:
                    yield self._combine_output_chunks(buffer)
                    buffer = []
                    last_output_time = datetime.utcnow()
                    
    async def _process_with_context(self, processing_context: Dict[str, Any]) -> Dict[str, Any]:
        """Process a chunk with its retrieved context through consciousness."""
        chunk_data = processing_context['chunk']
        context_data = processing_context['context']
        
        # Build enriched prompt with multi-scale context
        enriched_prompt = self._build_enriched_prompt(chunk_data['content'], context_data)
        
        # Process through consciousness
        result = await self.consciousness.process_input(
            input_text=enriched_prompt,
            session_id=processing_context['session']['session_id']
        )
        
        # Update streaming context
        await self.context_manager.stream_input(
            processing_context['session']['session_id'],
            self._async_text_generator(chunk_data['content'])
        )
        
        return result
        
    async def _generate_chunk_output(
        self,
        processing_result: Dict[str, Any],
        chunk_data: Dict[str, Any]
    ) -> List[OutputPayload]:
        """Generate output chunks for a processed input chunk."""
        output_chunks = []
        
        if processing_result.get('status') == 'success':
            output_text = processing_result.get('output', '')
            
            # Split output into smaller chunks for streaming
            sentences = self._split_into_sentences(output_text)
            
            for i, sentence in enumerate(sentences):
                output_chunks.append(OutputPayload(
                    type=OutputType.TEXT,
                    content=sentence + ' ',
                    metadata={
                        'chunk_id': chunk_data['chunk_id'],
                        'sentence_index': i,
                        'confidence': processing_result.get('final_states', {}).get('creative', {}).get('confidence', 0.5)
                    }
                ))
                
        elif processing_result.get('status') == 'blocked':
            output_chunks.append(OutputPayload(
                type=OutputType.TEXT,
                content=processing_result.get('output', 'Content blocked by safety filters.'),
                metadata={'blocked': True, 'chunk_id': chunk_data['chunk_id']}
            ))
        else:
            output_chunks.append(OutputPayload(
                type=OutputType.ERROR,
                content=f"Error processing chunk {chunk_data['chunk_id']}"
            ))
            
        return output_chunks
        
    def _should_process_chunk(self, buffer: str, chunk_count: int) -> bool:
        """Determine if the current buffer should be processed."""
        # Process if buffer is large enough
        if len(buffer) > self.chunk_size:
            return True
            
        # Process if we have complete sentences
        if buffer.count('.') > 3 or buffer.count('?') > 1 or buffer.count('!') > 1:
            return True
            
        # Process periodically even with small buffers
        if chunk_count % 10 == 0 and len(buffer) > 100:
            return True
            
        return False
        
    def _extract_complete_sentences(self, text: str) -> Dict[str, str]:
        """Extract complete sentences from text, returning complete and incomplete parts."""
        # Simple sentence boundary detection
        # In production, use more sophisticated NLP
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?' and len(current) > 1:
                sentences.append(current.strip())
                current = ""
                
        return {
            'complete': ' '.join(sentences),
            'incomplete': current.strip()
        }
        
    def _build_enriched_prompt(self, chunk_content: str, context_data: Dict[str, Any]) -> str:
        """Build an enriched prompt with multi-scale context."""
        prompt_parts = []
        
        # Add context summary if available
        if context_data.get('context_summary'):
            prompt_parts.append(f"[Context Summary]\n{context_data['context_summary']}\n")
            
        # Add relevant context excerpts
        if context_data.get('relevant_context'):
            prompt_parts.append("[Relevant Context]")
            for ctx in context_data['relevant_context'][:3]:  # Top 3
                prompt_parts.append(f"- {ctx['content'][:200]}...")
            prompt_parts.append("")
            
        # Add the current chunk
        prompt_parts.append(f"[Current Input]\n{chunk_content}")
        
        return '\n'.join(prompt_parts)
        
    def _combine_output_chunks(self, chunks: List[OutputPayload]) -> OutputPayload:
        """Combine multiple output chunks into a single payload."""
        if not chunks:
            return OutputPayload(type=OutputType.TEXT, content="")
            
        # Combine text content
        combined_text = ''.join(chunk.content for chunk in chunks if chunk.type == OutputType.TEXT)
        
        # Merge metadata
        combined_metadata = {}
        for chunk in chunks:
            if chunk.metadata:
                combined_metadata.update(chunk.metadata)
                
        return OutputPayload(
            type=OutputType.TEXT,
            content=combined_text,
            metadata=combined_metadata
        )
        
    def _split_into_sentences(self, text: str) -> List[str]:
        """Split text into sentences for streaming output."""
        # Simple implementation - in production use proper NLP
        sentences = []
        current = ""
        
        for char in text:
            current += char
            if char in '.!?' and len(current) > 1:
                sentences.append(current.strip())
                current = ""
                
        if current.strip():
            sentences.append(current.strip())
            
        return sentences
        
    async def _async_text_generator(self, text: str) -> AsyncGenerator[str, None]:
        """Convert text to async generator for streaming."""
        # Split into words for streaming
        words = text.split()
        for i in range(0, len(words), 10):
            chunk = ' '.join(words[i:i+10])
            yield chunk + ' '
            await asyncio.sleep(0.001)  # Small delay to simulate streaming
            
    async def _generate_session_summary(self, session: StreamingSession) -> Dict[str, Any]:
        """Generate a comprehensive summary of the streaming session."""
        # Get context summary from context manager
        context_export = await self.context_manager.export_session_context(session.session_id)
        
        summary = {
            'session_id': session.session_id,
            'stream_id': session.stream_id,
            'duration_seconds': (datetime.utcnow() - session.start_time).total_seconds(),
            'total_input_tokens': session.total_input_tokens,
            'total_output_tokens': session.total_output_tokens,
            'compression_ratio': session.total_output_tokens / max(session.total_input_tokens, 1),
            'context_checkpoints': len(context_export.get('checkpoints', [])),
            'memory_stats': context_export.get('memory_stats', {})
        }
        
        return summary
        
    async def get_stream_analytics(self, stream_id: str) -> Dict[str, Any]:
        """Get detailed analytics for a streaming session."""
        session = self.streaming_sessions.get(stream_id)
        if not session:
            return {"error": "Stream not found"}
            
        analytics = {
            'stream_id': stream_id,
            'session_id': session.session_id,
            'status': 'active' if session.active else 'completed',
            'start_time': session.start_time.isoformat(),
            'duration_seconds': (datetime.utcnow() - session.start_time).total_seconds(),
            'tokens': {
                'input': session.total_input_tokens,
                'output': session.total_output_tokens,
                'ratio': session.total_output_tokens / max(session.total_input_tokens, 1)
            },
            'processing_stats': self.processing_stats.get(stream_id, {}),
            'context_coverage': await self._calculate_context_coverage(session)
        }
        
        return analytics
        
    async def _calculate_context_coverage(self, session: StreamingSession) -> Dict[str, float]:
        """Calculate how well the context system covered the input."""
        if not session.context_manager_session:
            return {'coverage': 0.0}
            
        # This would analyze how much of the input was successfully retained
        # and made available for context retrieval
        return {
            'immediate_coverage': 1.0,  # Last 100 tokens always available
            'recent_coverage': 0.9,     # Compressed recent context
            'extended_coverage': 0.7,   # Checkpoint-based coverage
            'global_coverage': 0.5      # High-level summary coverage
        }