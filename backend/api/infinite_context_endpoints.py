# backend/api/infinite_context_endpoints.py

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException, BackgroundTasks
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List, Union # Added Union
import asyncio
import json
import uuid
from datetime import datetime
from collections import deque # Added for InfiniteContextMonitor

# Import base UnifiedConsciousness for type checking
from backend.core.consciousness.unified_consciousness import UnifiedConsciousness
# Import InfiniteConsciousness wrapper
from backend.core.consciousness.infinite_context_integration import InfiniteConsciousness
from backend.utils.logger import Logger


# Pydantic models for API
class InfiniteInputRequest(BaseModel):
    """Request model for processing unlimited input."""
    session_id: Optional[str] = Field(None, description="Session ID for context continuity")
    process_incrementally: bool = Field(True, description="Process and respond incrementally")
    chunk_size: int = Field(1000, description="Characters per processing chunk")
    
class InfiniteGenerationRequest(BaseModel):
    """Request model for unlimited generation."""
    prompt: str = Field(..., description="Initial prompt for generation")
    session_id: Optional[str] = Field(None, description="Session ID for context")
    max_tokens: Optional[int] = Field(None, description="Maximum tokens to generate (None for unlimited)")
    temperature: float = Field(0.7, description="Generation temperature")
    stop_sequences: List[str] = Field(default_factory=list, description="Sequences to stop generation")
    
class ContextQueryRequest(BaseModel):
    """Request model for querying with infinite context."""
    query: str = Field(..., description="Query text")
    session_id: str = Field(..., description="Session ID")
    max_context_tokens: int = Field(16384, description="Maximum context tokens to retrieve")
    include_semantic_links: bool = Field(True, description="Include semantically linked content")


class InfiniteContextAPI:
    """API endpoints for infinite context processing."""
    
    def __init__(self, consciousness_instance: Union[UnifiedConsciousness, InfiniteConsciousness]):
        self.logger = Logger(__name__)
        
        # --- MODIFICATION START ---
        # Check if the provided instance is already an InfiniteConsciousness wrapper
        if isinstance(consciousness_instance, InfiniteConsciousness):
            self.logger.info("InfiniteContextAPI received an already wrapped InfiniteConsciousness instance.")
            self.infinite_consciousness: InfiniteConsciousness = consciousness_instance
            # The underlying UnifiedConsciousness is accessed via self.infinite_consciousness.consciousness
        elif isinstance(consciousness_instance, UnifiedConsciousness):
            self.logger.info("InfiniteContextAPI received a base UnifiedConsciousness instance. Wrapping it.")
            # This is the original behavior: wrap the base UnifiedConsciousness
            self.infinite_consciousness: InfiniteConsciousness = InfiniteConsciousness(consciousness_instance)
        else:
            # This should ideally not happen if type hints are respected by callers.
            self.logger.critical(
                f"InfiniteContextAPI received an unexpected type for consciousness_instance: {type(consciousness_instance)}. "
                "This will likely lead to errors."
            )
            # Attempt a fallback, but this is problematic.
            # Cast to InfiniteConsciousness which would fail if it's not compatible.
            # Or, raise an error to halt misconfiguration.
            raise TypeError(
                "InfiniteContextAPI expects either UnifiedConsciousness or InfiniteConsciousness instance."
            )
        # --- MODIFICATION END ---
            
        # self.consciousness (the base UnifiedConsciousness) is now consistently accessible via
        # self.infinite_consciousness.consciousness
        
        self.router = APIRouter(prefix="/api/v1/infinite", tags=["Infinite Context"])
        self._setup_routes()
        
        self.active_websockets: Dict[str, WebSocket] = {}
        self.active_generations: Dict[str, asyncio.Task] = {}
        
    def _setup_routes(self):
        """Setup all API routes."""
        
        @self.router.websocket("/ws/stream-input/{session_id}")
        async def websocket_stream_input(websocket: WebSocket, session_id: str):
            await self._handle_websocket_input(websocket, session_id)
            
        @self.router.websocket("/ws/stream-output/{session_id}")
        async def websocket_stream_output(websocket: WebSocket, session_id: str):
            await self._handle_websocket_output(websocket, session_id)
            
        @self.router.post("/process-document")
        async def process_document(
            background_tasks: BackgroundTasks,
            file_content: str, # Assuming file content is sent as a large string in the request body
            request: InfiniteInputRequest # Using Pydantic model for other params
        ):
            session_id = request.session_id or str(uuid.uuid4())
            
            background_tasks.add_task(
                self._process_document_background,
                file_content,
                session_id,
                request.process_incrementally,
                request.chunk_size # Pass chunk_size from request
            )
            
            return {
                "session_id": session_id,
                "status": "processing_started",
                "message": "Document processing started in background. Connect to WebSocket or query session for updates."
            }
            
        @self.router.post("/generate")
        async def generate_unlimited(request: InfiniteGenerationRequest):
            session_id = request.session_id or str(uuid.uuid4())
            
            return StreamingResponse(
                self._generate_stream(request, session_id),
                media_type="text/event-stream"
            )
            
        @self.router.post("/query")
        async def query_with_context(request: ContextQueryRequest):
            try:
                response = await self.infinite_consciousness.query_with_infinite_context(
                    query=request.query,
                    session_id=request.session_id,
                    max_context_tokens=request.max_context_tokens
                    # include_semantic_links is handled by EnhancedInfiniteContextManager's default or method args
                )
                return response
            except Exception as e:
                self.logger.error(f"Error in context query for session '{request.session_id}': {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error during context query: {str(e)}")
                
        @self.router.get("/session/{session_id}/summary")
        async def get_session_summary(session_id: str, include_learning: bool = True):
            try:
                summary = await self.infinite_consciousness.get_session_summary(
                    session_id,
                    include_learning_insights=include_learning
                )
                return summary
            except Exception as e:
                self.logger.error(f"Error getting session summary for '{session_id}': {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error getting session summary: {str(e)}")
                
        @self.router.get("/stats")
        async def get_system_stats():
            try:
                # Accessing stats via the infinite_consciousness wrapper
                return self.infinite_consciousness.infinite_context.get_statistics()
            except Exception as e:
                self.logger.error(f"Error getting system stats: {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error retrieving system statistics: {str(e)}")
            
        @self.router.delete("/session/{session_id}")
        async def clear_session(session_id: str):
            try:
                # Accessing clear method via the infinite_consciousness wrapper
                await self.infinite_consciousness.infinite_context.clear(session_id)
                return {"status": "success", "message": f"Session {session_id} context cleared."}
            except Exception as e:
                self.logger.error(f"Error clearing session '{session_id}': {e}", exc_info=True)
                raise HTTPException(status_code=500, detail=f"Error clearing session: {str(e)}")
    
    async def _handle_websocket_input(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        ws_client_id = f"{session_id}_{uuid.uuid4().hex[:6]}"
        self.active_websockets[ws_client_id] = websocket
        self.logger.info(f"WebSocket input stream connected: {ws_client_id} for session {session_id}")
        
        try:
            async def websocket_generator():
                while True:
                    try:
                        data = await websocket.receive_text()
                        if data == "[END_STREAM]": # Define a clear end-of-stream signal
                            self.logger.info(f"WebSocket input stream {ws_client_id} received END_STREAM signal.")
                            break
                        yield data
                    except WebSocketDisconnect:
                        self.logger.info(f"WebSocket input stream {ws_client_id} disconnected by client.")
                        break
                    except Exception as e_ws_recv:
                        self.logger.error(f"Error receiving data on WebSocket {ws_client_id}: {e_ws_recv}", exc_info=True)
                        await websocket.send_json({"type": "error", "error": f"WebSocket receive error: {str(e_ws_recv)}"})
                        break
                        
            async for update in self.infinite_consciousness.process_unlimited_input(
                websocket_generator(),
                session_id,
                process_incrementally=True # Usually desired for WebSocket interaction
            ):
                try:
                    await websocket.send_json(update)
                except Exception as e_ws_send: # Catch errors during send
                    self.logger.error(f"Error sending update on WebSocket {ws_client_id}: {e_ws_send}", exc_info=True)
                    break # Stop trying to send if connection is broken
                
        except Exception as e:
            self.logger.error(f"Overall error in WebSocket input handler {ws_client_id}: {e}", exc_info=True)
            try:
                await websocket.send_json({"type": "error", "error": f"Server-side processing error: {str(e)}"})
            except: pass # Ignore if send also fails
        finally:
            if ws_client_id in self.active_websockets:
                del self.active_websockets[ws_client_id]
            if websocket.client_state != WebSocketDisconnect: # Check if already disconnected
                try: await websocket.close()
                except: pass
            self.logger.info(f"WebSocket input stream closed: {ws_client_id}")
    
    async def _handle_websocket_output(self, websocket: WebSocket, session_id: str):
        await websocket.accept()
        generation_id = f"gen_{session_id}_{uuid.uuid4().hex[:6]}"
        self.logger.info(f"WebSocket output stream connected: {generation_id} for session {session_id}")
        
        generation_task: Optional[asyncio.Task] = None
        try:
            request_data_str = await websocket.receive_text()
            request_data = json.loads(request_data_str)
            
            # Validate request_data using Pydantic model if it's complex
            # For simplicity, assuming it contains 'prompt', 'max_tokens', 'temperature'
            
            generation_task = asyncio.create_task(
                self._generate_for_websocket(websocket, request_data, session_id, generation_id)
            )
            self.active_generations[generation_id] = generation_task
            await generation_task
            
        except WebSocketDisconnect:
            self.logger.info(f"WebSocket output stream {generation_id} disconnected by client.")
            if generation_task and not generation_task.done():
                generation_task.cancel()
        except json.JSONDecodeError:
            err_msg = "Invalid JSON request for generation."
            self.logger.warning(f"WebSocket output stream {generation_id}: {err_msg}")
            try: await websocket.send_json({"type": "error", "error": err_msg})
            except: pass
        except Exception as e:
            self.logger.error(f"Overall error in WebSocket output handler {generation_id}: {e}", exc_info=True)
            try: await websocket.send_json({"type": "error", "error": f"Server-side generation error: {str(e)}"})
            except: pass
        finally:
            if generation_id in self.active_generations:
                task = self.active_generations.pop(generation_id)
                if task and not task.done():
                    task.cancel()
            if websocket.client_state != WebSocketDisconnect:
                try: await websocket.close()
                except: pass
            self.logger.info(f"WebSocket output stream closed: {generation_id}")
    
    async def _generate_for_websocket(
        self,
        websocket: WebSocket,
        request_data: Dict[str, Any],
        session_id: str,
        generation_id: str # For logging/tracking
    ):
        chunk_buffer = []
        chunk_size_words = request_data.get('chunk_size_words', 50) # Words per WebSocket message
        
        try:
            async for chunk in self.infinite_consciousness.generate_unlimited_output(
                prompt=request_data.get('prompt', ''),
                session_id=session_id,
                max_tokens=request_data.get('max_tokens'),
                temperature=request_data.get('temperature', 0.7),
                stream=True # Ensure streaming is enabled
            ):
                chunk_buffer.append(chunk)
                
                if len(' '.join(chunk_buffer).split()) >= chunk_size_words:
                    await websocket.send_json({
                        "type": "output_chunk",
                        "generation_id": generation_id,
                        "content": ''.join(chunk_buffer),
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    chunk_buffer = []
            
            if chunk_buffer: # Send any remaining content
                await websocket.send_json({
                    "type": "output_chunk",
                    "generation_id": generation_id,
                    "content": ''.join(chunk_buffer),
                    "timestamp": datetime.utcnow().isoformat()
                })
            
            await websocket.send_json({
                "type": "generation_complete",
                "generation_id": generation_id,
                "timestamp": datetime.utcnow().isoformat()
            })
        except asyncio.CancelledError:
            self.logger.info(f"Generation task {generation_id} was cancelled.")
        except Exception as e_gen:
            self.logger.error(f"Error during WebSocket generation {generation_id}: {e_gen}", exc_info=True)
            try: await websocket.send_json({"type": "error", "generation_id": generation_id, "error": f"Generation error: {str(e_gen)}"})
            except: pass # Ignore if send fails
    
    async def _generate_stream(
        self,
        request: InfiniteGenerationRequest,
        session_id: str
    ):
        self.logger.info(f"Starting SSE generation for session {session_id}, prompt: '{request.prompt[:50]}...'")
        try:
            async for chunk in self.infinite_consciousness.generate_unlimited_output(
                prompt=request.prompt,
                session_id=session_id,
                max_tokens=request.max_tokens,
                temperature=request.temperature,
                stream=True # Essential for SSE
            ):
                # SSE format: data: <json_payload>\n\n
                # Stop sequence check
                should_stop = False
                current_chunk_content = chunk
                if request.stop_sequences:
                    for stop_seq in request.stop_sequences:
                        if stop_seq in current_chunk_content:
                            current_chunk_content = current_chunk_content.split(stop_seq, 1)[0]
                            should_stop = True
                            break
                
                if current_chunk_content: # Send content if any after potential trimming
                    yield f"data: {json.dumps({'chunk': current_chunk_content})}\n\n"
                
                if should_stop:
                    self.logger.info(f"Stop sequence triggered during SSE generation for session {session_id}.")
                    yield f"data: {json.dumps({'done': True, 'reason': 'stop_sequence_met'})}\n\n"
                    break
            
            # If loop finishes without breaking due to stop_sequence, send a final done message
            if not should_stop: # Add this check to prevent double 'done'
                 yield f"data: {json.dumps({'done': True, 'reason': 'generation_completed'})}\n\n"

        except Exception as e:
            self.logger.error(f"SSE generation stream error for session {session_id}: {e}", exc_info=True)
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
        self.logger.info(f"SSE generation finished for session {session_id}.")

    async def _process_document_background(
        self,
        content: str,
        session_id: str,
        process_incrementally: bool,
        requested_chunk_size: int
    ):
        self.logger.info(f"Background document processing started for session {session_id}. Incremental: {process_incrementally}. Chunk size: {requested_chunk_size}")
        try:
            async def content_generator():
                # Use the chunk_size from the request if valid, otherwise default
                chunk_size = requested_chunk_size if requested_chunk_size > 0 else 1000
                for i in range(0, len(content), chunk_size):
                    yield content[i:i + chunk_size]
                    await asyncio.sleep(0.001) # Yield control briefly for very large docs
            
            async for update in self.infinite_consciousness.process_unlimited_input(
                content_generator(),
                session_id,
                process_incrementally
            ):
                # Optionally, send updates via WebSocket if a connection for this session_id exists
                # This requires a mechanism to map session_id to active WebSockets.
                # For simplicity, this example assumes updates are logged or stored for later query.
                if session_id in self.active_websockets:
                    websocket = self.active_websockets[session_id]
                    try:
                        await websocket.send_json(update)
                    except Exception as e_ws:
                        self.logger.warning(f"Failed to send background processing update to WebSocket for session {session_id}: {e_ws}")
                
                if update.get('type') == 'error':
                    self.logger.error(f"Error during background document processing for session {session_id}: {update.get('error')}")
                    break # Stop processing on error
        except Exception as e:
            self.logger.error(f"Critical error in background document processing for session {session_id}: {e}", exc_info=True)
        self.logger.info(f"Background document processing finished for session {session_id}.")


# --- InfiniteContextMonitor and Client Example classes remain largely unchanged ---
# --- but should use self.infinite_consciousness for interaction if they were part of InfiniteContextAPI ---
# --- Since they are utility/example classes, they are fine as is for now or could be moved. ---

class InfiniteContextMonitor:
    """Monitor and optimize infinite context performance."""
    
    def __init__(self, infinite_consciousness_instance: InfiniteConsciousness): # Takes IC instance
        self.infinite_consciousness = infinite_consciousness_instance # Stores IC instance
        self.logger = Logger(__name__)
        self.metrics = {
            'processing_rate': deque(maxlen=100),
            'compression_efficiency': deque(maxlen=100),
            'retrieval_latency': deque(maxlen=100),
            'memory_usage': deque(maxlen=100)
        }
        self._last_stats: Optional[Dict[str, Any]] = None # Initialize _last_stats
        
    async def start_monitoring(self, interval: int = 60):
        """Start periodic monitoring."""
        self.logger.info(f"InfiniteContextMonitor started with interval {interval}s.")
        while True:
            try:
                await self._collect_metrics()
                await asyncio.sleep(interval)
            except asyncio.CancelledError:
                self.logger.info("InfiniteContextMonitor monitoring task cancelled.")
                break
            except Exception as e:
                self.logger.error(f"Monitoring error in InfiniteContextMonitor: {e}", exc_info=True)
                await asyncio.sleep(interval * 2) # Wait longer on error
                
    async def _collect_metrics(self):
        stats = self.infinite_consciousness.infinite_context.get_statistics()
        
        if self._last_stats is not None:
            time_delta = 60  # Assuming 60 second interval
            tokens_delta = stats['total_tokens_processed'] - self._last_stats.get('total_tokens_processed', 0)
            processing_rate = tokens_delta / time_delta if time_delta > 0 else 0
            self.metrics['processing_rate'].append(processing_rate)
        else: # First run
            self.metrics['processing_rate'].append(0) 
            
        self.metrics['compression_efficiency'].append(stats.get('compression_ratio', 1.0))
        self.metrics['memory_usage'].append(stats.get('memory_usage_mb', 0))
        
        self._last_stats = stats
        
        await self._check_optimizations()
        
    async def _check_optimizations(self):
        if not self.metrics['processing_rate']: return
            
        avg_rate = sum(self.metrics['processing_rate']) / len(self.metrics['processing_rate'])
        if avg_rate < 1000 and len(self.metrics['processing_rate']) == self.metrics['processing_rate'].maxlen:
            self.logger.warning(f"Low processing rate observed: {avg_rate:.1f} tokens/sec. Consider performance review.")
            
        if self.metrics['memory_usage']:
            current_memory = self.metrics['memory_usage'][-1]
            if current_memory > 2000:  # Example: More than 2GB
                self.logger.warning(f"High memory usage for infinite context: {current_memory:.1f} MB. Triggering optimization.")
                await self._trigger_compression()
                
    async def _trigger_compression(self):
        self.logger.info("InfiniteContextMonitor triggering on-demand context compression due to high memory usage.")
        await self.infinite_consciousness.infinite_context._compress_old_blocks() # Access internal method
        
    def get_performance_report(self) -> Dict[str, Any]:
        def safe_avg(data): return sum(data) / len(data) if data else 0
        return {
            'avg_processing_rate_tokens_sec': safe_avg(self.metrics['processing_rate']),
            'avg_compression_ratio': safe_avg(self.metrics['compression_efficiency']),
            'avg_memory_usage_mb': safe_avg(self.metrics['memory_usage']),
            'current_stats': self.infinite_consciousness.infinite_context.get_statistics()
        }

async def setup_infinite_context_api(app, consciousness_object: Union[UnifiedConsciousness, InfiniteConsciousness]):
    """
    Setup infinite context API in FastAPI app.
    `consciousness_object` can now be either UnifiedConsciousness or InfiniteConsciousness.
    """
    
    infinite_api = InfiniteContextAPI(consciousness_object) # InfiniteContextAPI now handles both types
    app.include_router(infinite_api.router)
    
    monitor = InfiniteContextMonitor(infinite_api.infinite_consciousness) # Monitor uses the IC from the API instance
    
    # Store the monitor task to allow cancellation on shutdown
    monitor_task = asyncio.create_task(monitor.start_monitoring())
    
    if not hasattr(app.state, 'tasks_to_cancel_on_shutdown'):
        app.state.tasks_to_cancel_on_shutdown = []
    app.state.tasks_to_cancel_on_shutdown.append(monitor_task)

    # For potential access from other parts of the app if needed (e.g. admin panel)
    app.state.infinite_api_instance = infinite_api
    app.state.infinite_monitor_instance = monitor
    
    return infinite_api

# Client examples (InfiniteContextClient, example_process_book, etc.) can remain as they are,
# as they interact with the API endpoints, not directly with the server-side classes.
# Their implementation details are not critical for this specific integration step.
# Minimal changes for robustness:

class InfiniteContextClient:
    def __init__(self, base_url: str = "http://localhost:8000"): # Default port might be 8001
        self.base_url = base_url
        self.session = None # To be initialized in __aenter__
        self.logger = Logger(self.__class__.__name__) # Add logger

    async def __aenter__(self):
        try:
            import aiohttp # Import here to make it an optional dev dependency
            self.session = aiohttp.ClientSession()
        except ImportError:
            self.logger.error("aiohttp is not installed. InfiniteContextClient cannot function. Please install it: pip install aiohttp")
            raise
        return self
        
    async def __aexit__(self, exc_type, exc_val, exc_tb):
        if self.session:
            await self.session.close()
            
    async def process_large_file(self, file_path: str, session_id: Optional[str] = None):
        if not self.session: raise RuntimeError("Client session not initialized. Use 'async with InfiniteContextClient() as client:'.")
        try:
            with open(file_path, 'r', encoding='utf-8') as f: content = f.read()
        except FileNotFoundError:
            self.logger.error(f"File not found: {file_path}")
            yield {"type": "error", "error": f"File not found: {file_path}"}
            return
        except Exception as e:
            self.logger.error(f"Error reading file {file_path}: {e}", exc_info=True)
            yield {"type": "error", "error": f"Error reading file: {str(e)}"}
            return

        # Use InfiniteInputRequest model for consistency
        request_payload = InfiniteInputRequest(
            session_id=session_id,
            process_incrementally=True # Example: process document parts
        )
        
        async with self.session.post(
            f"{self.base_url}/api/v1/infinite/process-document",
            json={"file_content": content, **request_payload.model_dump(exclude_none=True)} # Pass content separately
        ) as resp:
            if resp.status != 200:
                err_text = await resp.text()
                self.logger.error(f"API error processing document ({resp.status}): {err_text}")
                yield {"type": "error", "error": f"API error ({resp.status}): {err_text}"}
                return

            result = await resp.json()
            session_id = result.get('session_id') # Get session_id from response
            if not session_id:
                yield {"type": "error", "error": "API did not return a session_id for document processing."}
                return
            
            yield result # Yield the initial response like "processing_started"

            # Example: If you had a WebSocket to listen for progress (conceptual)
            # This client doesn't implement the WebSocket listener part here,
            # but shows where it would connect if the API provided such updates.
            # For now, it just indicates processing started.

    # Other client methods (generate_unlimited, query_with_context) would also benefit
    # from similar robustness checks (session initialization, error handling from API responses).