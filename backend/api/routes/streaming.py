# backend/api/routes/streaming.py

from fastapi import (APIRouter, HTTPException, Depends, BackgroundTasks, # --- MODIFICATION: Added Body ---
                     WebSocket, WebSocketDisconnect, Path, Body) 
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field # Field is used for request body structuring
from typing import Dict, Any, Optional, AsyncGenerator, List 
import asyncio
import json
from datetime import datetime
import uuid 

from backend.api.middleware.authentication import authenticate_request 
from backend.core.consciousness.unified_consciousness import UnifiedConsciousness
from backend.core.consciousness.infinite_context_integration import InfiniteConsciousness
from backend.api.dependencies import get_consciousness

from backend.core.consciousness.infinite_context_manager import InfiniteContextManager
from backend.io_systems.infinite_stream_processor import InfiniteStreamProcessor
from backend.io_systems.io_types import OutputType, OutputPayload 
from backend.utils.logger import Logger

router = APIRouter()
logger = Logger(__name__)

_context_manager_instance: Optional[InfiniteContextManager] = None
_stream_processor_instance: Optional[InfiniteStreamProcessor] = None

def get_context_manager(
    wrapped_consciousness: InfiniteConsciousness = Depends(get_consciousness)
) -> InfiniteContextManager:
    global _context_manager_instance
    if not _context_manager_instance:
        underlying_uc = wrapped_consciousness.consciousness 
        if not underlying_uc:
            logger.critical("Underlying UnifiedConsciousness not found in InfiniteConsciousness for get_context_manager.")
            raise HTTPException(status_code=503, detail="Core consciousness engine components not available for context manager.")

        if not all([underlying_uc.hierarchical_memory, underlying_uc.nlp_processor, underlying_uc.logical_mind]):
            missing = [
                name for name, comp in [
                    ("Hierarchical Memory", underlying_uc.hierarchical_memory),
                    ("NLP Processor", underlying_uc.nlp_processor),
                    ("Logical Mind", underlying_uc.logical_mind)
                ] if comp is None
            ]
            logger.critical(f"Missing components for InfiniteContextManager: {', '.join(missing)}")
            raise HTTPException(status_code=503, detail=f"Required components for context manager are not initialized: {', '.join(missing)}")
        
        _context_manager_instance = InfiniteContextManager(
            underlying_uc.hierarchical_memory,
            underlying_uc.nlp_processor,
            underlying_uc.logical_mind
        )
        logger.info("InfiniteContextManager instance created for streaming routes.")
    return _context_manager_instance

def get_stream_processor(
    wrapped_consciousness: InfiniteConsciousness = Depends(get_consciousness),
    context_manager: InfiniteContextManager = Depends(get_context_manager)
) -> InfiniteStreamProcessor:
    global _stream_processor_instance
    if not _stream_processor_instance:
        _stream_processor_instance = InfiniteStreamProcessor(wrapped_consciousness, context_manager)
        logger.info("InfiniteStreamProcessor instance created for streaming routes.")
    return _stream_processor_instance

class CreateStreamRequest(BaseModel):
    session_id: str = Field(..., description="Session ID for the stream")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict)

class StreamInputRequest(BaseModel):
    stream_id: str = Field(..., description="Stream ID from create_stream")
    content_chunk: str = Field(..., description="A chunk of input content")
    end_of_stream: bool = Field(default=False, description="Signal end of input stream")

class StreamAnalyticsResponse(BaseModel):
    stream_id: str
    session_id: str
    status: str
    tokens: Dict[str, int]
    duration_seconds: float
    context_coverage: Dict[str, float] 

@router.post("/create_stream", dependencies=[Depends(authenticate_request)])
async def create_infinite_stream(
    request: CreateStreamRequest,
    stream_processor: InfiniteStreamProcessor = Depends(get_stream_processor)
) -> Dict[str, Any]:
    try:
        session = await stream_processor.create_streaming_session(
            request.session_id,
            request.metadata
        )
        logger.info(f"Created streaming session {session.stream_id} for session {request.session_id}")
        return {
            "stream_id": session.stream_id,
            "session_id": session.session_id,
            "created_at": session.start_time.isoformat(),
            "status": "ready"
        }
    except Exception as e:
        logger.error(f"Failed to create stream for session '{request.session_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to create stream: {str(e)}")

@router.post("/stream_chunk", dependencies=[Depends(authenticate_request)])
async def stream_input_chunk(
    request: StreamInputRequest,
    stream_processor: InfiniteStreamProcessor = Depends(get_stream_processor)
) -> Dict[str, Any]:
    session = stream_processor.streaming_sessions.get(request.stream_id)
    if not session:
        logger.warning(f"Stream_chunk request for unknown stream_id: {request.stream_id}")
        raise HTTPException(status_code=404, detail="Stream not found")
    
    if not session.active:
        logger.warning(f"Stream_chunk request for inactive stream_id: {request.stream_id}")
        raise HTTPException(status_code=400, detail="Stream is not active")
    
    try:
        session.input_buffer += request.content_chunk
        session.total_input_tokens += len(request.content_chunk.split()) 
        
        response = {
            "stream_id": request.stream_id,
            "chunk_received": True,
            "total_input_tokens": session.total_input_tokens,
            "buffer_size_chars": len(session.input_buffer)
        }
        
        if request.end_of_stream:
            session.active = False 
            response["status"] = "input_stream_ended"
            logger.info(f"Input stream ended by client for stream_id: {request.stream_id}")
            
        return response
    except Exception as e:
        logger.error(f"Error processing stream_chunk for stream_id '{request.stream_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing stream chunk: {str(e)}")

@router.get("/stream/{stream_id}/analytics", response_model=StreamAnalyticsResponse, dependencies=[Depends(authenticate_request)])
async def get_stream_analytics(
    stream_id: str = Path(..., description="The ID of the stream to get analytics for."),
    stream_processor: InfiniteStreamProcessor = Depends(get_stream_processor)
) -> StreamAnalyticsResponse:
    try:
        analytics = await stream_processor.get_stream_analytics(stream_id)
        if "error" in analytics: 
            logger.warning(f"Analytics request for stream_id '{stream_id}' resulted in error: {analytics['error']}")
            raise HTTPException(status_code=404, detail=analytics["error"])
        
        return StreamAnalyticsResponse(
            stream_id=analytics.get("stream_id", stream_id),
            session_id=analytics.get("session_id", "unknown_session"),
            status=analytics.get("status", "unknown"),
            tokens=analytics.get("tokens", {"input": 0, "output": 0, "ratio": 0.0}),
            duration_seconds=analytics.get("duration_seconds", 0.0),
            context_coverage=analytics.get("context_coverage", {"coverage":0.0})
        )
    except HTTPException as http_exc:
        raise http_exc 
    except Exception as e:
        logger.error(f"Error retrieving analytics for stream_id '{stream_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error retrieving stream analytics: {str(e)}")

@router.post("/stream/{stream_id}/feedback", dependencies=[Depends(authenticate_request)])
async def submit_stream_feedback(
    stream_id: str = Path(..., description="The ID of the stream to submit feedback for."),
    # --- MODIFICATION START: Use Body(...) to explicitly mark 'feedback' as request body ---
    feedback: Dict[str, Any] = Body(..., description="Feedback data as a JSON object."), 
    # --- MODIFICATION END ---
    context_manager: InfiniteContextManager = Depends(get_context_manager),
    stream_processor: InfiniteStreamProcessor = Depends(get_stream_processor) 
) -> Dict[str, Any]:
    
    try:
        target_session_id = feedback.get("session_id")
        if not target_session_id:
            session = stream_processor.streaming_sessions.get(stream_id)
            if session:
                target_session_id = session.session_id
            else:
                logger.warning(f"Stream ID '{stream_id}' not found for feedback, and no session_id provided in feedback payload.")
                raise HTTPException(status_code=404, detail=f"Stream '{stream_id}' not found, and 'session_id' missing in feedback payload.")
        
        if not target_session_id: 
             raise HTTPException(status_code=400, detail="Could not determine session_id for feedback.")

        feedback_content = feedback.get("feedback_data", feedback) # Allow feedback_data or top-level dict
        if not isinstance(feedback_content, dict):
            logger.warning(f"Invalid feedback content for stream '{stream_id}'. Expected a dictionary, got {type(feedback_content)}")
            raise HTTPException(status_code=400, detail="Feedback content must be a valid JSON object.")


        await context_manager.learn_attention_patterns(
            target_session_id, 
            feedback_content
        )
        logger.info(f"Feedback received and processed for stream '{stream_id}' (session '{target_session_id}').")
        return {
            "stream_id": stream_id,
            "session_id": target_session_id,
            "feedback_received": True,
            "patterns_updated": True 
        }
    except HTTPException as http_exc:
        raise http_exc
    except Exception as e:
        logger.error(f"Error processing feedback for stream '{stream_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing stream feedback: {str(e)}")


@router.websocket("/ws/infinite_stream/{session_id}")
async def websocket_infinite_stream(
    websocket: WebSocket,
    session_id: str = Path(..., description="The session ID for this WebSocket stream."),
    stream_processor: InfiniteStreamProcessor = Depends(get_stream_processor),
    context_manager: InfiniteContextManager = Depends(get_context_manager)
):
    await websocket.accept()
    ws_client_id = f"ws_{session_id}_{uuid.uuid4().hex[:6]}"
    logger.info(f"WebSocket stream connected: {ws_client_id}")

    session: Optional[StreamingSession] = None 
    output_task: Optional[asyncio.Task] = None
    input_task: Optional[asyncio.Task] = None

    try:
        session = await stream_processor.create_streaming_session(session_id)
        stream_id = session.stream_id 
        
        input_queue: asyncio.Queue[Optional[str]] = asyncio.Queue() 
        
        async def handle_websocket_input():
            try:
                while True:
                    data_str = await websocket.receive_text()
                    try:
                        message = json.loads(data_str)
                    except json.JSONDecodeError:
                        logger.warning(f"WebSocket {ws_client_id} received invalid JSON: {data_str}")
                        if websocket.client_state != WebSocketDisconnect:
                            await websocket.send_json({"type": "error", "message": "Invalid JSON format"})
                        continue # Skip processing this message
                    
                    msg_type = message.get("type", "unknown")
                    logger.debug(f"WebSocket {ws_client_id} received message type: {msg_type}")

                    if msg_type == "input_chunk":
                        content_chunk = message.get("content")
                        if isinstance(content_chunk, str):
                            await input_queue.put(content_chunk)
                        else:
                            logger.warning(f"WebSocket {ws_client_id} received non-string content for input_chunk.")
                            if websocket.client_state != WebSocketDisconnect:
                                await websocket.send_json({"type": "error", "message": "Invalid content for input_chunk, must be string."})
                    elif msg_type == "end_input_stream":
                        await input_queue.put(None) 
                        logger.info(f"WebSocket {ws_client_id} signaled end of input stream.")
                        break 
                    elif msg_type == "feedback":
                        feedback_data = message.get("feedback_data", message.get("feedback")) 
                        if isinstance(feedback_data, dict):
                            await context_manager.learn_attention_patterns(session_id, feedback_data)
                            if websocket.client_state != WebSocketDisconnect:
                                await websocket.send_json({"type": "feedback_ack", "status": "received"})
                        else:
                            logger.warning(f"WebSocket {ws_client_id} received invalid feedback data type.")
                            if websocket.client_state != WebSocketDisconnect:
                                await websocket.send_json({"type": "error", "message": "Invalid feedback_data, must be a JSON object."})
                    else:
                        logger.warning(f"WebSocket {ws_client_id} received unknown message type: {msg_type}")
                        if websocket.client_state != WebSocketDisconnect:
                            await websocket.send_json({"type": "error", "message": f"Unknown message type: {msg_type}"})
            except WebSocketDisconnect:
                logger.info(f"WebSocket {ws_client_id} disconnected by client during input handling.")
                if input_queue and not input_queue.full(): await input_queue.put(None) 
            except Exception as e_input: # Catch other errors during input handling
                logger.error(f"Error in WebSocket input handler {ws_client_id}: {e_input}", exc_info=True)
                try: 
                    if websocket.client_state != WebSocketDisconnect:
                        await websocket.send_json({"type": "error", "message": f"Input handling error: {str(e_input)}"})
                except: pass 
                if input_queue and not input_queue.full(): await input_queue.put(None) 

        async def generate_output_from_stream():
            async def input_generator_from_queue():
                while True:
                    chunk = await input_queue.get()
                    if chunk is None: 
                        break
                    yield chunk
            
            try:
                async for output_payload in stream_processor.stream_unlimited_input(
                    stream_id, 
                    input_generator_from_queue()
                ):
                    if websocket.client_state == WebSocketDisconnect: 
                        logger.info(f"WebSocket {ws_client_id} disconnected before sending output payload.")
                        break 
                    await websocket.send_json({
                        "type": output_payload.type.value if output_payload.type else "unknown_payload",
                        "content": output_payload.content,
                        "metadata": output_payload.metadata,
                        "timestamp": datetime.utcnow().isoformat()
                    })
                    if output_payload.type == OutputType.ERROR:
                        logger.error(f"Error payload sent to WebSocket {ws_client_id}: {output_payload.content}")
                        break 
            except Exception as e_process:
                logger.error(f"Error during stream processing for WebSocket {ws_client_id}: {e_process}", exc_info=True)
                if websocket.client_state != WebSocketDisconnect:
                    try:
                        await websocket.send_json({"type": "error", "message": f"Core processing error: {str(e_process)}"})
                    except: pass # Ignore if send fails on already broken socket

        input_task = asyncio.create_task(handle_websocket_input())
        output_task = asyncio.create_task(generate_output_from_stream())
        
        # Wait for both tasks to complete or one to error out
        done, pending = await asyncio.wait([input_task, output_task], return_when=asyncio.FIRST_COMPLETED)
        
        for task in pending: # Cancel any still pending tasks
            task.cancel()
        await asyncio.gather(*pending, return_exceptions=True) # Wait for cancellations

        # Check for exceptions in completed tasks
        for task in done:
            if task.exception():
                logger.error(f"Task completed with exception in WebSocket {ws_client_id}: {task.exception()}", exc_info=task.exception())
        
    except WebSocketDisconnect: 
        logger.info(f"WebSocket {ws_client_id} disconnected (caught in main try block).")
    except Exception as e:
        logger.error(f"Critical error in WebSocket connection {ws_client_id}: {e}", exc_info=True)
        try:
            if websocket.client_state != WebSocketDisconnect:
                 await websocket.send_json({"type": "error", "message": f"Server error: {str(e)}"})
        except: pass 
    finally:
        # Ensure tasks are cancelled one more time if not done by gather
        if input_task and not input_task.done(): input_task.cancel()
        if output_task and not output_task.done(): output_task.cancel()
        try: # Await cancellation if tasks were cancelled
            if input_task: await input_task
            if output_task: await output_task
        except asyncio.CancelledError:
            logger.debug(f"Tasks for WebSocket {ws_client_id} were cancelled.")
        except Exception as e_final_gather:
            logger.error(f"Exception during final task gathering for WebSocket {ws_client_id}: {e_final_gather}")


        if session and not session.active: 
            logger.info(f"Streaming session {session.stream_id} for WebSocket {ws_client_id} ended.")
        elif session: 
            session.active = False
            logger.info(f"Manually marking streaming session {session.stream_id} inactive for WebSocket {ws_client_id}.")

        if websocket.client_state != WebSocketDisconnect:
            try: await websocket.close()
            except RuntimeError as e_close: 
                 logger.warning(f"Error closing WebSocket {ws_client_id} (already closed?): {e_close}")
            except Exception as e_close_final:
                 logger.error(f"Unexpected error closing WebSocket {ws_client_id}: {e_close_final}", exc_info=True)
        logger.info(f"WebSocket stream connection closed: {ws_client_id}")