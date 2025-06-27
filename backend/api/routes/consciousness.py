# backend/api/routes/consciousness.py

from fastapi import APIRouter, Depends
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field
from typing import Dict, Any, Optional, AsyncGenerator

# --- MODIFICATION START ---
# UnifiedConsciousness and ConsciousnessState are part of the underlying object
from backend.core.consciousness.unified_consciousness import ConsciousnessState 
# InfiniteConsciousness is the type we expect from get_consciousness
from backend.core.consciousness.infinite_context_integration import InfiniteConsciousness
# --- MODIFICATION END ---

from backend.io_systems.io_types import InputType, OutputType, OutputPayload
from backend.utils.logger import Logger
from backend.api.dependencies import get_consciousness

# Pydantic models for API data validation and serialization
class ProcessRequest(BaseModel):
    session_id: str = Field(..., description="A unique identifier for the conversation session.")
    content: str = Field(..., description="The main content of the request (e.g., text, base64 encoded file).")
    input_type: InputType = Field(default=InputType.TEXT, description="The type of the input content.")
    preferred_output_type: OutputType = Field(default=OutputType.TEXT, description="The desired output format.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata for the request.")

class StateResponse(BaseModel):
    state: str
    is_autonomous_process_running: bool

router = APIRouter()
logger = Logger(__name__)

@router.post("/process", summary="Process an input through the consciousness and stream the response")
async def process_input(
    request: ProcessRequest,
    # --- MODIFICATION START ---
    # consciousness is now an InfiniteConsciousness instance
    consciousness: InfiniteConsciousness = Depends(get_consciousness)
    # --- MODIFICATION END ---
) -> StreamingResponse:
    logger.info(f"Received API stream request for session '{request.session_id}' of type '{request.input_type.value}'")
    
    # --- MODIFICATION START ---
    # Access StreamManager from the underlying UnifiedConsciousness instance
    underlying_uc = consciousness.consciousness 
    if not underlying_uc:
        logger.critical("Underlying UnifiedConsciousness not found in InfiniteConsciousness wrapper for /process endpoint.")
        async def error_gen_uc_missing():
            err_payload = OutputPayload(type=OutputType.ERROR, content="Critical Error: Core consciousness engine not available.")
            yield f"data: {err_payload.model_dump_json()}\n\n"
        return StreamingResponse(error_gen_uc_missing(), media_type="text/event-stream", status_code=500)

    stream_manager = underlying_uc.get_stream_manager()
    # --- MODIFICATION END ---

    if not stream_manager:
        logger.error("StreamManager is not initialized in UnifiedConsciousness for /process endpoint.")
        async def error_generator():
            error_payload = OutputPayload(type=OutputType.ERROR, content="Critical Error: StreamManager is not initialized.")
            yield f"data: {error_payload.model_dump_json()}\n\n"
        return StreamingResponse(error_generator(), media_type="text/event-stream", status_code=500)
    
    raw_request = {
        "type": request.input_type,
        "content": request.content,
        "session_id": request.session_id,
        "metadata": request.metadata,
        "options": {
            "preferred_output_type": request.preferred_output_type
        }
    }

    async def stream_generator() -> AsyncGenerator[str, None]:
        try:
            async for payload in stream_manager.handle_stream(raw_request):
                json_payload = payload.model_dump_json()
                yield f"data: {json_payload}\n\n"
                if payload.type == OutputType.ERROR:
                    break
        except Exception as e:
            logger.critical(f"A critical error occurred within the stream generator for session '{request.session_id}': {e}", exc_info=True)
            error_payload_critical = OutputPayload(type=OutputType.ERROR, content=f"A critical server error occurred: {str(e)}")
            json_error_payload_critical = error_payload_critical.model_dump_json()
            yield f"data: {json_error_payload_critical}\n\n"

    return StreamingResponse(stream_generator(), media_type="text/event-stream")


@router.get("/state", response_model=StateResponse, summary="Get the current state of the consciousness")
async def get_current_state(
    # --- MODIFICATION START ---
    # consciousness is now an InfiniteConsciousness instance
    consciousness: InfiniteConsciousness = Depends(get_consciousness)
    # --- MODIFICATION END ---
) -> StateResponse:
    # --- MODIFICATION START ---
    # Access state and tasks from the underlying UnifiedConsciousness instance
    underlying_uc = consciousness.consciousness
    if not underlying_uc:
        logger.error("Underlying UnifiedConsciousness not found in InfiniteConsciousness wrapper for /state endpoint.")
        # Return a default error state or raise an HTTPException
        return StateResponse(state=ConsciousnessState.OFFLINE.name, is_autonomous_process_running=False)

    current_uc_state = underlying_uc.state
    is_running_task = False
    if underlying_uc.thinking_task and not underlying_uc.thinking_task.done():
        is_running_task = True
    if underlying_uc.dreaming_task and not underlying_uc.dreaming_task.done():
        is_running_task = True
    # --- MODIFICATION END ---
        
    return StateResponse(
        state=current_uc_state.name, # Use the state from the underlying UC
        is_autonomous_process_running=is_running_task
    )