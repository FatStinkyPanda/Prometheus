# backend/api/routes/conversation.py

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field
from typing import List, Dict, Any, Optional

from backend.utils.logger import Logger
# --- MODIFICATION START ---
# UnifiedConsciousness for underlying access, InfiniteConsciousness for the injected type
from backend.core.consciousness.unified_consciousness import UnifiedConsciousness 
from backend.core.consciousness.infinite_context_integration import InfiniteConsciousness
# --- MODIFICATION END ---
from backend.api.dependencies import get_consciousness

class Message(BaseModel):
    role: str = Field(..., description="The role of the sender (e.g., 'user', 'ai', 'system').")
    content: str = Field(..., description="The text content of the message.")
    timestamp: str = Field(..., description="The ISO 8601 timestamp of the message.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Optional metadata for the message.")

class ConversationHistoryResponse(BaseModel):
    session_id: str
    history: List[Message]
    total_messages: int

class SendMessageRequest(BaseModel):
    session_id: str = Field(..., description="The session to which this message belongs.")
    content: str = Field(..., description="The text content of the message.")

class SendMessageResponse(BaseModel):
    session_id: str
    status: str = Field(..., description="Status of the message processing (e.g., 'success', 'blocked', 'error').")
    response_content: str = Field(..., description="The AI's textual response.")
    metadata: Optional[Dict[str, Any]] = Field(default_factory=dict, description="Metadata associated with the response, such as mind states.")

router = APIRouter()
logger = Logger(__name__)

@router.post(
    "/message",
    response_model=SendMessageResponse,
    summary="Send a message to a session (non-streaming)"
)
async def send_message(
    request: SendMessageRequest,
    # --- MODIFICATION START ---
    # consciousness is now an InfiniteConsciousness instance
    consciousness: InfiniteConsciousness = Depends(get_consciousness)
    # --- MODIFICATION END ---
) -> SendMessageResponse:
    logger.info(f"API received non-streaming message for session '{request.session_id}': '{request.content[:50]}...'")
    try:
        # InfiniteConsciousness.process_input proxies to UnifiedConsciousness.process_input,
        # so this call should work directly.
        response_payload = await consciousness.process_input(
            input_text=request.content,
            session_id=request.session_id
        )

        status = response_payload.get("status", "error")
        # Ensure output is a string, provide a default if not.
        output_content = response_payload.get("output", "An unknown error occurred or no textual output was generated.")
        if not isinstance(output_content, str):
            logger.warning(f"Received non-string output content in /message: {type(output_content)}. Converting to string.")
            output_content = str(output_content)
        
        return SendMessageResponse(
            session_id=request.session_id,
            status=status,
            response_content=output_content,
            metadata={"final_states": response_payload.get("final_states")}
        )

    except Exception as e:
        logger.critical(f"An unexpected error occurred in /message endpoint for session '{request.session_id}': {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"A critical internal error occurred while processing the message: {str(e)}")


@router.get(
    "/history/{session_id}",
    response_model=ConversationHistoryResponse,
    summary="Retrieve the conversation history for a session"
)
async def get_history(
    session_id: str = Path(..., description="The ID of the session to retrieve history for."),
    # --- MODIFICATION START ---
    # consciousness is now an InfiniteConsciousness instance
    consciousness: InfiniteConsciousness = Depends(get_consciousness)
    # --- MODIFICATION END ---
) -> ConversationHistoryResponse:
    logger.info(f"API request for conversation history for session '{session_id}'.")

    # --- MODIFICATION START ---
    # Access hierarchical_memory from the underlying UnifiedConsciousness instance
    underlying_uc = consciousness.consciousness
    if not underlying_uc:
        logger.error("Underlying UnifiedConsciousness not found in InfiniteConsciousness wrapper for /history endpoint.")
        raise HTTPException(status_code=503, detail="Core consciousness engine not available.")

    if not underlying_uc.hierarchical_memory:
        logger.error("HierarchicalMemory is not available in UnifiedConsciousness to retrieve history.")
        raise HTTPException(status_code=503, detail="Memory system (hierarchical) is not available.")
    # --- MODIFICATION END ---

    try:
        # Use the hierarchical_memory from the underlying_uc
        interactions = await underlying_uc.hierarchical_memory.search(
            query="", 
            session_id=session_id,
            limit=1000 
        )

        interactions.reverse()

        history: List[Message] = []
        for node in interactions:
            if not isinstance(node, dict): # Ensure node is a dict
                logger.warning(f"Skipping non-dict node in history for session {session_id}: {type(node)}")
                continue

            metadata = node.get('metadata', {})
            if not isinstance(metadata, dict): # Ensure metadata is a dict
                logger.warning(f"Skipping node with non-dict metadata for session {session_id}")
                metadata = {} # Use empty dict as fallback

            timestamp = node.get('timestamp')
            timestamp_str = timestamp.isoformat() if hasattr(timestamp, 'isoformat') else str(timestamp or "N/A")

            input_text = metadata.get('input_text')
            if input_text and isinstance(input_text, str):
                history.append(Message(role="user", content=input_text, timestamp=timestamp_str, metadata=metadata.get("input_metadata", {}))) # Pass specific input metadata if available
            
            output_text = metadata.get('output_text')
            if output_text and isinstance(output_text, str):
                role = "ai"
                if metadata.get("status") == "blocked":
                    role = "system" 
                history.append(Message(role=role, content=output_text, timestamp=timestamp_str, metadata=metadata.get("output_metadata", {}))) # Pass specific output metadata
                
        return ConversationHistoryResponse(
            session_id=session_id,
            history=history,
            total_messages=len(history)
        )

    except Exception as e:
        logger.error(f"Failed to retrieve history for session '{session_id}'. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Failed to retrieve conversation history: {str(e)}")