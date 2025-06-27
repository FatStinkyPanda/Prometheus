# backend/api/routes/memory.py

from fastapi import APIRouter, Depends, HTTPException, Path
from pydantic import BaseModel, Field
from enum import Enum
from typing import List, Dict, Any, Optional

from backend.utils.logger import Logger
# --- MODIFICATION START ---
# UnifiedConsciousness for underlying access, InfiniteConsciousness for the injected type
from backend.core.consciousness.unified_consciousness import UnifiedConsciousness
from backend.core.consciousness.infinite_context_integration import InfiniteConsciousness
# --- MODIFICATION END ---
from backend.api.dependencies import get_consciousness

class MemoryType(str, Enum):
    TRUTH = "truth"
    DREAM = "dream"
    HIERARCHICAL = "hierarchical" # Previously was Contextual, now matches HierarchicalMemory

class MemorySearchRequest(BaseModel):
    query_text: str = Field(..., description="The text to search for semantically. For HIERARCHICAL, leave empty to retrieve history for a session_id.")
    memory_type: MemoryType = Field(..., description="The type of memory to search in.")
    session_id: Optional[str] = Field(None, description="The session ID to filter by. Required for HIERARCHICAL history retrieval.")
    limit: int = Field(default=25, ge=1, le=200, description="The maximum number of results to return.")
    min_importance: float = Field(default=0.0, ge=0.0, le=1.0, description="Minimum importance score for HIERARCHICAL search.")

class MemorySearchResponse(BaseModel):
    query: MemorySearchRequest
    results: List[Dict[str, Any]]

router = APIRouter()
logger = Logger(__name__)

@router.post(
    "/search",
    response_model=MemorySearchResponse,
    summary="Search across different memory systems"
)
async def search_memory(
    request: MemorySearchRequest,
    # --- MODIFICATION START ---
    consciousness: InfiniteConsciousness = Depends(get_consciousness)
    # --- MODIFICATION END ---
) -> MemorySearchResponse:
    logger.info(f"API memory search request for type '{request.memory_type.value}' with query: '{request.query_text[:50]}...' for session '{request.session_id}'")
    
    results: List[Dict[str, Any]] = []
    try:
        # --- MODIFICATION START ---
        underlying_uc = consciousness.consciousness
        if not underlying_uc:
            logger.error("Underlying UnifiedConsciousness not found in InfiniteConsciousness wrapper for /search endpoint.")
            raise HTTPException(status_code=503, detail="Core consciousness engine not available.")
        # --- MODIFICATION END ---

        if request.memory_type == MemoryType.TRUTH:
            if not underlying_uc.truth_memory:
                raise HTTPException(status_code=503, detail="Truth Memory system is not available.")
            results = await underlying_uc.truth_memory.search(request.query_text, limit=request.limit)

        elif request.memory_type == MemoryType.DREAM:
            if not underlying_uc.dream_memory:
                raise HTTPException(status_code=503, detail="Dream Memory system is not available.")
            results = await underlying_uc.dream_memory.search(query_text=request.query_text, limit=request.limit)

        elif request.memory_type == MemoryType.HIERARCHICAL:
            if not underlying_uc.hierarchical_memory:
                raise HTTPException(status_code=503, detail="Hierarchical Memory system is not available.")
            
            if not request.query_text and not request.session_id:
                raise HTTPException(status_code=400, detail="For Hierarchical memory, either 'query_text' or 'session_id' for history retrieval is required.")

            results = await underlying_uc.hierarchical_memory.search(
                query=request.query_text,
                session_id=request.session_id,
                limit=request.limit,
                min_importance=request.min_importance
            )
        
        return MemorySearchResponse(query=request, results=results)

    except HTTPException as http_exc:
        logger.warning(f"HTTPException during memory search: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"An unexpected error occurred during memory search. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred while searching memory: {str(e)}")


@router.get(
    "/{memory_type}/{entry_id}",
    response_model=Dict[str, Any], # The response can be any dict from the memory systems
    summary="Get a specific entry from a memory system"
)
async def get_memory_entry(
    memory_type: MemoryType = Path(..., description="The type of memory to retrieve from."),
    entry_id: str = Path(..., description="The UUID or ID of the memory entry to retrieve."),
    # --- MODIFICATION START ---
    consciousness: InfiniteConsciousness = Depends(get_consciousness)
    # --- MODIFICATION END ---
) -> Dict[str, Any]:
    logger.info(f"API request for memory entry '{entry_id}' from '{memory_type.value}' memory.")

    entry: Optional[Dict[str, Any]] = None
    try:
        # --- MODIFICATION START ---
        underlying_uc = consciousness.consciousness
        if not underlying_uc:
            logger.error("Underlying UnifiedConsciousness not found in InfiniteConsciousness wrapper for /get_memory_entry endpoint.")
            raise HTTPException(status_code=503, detail="Core consciousness engine not available.")
        # --- MODIFICATION END ---

        if memory_type == MemoryType.TRUTH:
            if not underlying_uc.truth_memory:
                raise HTTPException(status_code=503, detail="Truth Memory system is not available.")
            # Truth memory get method usually takes the claim text. For API consistency by ID:
            record = await underlying_uc.truth_memory._fetchrow("SELECT * FROM truths WHERE id = $1;", entry_id) # Use internal method carefully
            if record:
                 entry = underlying_uc.truth_memory._format_record(record)

        elif memory_type == MemoryType.DREAM:
            if not underlying_uc.dream_memory:
                raise HTTPException(status_code=503, detail="Dream Memory system is not available.")
            entry = await underlying_uc.dream_memory.get(entry_id)

        elif memory_type == MemoryType.HIERARCHICAL:
            if not underlying_uc.hierarchical_memory:
                raise HTTPException(status_code=503, detail="Hierarchical Memory system is not available.")
            entry = await underlying_uc.hierarchical_memory.get(entry_id) # Assumes hierarchical_memory has a get(node_id)

        if not entry:
            raise HTTPException(status_code=404, detail=f"Entry with ID '{entry_id}' not found in {memory_type.value} memory.")
        
        return entry

    except HTTPException as http_exc:
        logger.warning(f"HTTPException during memory entry retrieval: {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Failed to retrieve memory entry '{entry_id}'. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An internal error occurred while retrieving the memory entry: {str(e)}")