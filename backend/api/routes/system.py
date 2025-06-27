# backend/api/routes/system.py

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
from backend.hardware.resource_manager import HardwareResourceManager
from backend.api.dependencies import get_consciousness

def get_resource_manager() -> HardwareResourceManager:
    return HardwareResourceManager()

class GPUStatus(BaseModel):
    id: int
    load_percent: float = Field(..., description="Current GPU core load percentage.")
    memory_percent: float = Field(..., description="Current GPU memory usage percentage.")
    memory_used_gb: float = Field(..., description="Used GPU memory in GiB.")

class RAMStatus(BaseModel):
    percent: float = Field(..., description="Current RAM usage percentage.")
    used_gb: float = Field(..., description="Used RAM in GiB.")
    total_gb: float = Field(..., description="Total system RAM in GiB.")

class SystemMetricsResponse(BaseModel):
    cpu_percent: float = Field(..., description="Overall CPU usage percentage.")
    ram: RAMStatus
    gpus: List[GPUStatus]

class ActionResponse(BaseModel):
    action: str
    status: str
    message: Optional[str] = None

class SystemAction(str, Enum):
    CLEAR_MIND_CACHES = "clear_mind_caches"
    TRIGGER_MEMORY_CLEANUP = "trigger_memory_cleanup"

router = APIRouter()
logger = Logger(__name__)

@router.get(
    "/metrics",
    response_model=SystemMetricsResponse,
    summary="Get real-time hardware resource metrics"
)
async def get_system_metrics(
    resource_manager: HardwareResourceManager = Depends(get_resource_manager)
) -> SystemMetricsResponse:
    # This endpoint relies on HardwareResourceManager, which is independent of consciousness type.
    metrics = resource_manager.get_current_metrics()
    if not metrics:
        logger.warning("Resource metrics are not available yet. The monitoring thread may not have run or data is missing.")
        # Ensure a valid, though potentially empty, structure is returned if metrics are partially formed.
        return SystemMetricsResponse(
            cpu_percent=metrics.get('cpu_percent', 0.0) if metrics else 0.0,
            ram=RAMStatus(**metrics.get('ram', {'percent':0.0, 'used_gb':0.0, 'total_gb':0.0})) if metrics else RAMStatus(percent=0.0, used_gb=0.0, total_gb=0.0),
            gpus=metrics.get('gpus', []) if metrics else []
        )
    
    try:
        # Pydantic will validate the structure of metrics.
        # Ensure default values for missing sub-dictionaries if metrics can be sparse.
        ram_data = metrics.get('ram', {'percent': 0.0, 'used_gb': 0.0, 'total_gb': 0.0})
        gpu_data = metrics.get('gpus', [])

        return SystemMetricsResponse(
            cpu_percent=metrics.get('cpu_percent', 0.0),
            ram=RAMStatus(**ram_data),
            gpus=[GPUStatus(**gpu) for gpu in gpu_data] # Ensure each GPU item conforms to GPUStatus
        )
    except Exception as e:
        logger.error(f"Error structuring metrics response: {e}", exc_info=True)
        # Fallback to a default error-indicating response
        return SystemMetricsResponse(
            cpu_percent=0.0,
            ram=RAMStatus(percent=0.0, used_gb=0.0, total_gb=0.0),
            gpus=[]
        )


def _sanitize_config(config: Dict[str, Any]) -> Dict[str, Any]:
    sensitive_keys = {"password", "secret", "token", "api_key", "license_key"} # Expanded list
    sanitized_dict = {}
    if not isinstance(config, dict): # Handle non-dict input gracefully
        return {"error": "Invalid config format, expected a dictionary."}

    for key, value in config.items():
        if isinstance(key, str) and key.lower() in sensitive_keys: # Check if key is string
            sanitized_dict[key] = "********"
        elif isinstance(value, dict):
            sanitized_dict[key] = _sanitize_config(value)
        elif isinstance(value, list): # Handle lists, potentially containing sensitive dicts
            sanitized_dict[key] = [_sanitize_config(item) if isinstance(item, dict) else item for item in value]
        else:
            sanitized_dict[key] = value
    return sanitized_dict

@router.get(
    "/config",
    response_model=Dict[str, Any],
    summary="Get the current (sanitized) system configuration"
)
async def get_system_config(
    # --- MODIFICATION START ---
    consciousness: InfiniteConsciousness = Depends(get_consciousness)
    # --- MODIFICATION END ---
) -> Dict[str, Any]:
    # --- MODIFICATION START ---
    # Access config from the underlying UnifiedConsciousness instance
    underlying_uc = consciousness.consciousness
    if not underlying_uc or not hasattr(underlying_uc, 'config') or not underlying_uc.config:
        logger.error("Configuration not found in underlying UnifiedConsciousness for /config endpoint.")
        raise HTTPException(status_code=503, detail="System configuration is not available.")
    
    config_to_sanitize = underlying_uc.config
    # --- MODIFICATION END ---
    
    if not isinstance(config_to_sanitize, dict):
         logger.error(f"Loaded config is not a dictionary (type: {type(config_to_sanitize)}). Cannot sanitize.")
         raise HTTPException(status_code=500, detail="Invalid server configuration format.")

    try:
        sanitized = _sanitize_config(config_to_sanitize)
        return sanitized
    except Exception as e:
        logger.error(f"Error sanitizing system config: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing system configuration: {str(e)}")


@router.post(
    "/actions/{action_name}",
    response_model=ActionResponse,
    summary="Trigger a system-level administrative action"
)
async def perform_system_action(
    action_name: SystemAction = Path(..., description="The system action to perform."),
    # --- MODIFICATION START ---
    consciousness: InfiniteConsciousness = Depends(get_consciousness)
    # --- MODIFICATION END ---
) -> ActionResponse:
    logger.warning(f"API request to perform system action: '{action_name.value}'")
    try:
        # --- MODIFICATION START ---
        underlying_uc = consciousness.consciousness
        if not underlying_uc:
            logger.error("Underlying UnifiedConsciousness not found for /actions endpoint.")
            raise HTTPException(status_code=503, detail="Core consciousness engine not available.")
        # --- MODIFICATION END ---

        if action_name == SystemAction.CLEAR_MIND_CACHES:
            # Access minds from the underlying_uc
            if underlying_uc.logical_mind: underlying_uc.logical_mind.clear_cache()
            if underlying_uc.creative_mind: underlying_uc.creative_mind.clear_cache()
            if underlying_uc.emotional_mind: underlying_uc.emotional_mind.clear_cache()
            msg = "All mind caches have been cleared."
            logger.info(msg)
            return ActionResponse(action=action_name.value, status="success", message=msg)

        elif action_name == SystemAction.TRIGGER_MEMORY_CLEANUP:
            # Access working_memory from the underlying_uc
            if not underlying_uc.working_memory:
                 raise HTTPException(status_code=503, detail="Working Memory system is not available.")
            deleted_count = await underlying_uc.working_memory.cleanup_expired()
            msg = f"Triggered manual cleanup. Removed {deleted_count} expired working memory entries."
            logger.info(msg)
            return ActionResponse(action=action_name.value, status="success", message=msg)
            
        else:
            # This case should not be reachable due to Pydantic validation of SystemAction enum
            # but as a safeguard:
            logger.error(f"Attempted to perform unknown system action: {action_name}")
            raise HTTPException(status_code=400, detail=f"Unknown action: {action_name}")

    except HTTPException as http_exc:
        logger.warning(f"HTTPException during system action '{action_name.value}': {http_exc.detail}")
        raise http_exc
    except Exception as e:
        logger.error(f"Error performing system action '{action_name.value}'. Error: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"An error occurred while performing action '{action_name.value}': {str(e)}")