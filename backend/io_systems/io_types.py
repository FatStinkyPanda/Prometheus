# backend/io_systems/io_types.py

from enum import Enum
from typing import Any, Dict, List, Optional, Union
from pydantic import BaseModel, Field

# Using str as a mixin makes the Enum members easily JSON-serializable
class InputType(str, Enum):
    """Enumerates the types of input the system can receive."""
    TEXT = "text"
    VOICE = "voice"
    IMAGE = "image"
    AUDIO = "audio"
    VIDEO = "video"
    DOCUMENT = "document"
    SYSTEM_COMMAND = "system_command"

class OutputType(str, Enum):
    """Enumerates the types of output the system can generate."""
    TEXT = "text"
    AUDIO = "audio"
    IMAGE = "image"
    CODE = "code"
    ANALYSIS = "analysis"
    EMOTIONAL = "emotional"
    CREATIVE = "creative"
    STRUCTURED = "structured"
    COMPLEX = "complex"
    ERROR = "error"

# --- FIX: Converted from dataclass to Pydantic BaseModel ---
class InputPayload(BaseModel):
    """
    A structured, Pydantic-based data object representing a single input to the system.
    This standardizes the data passed from any frontend to the UnifiedConsciousness.
    """
    type: InputType
    content: Union[str, bytes]
    session_id: str
    metadata: Dict[str, Any] = Field(default_factory=dict)
    options: Dict[str, Any] = Field(default_factory=dict)

# --- FIX: Converted from dataclass to Pydantic BaseModel ---
class OutputPayload(BaseModel):
    """
    A structured, Pydantic-based data object representing a single piece of output from the system.
    """
    type: OutputType
    content: Any
    summary: Optional[str] = None
    metadata: Dict[str, Any] = Field(default_factory=dict)

# --- FIX: Converted from dataclass to Pydantic BaseModel ---
class ComplexOutputPayload(OutputPayload):
    """
    A specialized OutputPayload for representing a multi-part response.
    The 'content' field holds a list of other OutputPayload objects.
    """
    type: OutputType = Field(default=OutputType.COMPLEX, frozen=True)
    content: List[OutputPayload]