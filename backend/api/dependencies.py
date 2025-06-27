# backend/api/dependencies.py

from typing import Optional, TYPE_CHECKING

# --- MODIFICATION START ---
# Import InfiniteConsciousness for type hinting.
# UnifiedConsciousness is still needed as InfiniteConsciousness wraps it.
if TYPE_CHECKING:
    from backend.core.consciousness.unified_consciousness import UnifiedConsciousness # Keep for clarity of underlying type
    from backend.core.consciousness.infinite_context_integration import InfiniteConsciousness

# This global variable will now hold the single instance of InfiniteConsciousness.
_consciousness_instance: Optional['InfiniteConsciousness'] = None
# --- MODIFICATION END ---


def set_consciousness_instance(instance: 'InfiniteConsciousness'): # Changed type hint
    """
    Sets the global consciousness instance. This should be called once at startup
    with an InfiniteConsciousness instance.
    
    Args:
        instance (InfiniteConsciousness): The fully initialized and wrapped consciousness engine.
    """
    global _consciousness_instance
    if _consciousness_instance is not None and instance is not _consciousness_instance :
        # This check is more for development sanity, as the singleton pattern
        # in DatabaseManager and the main app flow should prevent this.
        # However, if a logger is available here, it would be good to log a warning.
        print("WARNING: Overwriting existing global consciousness instance in api.dependencies. This might indicate an issue in the setup flow.")
    _consciousness_instance = instance


def get_consciousness() -> 'InfiniteConsciousness': # Changed return type hint
    """
    FastAPI dependency injector that provides the single InfiniteConsciousness instance.
    
    Raises:
        RuntimeError: If the consciousness instance has not been set before this is called.
        
    Returns:
        InfiniteConsciousness: The active, wrapped consciousness engine instance.
    """
    if _consciousness_instance is None:
        # This error indicates a problem in the application's startup sequence.
        raise RuntimeError("The InfiniteConsciousness instance has not been set for the API.")
    return _consciousness_instance

# --- MODIFICATION START ---
# We import the actual types at the end to avoid circular dependency issues
# at the type-checking level itself, ensuring they are available for the hints.
from backend.core.consciousness.unified_consciousness import UnifiedConsciousness # For clarity
from backend.core.consciousness.infinite_context_integration import InfiniteConsciousness
# --- MODIFICATION END ---