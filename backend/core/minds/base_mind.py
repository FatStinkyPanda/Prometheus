# backend/core/minds/base_mind.py

import abc
import torch
from collections import OrderedDict
from typing import Dict, Any, Optional, List, Tuple
from transformers import AutoModel, AutoTokenizer, PreTrainedModel, PreTrainedTokenizer

from backend.utils.logger import Logger

class BaseMind(abc.ABC):
    """
    An abstract base class for all specialized minds within the Prometheus system.

    This class defines the core interface and shared functionalities for each mind,
    including model loading, device management, caching, and processing logic.
    Each subclass (Logical, Creative, Emotional) must implement the abstract
    `process` and `reflect` methods.
    """

    def __init__(self, config: Dict[str, Any], neural_config: Dict[str, Any]):
        """
        Initializes the BaseMind.

        Args:
            config (Dict[str, Any]): Configuration specific to this mind (e.g., model name).
            neural_config (Dict[str, Any]): Global neural configuration (e.g., device, precision).
        """
        self.config = config
        self.neural_config = neural_config
        self.logger = Logger(self.__class__.__name__)

        self.device_name = self.config.get('device', self.neural_config.get('device', 'auto'))
        self.device = self._get_torch_device()

        self.model_name: str = self.config.get('model_name')
        if not self.model_name:
            raise ValueError(f"No 'model_name' provided in the configuration for {self.__class__.__name__}")

        self.model: Optional[PreTrainedModel] = None
        self.tokenizer: Optional[PreTrainedTokenizer] = None

        self.is_initialized: bool = False
        self.use_amp: bool = False

        # Simple LRU cache implementation
        self.cache_size: int = self.config.get('cache_size', 128)
        self.cache: OrderedDict[str, Any] = OrderedDict()

        self.logger.info(f"Instantiated on device '{self.device}'. Model to be loaded: '{self.model_name}'.")

    def _get_torch_device(self) -> torch.device:
        """Determines and returns the appropriate torch.device."""
        if self.device_name == 'auto':
            if torch.cuda.is_available():
                return torch.device('cuda')
            elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
                return torch.device('mps')
            else:
                return torch.device('cpu')
        return torch.device(self.device_name)

    async def _load_model_and_tokenizer(self):
        """
        Loads the Hugging Face model and tokenizer from the specified model_name.
        This is a separate async method to allow for non-blocking I/O if models
        are downloaded.
        """
        self.logger.info(f"Loading model and tokenizer for '{self.model_name}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModel.from_pretrained(self.model_name)
            self.model.to(self.device)
            self.model.eval()  # Set model to evaluation mode by default
            self.logger.info(f"Successfully loaded '{self.model_name}' to device '{self.device}'.")
        except OSError as e:
            self.logger.critical(
                f"Could not load model '{self.model_name}'. "
                "Ensure the model name is correct and you have an internet connection "
                "if the model is not cached. Error: %s", e, exc_info=True
            )
            raise
        except Exception as e:
            self.logger.critical(
                f"An unexpected error occurred while loading model '{self.model_name}'. Error: %s",
                e, exc_info=True
            )
            raise

    async def initialize(self):
        """
        Initializes the mind by loading its neural model and other resources.
        This must be called before the mind can be used.
        """
        if self.is_initialized:
            self.logger.warning("Mind is already initialized.")
            return

        await self._load_model_and_tokenizer()
        self.is_initialized = True
        self.logger.info("Initialization complete.")

    def _check_initialized(self):
        """Raises a RuntimeError if the mind has not been initialized."""
        if not self.is_initialized or not self.model or not self.tokenizer:
            raise RuntimeError(f"{self.__class__.__name__} has not been initialized. Call await .initialize() first.")

    def enable_mixed_precision(self):
        """Enables Automatic Mixed Precision (AMP) for CUDA devices."""
        if self.device.type == 'cuda':
            self.use_amp = True
            self.logger.info("Automatic Mixed Precision (AMP) enabled for this mind.")
        else:
            self.logger.warning("Mixed precision is only available for CUDA devices. Request ignored.")

    def compile_model(self):
        """Compiles the model using torch.compile for performance gains (PyTorch 2.0+)."""
        self._check_initialized()
        if hasattr(torch, 'compile'):
            self.logger.info("Attempting to compile model with torch.compile()...")
            try:
                # 'max-autotune' is good for long-running inference workloads.
                self.model = torch.compile(self.model, mode="max-autotune")
                self.logger.info("Model successfully compiled.")
            except Exception as e:
                self.logger.error("Failed to compile model. It will run in eager mode. Error: %s", e, exc_info=True)
        else:
            self.logger.info("torch.compile not available in this PyTorch version. Model will run in eager mode.")

    def _get_from_cache(self, key: str) -> Optional[Any]:
        """Retrieves an item from the cache and moves it to the end (most recently used)."""
        if key in self.cache:
            self.cache.move_to_end(key)
            return self.cache[key]
        return None

    def _set_in_cache(self, key: str, value: Any):
        """Adds an item to the cache, evicting the oldest if the cache is full."""
        self.cache[key] = value
        if len(self.cache) > self.cache_size:
            self.cache.popitem(last=False)  # Pop the least recently used item

    def clear_cache(self):
        """Clears all items from the mind's cache."""
        self.cache.clear()
        self.logger.info("Cache has been cleared.")

    @abc.abstractmethod
    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process an input and return the mind's generated state.
        This is the primary method for responding to user input or other stimuli.

        Args:
            input_data (Dict[str, Any]): The processed input from the IO systems.
                                         Typically contains text, metadata, etc.

        Returns:
            Dict[str, Any]: A dictionary representing the mind's state, including its
                            output, confidence, and other relevant metrics.
        """
        pass

    @abc.abstractmethod
    async def reflect(self) -> Tuple[str, float]:
        """
        Perform a cycle of autonomous "thinking" or reflection.
        This is used for background processing, self-analysis, and idea generation.

        Returns:
            Tuple[str, float]: A tuple containing a summary of the reflection and
                               a confidence or relevance score.
        """
        pass