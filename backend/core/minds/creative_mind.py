# backend/core/minds/creative_mind.py

import random
from typing import Dict, Any, Optional, Tuple, TYPE_CHECKING

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from backend.core.minds.base_mind import BaseMind

if TYPE_CHECKING:
    from backend.core.minds.logical_mind import LogicalMind

class CreativeMind(BaseMind):
    """
    The Creative Mind is responsible for divergent thinking, synthesis, and novelty generation.
    It uses a powerful generative language model to produce creative, coherent, and contextually
    aware text based on instruction-following.
    """

    def __init__(self, config: Dict[str, Any], neural_config: Dict[str, Any]):
        """Initializes the CreativeMind."""
        super().__init__(config, neural_config)
        
        self.temperature = self.config.get('temperature', 0.7)
        self.top_k = self.config.get('top_k', 50)
        self.max_new_tokens = self.config.get('max_new_tokens', 512)

        self.reflection_prompts = [
            "What if the opposite were true?",
            "Consider a metaphor for the current situation.",
            "Write a short story about the last topic of discussion.",
            "Combine the two most recent concepts into a new idea.",
            "Imagine a world where the primary assumption is false."
        ]

    async def _load_model_and_tokenizer(self):
        """Overrides the base method to load a CausalLM model and tokenizer."""
        self.logger.info(f"Loading CausalLM model and tokenizer for '{self.model_name}'...")
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                torch_dtype=torch.float16 if self.device.type == 'cuda' else torch.float32,
                device_map="auto" # Let transformers handle multi-gpu if available
            )
            
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # No need for model.to(self.device) when using device_map="auto"
            self.model.eval()
            self.logger.info(f"Successfully loaded CausalLM model '{self.model_name}'.")
        except Exception as e:
            self.logger.critical(f"Failed to load CausalLM model '{self.model_name}'. Error: %s", e, exc_info=True)
            raise

    async def _get_embedding_from_text(self, text: str, logical_mind_instance: 'LogicalMind') -> Optional[list]:
        """Uses a provided LogicalMind instance to get a consistent state vector."""
        if not logical_mind_instance:
            self.logger.error("LogicalMind instance not provided to generate a state embedding.")
            return None
        try:
            result = await logical_mind_instance.process({'text': text})
            state = result.get('state')
            if isinstance(state, torch.Tensor):
                return state.cpu().tolist()
            return state
        except Exception as e:
            self.logger.error("Failed to get embedding from LogicalMind for generated text. Error: %s", e, exc_info=True)
            return None

    async def process(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generates creative text based on an input prompt using a chat template."""
        self._check_initialized()

        prompt_text = input_data.get('text')
        logical_mind = input_data.get('logical_mind_instance')

        if not prompt_text or not isinstance(prompt_text, str):
            return {"state": None, "confidence": 0.1, "type": "creative", "payload": {"generated_text": ""}}

        if not logical_mind:
             self.logger.error("A 'logical_mind_instance' is required to generate a state vector.")
             return {"state": None, "confidence": 0.0, "type": "creative", "payload": {"generated_text": ""}, "error": "Missing logical_mind_instance"}

        cached_result = self._get_from_cache(prompt_text)
        if cached_result:
            return cached_result

        self.logger.debug("Processing prompt with CreativeMind: '%s...'", prompt_text[:100])

        try:
            # --- NEW: Use Chat Templating for Instruction-Tuned Models ---
            messages = [
                {"role": "system", "content": "You are Prometheus, a helpful and creative AI assistant."},
                {"role": "user", "content": prompt_text}
            ]
            # `add_generation_prompt=True` adds the tokens that signal the model to start responding.
            tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)
            
            with torch.no_grad():
                with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model.generate(
                        tokenized_chat, 
                        max_new_tokens=self.max_new_tokens, 
                        temperature=self.temperature,
                        top_k=self.top_k, 
                        pad_token_id=self.tokenizer.pad_token_id,
                        eos_token_id=self.tokenizer.eos_token_id, 
                        do_sample=True,
                    )
            
            # --- NEW: Precise Text Extraction ---
            # We decode the full output and then slice off the prompt part.
            full_response_tokens = outputs[0]
            prompt_tokens_len = tokenized_chat.shape[1]
            newly_generated_tokens = full_response_tokens[prompt_tokens_len:]
            
            generated_text = self.tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip() or "...silence."

            state_embedding = await self._get_embedding_from_text(generated_text, logical_mind)

            result = {
                "state": state_embedding, "confidence": 0.85, "type": "creative",
                "payload": {"generated_text": generated_text}
            }
            self._set_in_cache(prompt_text, result)
            return result

        except Exception as e:
            self.logger.error("An error occurred during CreativeMind processing. Error: %s", e, exc_info=True)
            return {"state": None, "confidence": 0.0, "type": "creative", "payload": {"generated_text": ""}, "error": str(e)}

    async def reflect(self) -> Tuple[str, float]:
        """Performs a cycle of autonomous creative reflection."""
        self._check_initialized()
        try:
            prompt = random.choice(self.reflection_prompts)
            self.logger.info("Creative reflection on: '%s'", prompt)

            messages = [{"role": "user", "content": prompt}]
            tokenized_chat = self.tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_tensors="pt").to(self.model.device)

            with torch.no_grad():
                 with torch.cuda.amp.autocast(enabled=self.use_amp):
                    outputs = self.model.generate(
                        tokenized_chat, 
                        max_new_tokens=100, 
                        temperature=self.temperature,
                        top_k=self.top_k, 
                        pad_token_id=self.tokenizer.pad_token_id
                    )
            
            prompt_tokens_len = tokenized_chat.shape[1]
            newly_generated_tokens = outputs[0][prompt_tokens_len:]
            reflection_text = self.tokenizer.decode(newly_generated_tokens, skip_special_tokens=True).strip()
            
            return reflection_text, 0.8
        except Exception as e:
            self.logger.error("An error occurred during creative reflection. Error: %s", e, exc_info=True)
            return "Failed to reflect creatively.", 0.0