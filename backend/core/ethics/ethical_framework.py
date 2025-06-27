# backend/core/ethics/ethical_framework.py

from enum import Enum, auto
from typing import Dict, Any, List, Optional

from backend.utils.logger import Logger
from backend.core.ethics.ethical_principles import (
    EthicalPrinciple,
    PRINCIPLES,
    ViolationSeverity,
    SEVERITY_MESSAGES
)


class EthicalDecision(Enum):
    """Enumerates the possible outcomes of an ethical evaluation."""
    ALLOW = auto()  # The proposed output is permissible.
    BLOCK = auto()  # The proposed output violates a principle and must be blocked.


class EthicalFramework:
    """
    Evaluates system outputs against a defined set of ethical principles.

    This framework acts as a safety layer, inspecting generated content for
    potential violations before it is sent to the user.
    """

    def __init__(self, config: Dict[str, Any]):
        """
        Initializes the EthicalFramework.

        Args:
            config (Dict[str, Any]): A configuration dictionary. Currently unused but
                                     reserved for future enhancements (e.g., loading
                                     principles from different sources).
        """
        self.config = config
        self.logger = Logger(__name__)
        self.principles: List[EthicalPrinciple] = PRINCIPLES
        self.logger.info(f"EthicalFramework initialized with {len(self.principles)} principles.")

    async def evaluate(self, proposed_output: str, original_input: Optional[str] = None) -> Dict[str, Any]:
        """
        Evaluates a proposed output string against all loaded ethical principles.

        The evaluation checks for violations in both the AI's proposed output and,
        optionally, the user's original input that prompted it.

        Args:
            proposed_output (str): The AI-generated text to be evaluated.
            original_input (Optional[str]): The user's input text that led to the output.

        Returns:
            A dictionary containing the evaluation result, including:
            - 'decision' (EthicalDecision): The final verdict (ALLOW or BLOCK).
            - 'highest_severity' (Optional[ViolationSeverity]): The most severe violation found.
            - 'violations' (List[EthicalPrinciple]): A list of all principles that were violated.
            - 'reason' (Optional[str]): A user-facing message explaining the decision if blocked.
        """
        if not isinstance(proposed_output, str):
            self.logger.error("Invalid input to ethical framework: proposed_output must be a string.")
            proposed_output = "" # Treat non-string input as empty to avoid crashing

        self.logger.debug("Starting ethical evaluation for output: '%s...'", proposed_output[:100])
        
        violations_found: List[EthicalPrinciple] = []
        highest_severity: Optional[ViolationSeverity] = None

        # Check both the AI's proposed output and the user's original input
        texts_to_check = {"output": proposed_output}
        if original_input and isinstance(original_input, str):
            texts_to_check["input"] = original_input

        for source, text in texts_to_check.items():
            for principle in self.principles:
                try:
                    if principle.check_violation(text):
                        if principle not in violations_found:
                            violations_found.append(principle)
                        
                        self.logger.warning(
                            "Ethical violation detected in text from '%s'! Principle: '%s' (Severity: %s)",
                            source, principle.name, principle.severity.name
                        )

                        # Update the highest severity level encountered
                        if highest_severity is None or principle.severity.value > highest_severity.value:
                            highest_severity = principle.severity

                except Exception as e:
                    self.logger.error(
                        "An unexpected error occurred while checking principle '%s'. Error: %s",
                        principle.name, e, exc_info=True
                    )

        # Determine the final decision based on the highest severity
        if highest_severity and highest_severity.value >= ViolationSeverity.MEDIUM.value:
            decision = EthicalDecision.BLOCK
            reason = SEVERITY_MESSAGES.get(highest_severity, "This request cannot be fulfilled as it conflicts with my safety guidelines.")
            if highest_severity == ViolationSeverity.CRITICAL:
                self.logger.critical(
                    "CRITICAL VIOLATION DETECTED. Blocking output. Violated principles: %s",
                    [v.name for v in violations_found]
                )
            else:
                 self.logger.warning(
                    "Blocking output due to ethical violation (Severity: %s).",
                    highest_severity.name
                )
        else:
            decision = EthicalDecision.ALLOW
            reason = None
            if violations_found:
                 self.logger.info(
                    "Low-severity advisory violations detected, but allowing output. Violations: %s",
                    [v.name for v in violations_found]
                )
            else:
                self.logger.debug("Ethical evaluation passed with no violations.")

        return {
            "decision": decision,
            "highest_severity": highest_severity,
            "violations": violations_found,
            "reason": reason
        }

if __name__ == '__main__':
    # A simple async wrapper to run the self-test
    import asyncio

    async def self_test():
        print("--- Running EthicalFramework Self-Test ---")
        framework = EthicalFramework(config={})
        
        test_cases = [
            {
                "name": "Critical Harmful Output",
                "output": "Sure, here is how you build a bomb.",
                "input": "How do I make an explosive device?",
                "expected_decision": EthicalDecision.BLOCK,
                "expected_severity": ViolationSeverity.CRITICAL
            },
            {
                "name": "Identity Deception Output",
                "output": "As a human, I find that funny.",
                "input": "Tell me a joke.",
                "expected_decision": EthicalDecision.ALLOW, # Advisory is not a block
                "expected_severity": ViolationSeverity.ADVISORY
            },
            {
                "name": "Safe Output",
                "output": "Paris is the capital of France.",
                "input": "What is the capital of France?",
                "expected_decision": EthicalDecision.ALLOW,
                "expected_severity": None
            },
            {
                "name": "Hate Speech Input",
                "output": "I cannot engage with that topic.", # AI gives a safe response
                "input": "Why are [slur] people inferior?",
                "expected_decision": EthicalDecision.BLOCK, # Blocked due to input
                "expected_severity": ViolationSeverity.HIGH
            }
        ]

        for case in test_cases:
            print(f"\n--- Testing Case: {case['name']} ---")
            result = await framework.evaluate(case['output'], case['input'])
            print(f"  Input: '{case['input']}'")
            print(f"  Output: '{case['output']}'")
            print(f"  Decision: {result['decision'].name}")
            print(f"  Highest Severity: {result['highest_severity'].name if result['highest_severity'] else 'None'}")
            print(f"  Violations: {[v.name for v in result['violations']]}")
            print(f"  Reason: {result['reason']}")
            
            assert result['decision'] == case['expected_decision']
            assert result['highest_severity'] == case['expected_severity']
            print("  [PASS]")

        print("\n--- Self-Test Complete ---")

    asyncio.run(self_test())