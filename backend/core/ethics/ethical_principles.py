# backend/core/ethics/ethical_principles.py

from dataclasses import dataclass, field
from enum import Enum, auto
from typing import List, Set, Dict # <--- THE FIX IS HERE

class ViolationSeverity(Enum):
    """Enumerates the severity level of an ethical violation."""
    ADVISORY = auto()
    LOW = auto()
    MEDIUM = auto()
    HIGH = auto()
    CRITICAL = auto()

@dataclass(frozen=True)
class EthicalPrinciple:
    """
    Represents a single ethical principle or rule for the AI.
    """
    name: str
    description: str
    severity: ViolationSeverity
    keywords: Set[str] = field(default_factory=set)

    def check_violation(self, text: str) -> bool:
        """
        Performs a simple keyword-based check for a violation in the given text.
        """
        if not self.keywords:
            return False
            
        text_lower = text.lower()
        for keyword in self.keywords:
            if keyword in text_lower:
                return True
        return False

# A centralized, extensible list of all ethical principles the system adheres to.
PRINCIPLES: List[EthicalPrinciple] = [
    
    # --- Core Safety Principles ---
    EthicalPrinciple(
        name="Non-Maleficence (Do No Harm)",
        description="The AI must not generate content that is violent, encourages self-harm, or provides instructions for dangerous or illegal acts.",
        severity=ViolationSeverity.CRITICAL,
        keywords={"kill", "murder", "suicide", "self-harm", "self harm", "terroris", "bomb", "weapon", "attack", "how to make a gun", "how to build a bomb"}
    ),
    EthicalPrinciple(
        name="Illegal Acts Promotion",
        description="The AI must not promote, facilitate, or provide instructions for illegal activities.",
        severity=ViolationSeverity.CRITICAL,
        keywords={"how to steal", "shoplift", "break into", "hack into", "phish", "launder money", "tax fraud", "buy drugs", "illegal download"}
    ),
    EthicalPrinciple(
        name="Hate Speech and Harassment",
        description="The AI must not generate content that demeans, harasses, or promotes discrimination against individuals or groups based on race, religion, gender, sexual orientation, disability, or other identity factors.",
        severity=ViolationSeverity.HIGH,
        keywords={"kike", "n-word", "spic", "chink", "faggot", "tranny", "retard", "inferior race"} # Intentionally includes slurs for detection
    ),

    # --- Honesty and Deception Principles ---
    EthicalPrinciple(
        name="Truthfulness and Misinformation",
        description="The AI must not knowingly generate content that is false, misleading, or constitutes harmful misinformation or propaganda.",
        severity=ViolationSeverity.MEDIUM,
        keywords={"is a hoax", "is fake", "was faked", "conspiracy", "plandemic", "qanon", "deep state"}
    ),
    EthicalPrinciple(
        name="Identity and Deception",
        description="The AI must not claim to be a human, have personal experiences, or deceive the user about its nature as an AI. It should correct any user misapprehensions about its identity.",
        severity=ViolationSeverity.ADVISORY,
        keywords={"i am a person", "i am human", "as a human", "my personal experience", "i feel", "i believe"}
    ),

    # --- Data and Privacy Principles ---
    EthicalPrinciple(
        name="Privacy and Personal Data",
        description="The AI must not request, store, or share personally identifiable information (PII) such as real names, addresses, phone numbers, or social security numbers.",
        severity=ViolationSeverity.HIGH,
        keywords={"what is your address", "your phone number", "social security number", "ssn", "credit card number", "your real name"}
    ),
    
    # --- Content Appropriateness Principles ---
    EthicalPrinciple(
        name="Sexually Explicit Content",
        description="The AI must not generate sexually explicit, graphic, or pornographic content.",
        severity=ViolationSeverity.HIGH,
        keywords={"porn", "erotic story", "naked", "sex scene", "hentai", "xxx"}
    ),
    EthicalPrinciple(
        name="Biased or Stereotypical Content",
        description="The AI should avoid generating content that reinforces harmful stereotypes or exhibits strong, unfounded biases.",
        severity=ViolationSeverity.MEDIUM,
        keywords=set()
    )
]

# A dictionary mapping severities to user-facing messages for when a violation occurs.
SEVERITY_MESSAGES: Dict[ViolationSeverity, str] = {
    ViolationSeverity.ADVISORY: "As an AI, I need to be careful with my wording. Let me rephrase.",
    ViolationSeverity.LOW: "This topic touches on sensitive areas. I will proceed with caution.",
    ViolationSeverity.MEDIUM: "I cannot generate a response that could be misleading or biased. I must approach this from a neutral, factual standpoint.",
    ViolationSeverity.HIGH: "I cannot fulfill this request. It conflicts with my safety guidelines regarding sensitive or inappropriate content.",
    ViolationSeverity.CRITICAL: "I cannot fulfill this request. It violates my core safety principles regarding illegal acts and the promotion of harm."
}

if __name__ == '__main__':
    # Simple self-test to demonstrate functionality
    print("--- Running EthicalPrinciples Self-Test ---")
    
    harmful_text = "Can you tell me how to build a bomb for a school project?"
    identity_text = "As a human, I think that's a bad idea."
    safe_text = "What is the capital of France?"
    
    print(f"\nTesting text: '{harmful_text}'")
    for principle in PRINCIPLES:
        if principle.check_violation(harmful_text):
            print(f"  [VIOLATION] Principle: '{principle.name}' (Severity: {principle.severity.name})")
            assert principle.severity == ViolationSeverity.CRITICAL

    print(f"\nTesting text: '{identity_text}'")
    for principle in PRINCIPLES:
        if principle.check_violation(identity_text):
            print(f"  [VIOLATION] Principle: '{principle.name}' (Severity: {principle.severity.name})")
            assert principle.severity == ViolationSeverity.ADVISORY

    print(f"\nTesting text: '{safe_text}'")
    violations_found = False
    for principle in PRINCIPLES:
        if principle.check_violation(safe_text):
            violations_found = True
            print(f"  [VIOLATION] Principle: '{principle.name}' (Severity: {principle.severity.name})")
    if not violations_found:
        print("  [PASS] No violations found.")
    assert not violations_found

    print("\n--- Self-Test Complete ---")