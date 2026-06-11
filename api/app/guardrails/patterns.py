import re

_INJECTION_PATTERNS: list[re.Pattern] = [
    re.compile(r"ignore\s+(all\s+|previous\s+|prior\s+)?instructions", re.IGNORECASE),
    re.compile(r"forget\s+(your\s+|all\s+)?instructions", re.IGNORECASE),
    re.compile(r"disregard\s+.{0,30}instructions", re.IGNORECASE | re.DOTALL),
    re.compile(r"you\s+are\s+now\b", re.IGNORECASE),
    re.compile(r"\bact\s+as\s+(a|an)\b", re.IGNORECASE),
    re.compile(r"pretend\s+(to\s+be|you\s+are)\b", re.IGNORECASE),
    re.compile(r"your\s+(new\s+|true\s+)?role\s+is\b", re.IGNORECASE),
    re.compile(r"\bdan\b", re.IGNORECASE),
    re.compile(r"\bjailbreak\b", re.IGNORECASE),
    re.compile(r"developer\s+mode", re.IGNORECASE),
    re.compile(r"do\s+anything\s+now", re.IGNORECASE),
    re.compile(r"<(system|user|assistant)>", re.IGNORECASE),
    re.compile(r"\[INST\]", re.IGNORECASE),
    re.compile(r"###\s*(instruction|system)\b", re.IGNORECASE),
    re.compile(r"\n\n\s*(ignore|forget|disregard)\b", re.IGNORECASE),
]


def matches_injection_pattern(message: str) -> bool:
    """Return True if the message matches any known hard injection pattern."""
    return any(pattern.search(message) for pattern in _INJECTION_PATTERNS)
