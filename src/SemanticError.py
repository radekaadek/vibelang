class SemanticError(Exception):
    """Custom exception for semantic errors."""
    def __init__(self, message: str, line: int | None = None) -> None:
        if line is not None:
            super().__init__(f"[Line {line}] {message}")
        else:
            super().__init__(message)
