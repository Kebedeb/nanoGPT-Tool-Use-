# stair_memory.py
from typing import Dict, Any

class MemoryStore:
    def __init__(self):
        # The categorized storage buckets
        self.data: Dict[str, Any] = {
            "variables": {},      # key-value pairs (e.g., {"x": 42})
            "calculations": [],   # history of operations (e.g., ["12 * 4 = 48"])
            "notes": []           # arbitrary factual notes
        }

    def get_toc(self) -> str:
        """
        Return a compact ToC with calculation records represented as leaves.
        """
        lines = ["[TABLE OF CONTENTS]"]
        lines.append(f" - variables ({len(self.data['variables'])} items)")
        calculations = self.data["calculations"]
        lines.append(f" - calculations ({len(calculations)} items)")
        for index, calculation in enumerate(calculations, start=1):
            expression = calculation.split(" = ", 1)[0]
            lines.append(f"   leaf calculation_{index}: {expression}")
        lines.append(f" - notes ({len(self.data['notes'])} items)")
        lines.append("[END TABLE OF CONTENTS]")
        return "\n".join(lines)

    def fetch_category(self, category: str) -> str:
        """Retrieves only the requested bucket."""
        if category in self.data:
            return f"[{category.upper()} CONTENT]: {self.data[category]}"
        return f"[ERROR]: Category '{category}' does not exist."

    def get_calculations(self) -> str:
        """Return calculation history in a stable, readable form."""
        calculations = self.data["calculations"]
        if not calculations:
            return "No calculations have been saved yet."
        return "\n".join(
            f"{index}. {calculation}"
            for index, calculation in enumerate(calculations, start=1)
        )

    def get_calculation(self, index: int = -1) -> str | None:
        """Get one saved calculation by zero-based index (default: most recent)."""
        calculations = self.data["calculations"]
        if not calculations:
            return None
        try:
            return calculations[index]
        except IndexError:
            return None

    def add_variable(self, name: str, val: Any):
        self.data["variables"][name] = val

    def add_calculation(self, expr: str, res: Any):
        self.data["calculations"].append(f"{expr} = {res}")

    def add_note(self, text: str):
        self.data["notes"].append(text)
