
# def add(num1, num2):
#     "Simple addition tool for nanoGPT"
#     return num1 + num2

# calculator.py
def add(num1, num2):
    "Simple addition tool for nanoGPT"
    return num1 + num2

def evaluate(expression: str):
    """Evaluates a math expression string safely or falls back to eval"""
    try:
        # Agar simple addition format hai jaise "45+55"
        if "+" in expression:
            parts = expression.split("+")
            return add(float(parts[0]), float(parts[1]))
        return eval(expression)
    except Exception as e:
        return f"Error: {e}"

