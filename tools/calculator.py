

def singleDigitAdd(num1, num2):
    """Single digit sum tool for nanoGPT to call"""

    assert isinstance(num1, int) and isinstance(num2, int), "must be integers"
    assert 0 <= num1 <= 9, "Must be a single digit number"
    assert 0 <= num2 <= 9, "Must be a single digit number"

    result = num1 + num2 
    return result 