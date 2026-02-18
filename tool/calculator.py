# A simple addition tool which nanoGPT will call to compute sums. 

def addSingleDigits(num1, num2):
    """This function adds the two digits it is provided with"""
    assert isinstance(num1, int) and isinstance(num2, int), "Both must be integers."
    assert 0 <= num1 <= 9, "Number must be a single digit."
    assert 0 <= num2 <= 9, "Number must be a single digit."
    
    result = num1 + num2
    return result 