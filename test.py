import numpy as np
import math
import matplotlib as plt
def percentage_error(a, b):
    def sigmoid(x, center, scale):
        return 1 / (1 + np.exp(-(x - center) / scale))
    def smooth_output(a, b):
        # Sigmoid function parameters
        center_high = 100
        center_low = 20
        scale = (center_high - center_low) / 4  # Scale factor to control smoothness
    
        # Calculate sigmoid values for a and b
        sigmoid_a = sigmoid(a, center_high, scale)
        sigmoid_b = sigmoid(b, center_high, scale)
    
        # Invert sigmoid values to get the desired behavior:
        # Close to 0 when a or b is greater than 100, close to 1 when both are less than 20
        inverted_sigmoid_a = 1 - sigmoid_a
        inverted_sigmoid_b = 1 - sigmoid_b
    
        # Combine sigmoid values (using multiplication to enforce both conditions)
        # and scale to the desired output range (0 to 1)
        combined = inverted_sigmoid_a * inverted_sigmoid_b
    
        # Since we want a smooth transition from 1 (both < 20) to 0 (both > 100),
        # we don't need any additional scaling if the sigmoid is properly tuned.
        # However, we can apply a final scaling if necessary.
    
        # Note: The output will never be exactly 0 or 1 due to the nature of the sigmoid function,
        # but it will be very close to these values when a and b are far from the center points.
        return combined


    if a == b:
        return 1
    else:
        ans = smooth_output(a, b) + (1- abs(a-b)/max(a,b))*(1-smooth_output(a, b))
        if ans > 1:
            return 1
        else :
            return ans


print(f'{percentage_error(30,20):.4f}')
