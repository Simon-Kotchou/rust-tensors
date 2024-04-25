import torch
import torch.nn.functional as F
import json

def generate_test_cases(num_cases):
    test_cases = []

    for _ in range(num_cases):
        # Generate random tensor sizes
        b = torch.randint(1, 10, (1,)).item()
        t = torch.randint(1, 20, (1,)).item()
        c = torch.randint(1, 100, (1,)).item()

        # Generate random inputs
        x = torch.randn(b, t, c, dtype=torch.float32)
        weight = torch.randn(c, dtype=torch.float32)
        bias = torch.randn(c, dtype=torch.float32)
        
        # Apply layer normalization using the functional version
        output = F.layer_norm(x, (c,), weight=weight, bias=bias, eps=1e-5)

        # Convert tensors to lists for JSON serialization
        test_case = {
            'input': x.tolist(),
            'weight': weight.tolist(),
            'bias': bias.tolist(),
            'output': output.tolist(),
            'b': b,
            't': t,
            'c': c
        }
        test_cases.append(test_case)

    return test_cases

# Generate test cases
num_test_cases = 20
test_cases = generate_test_cases(num_test_cases)

# Export test cases to JSON file
with open('layernorm_test_cases.json', 'w') as f:
    json.dump(test_cases, f)