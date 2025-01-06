import sys
import torch
import numpy as np

def check_cuda_installation():
    """
    Comprehensive CUDA installation verification script.
    Tests PyTorch CUDA availability and performs basic operations.
    """
    print("=== Python Environment ===")
    print(f"Python version: {sys.version}")
    
    print("\n=== PyTorch Installation ===")
    print(f"PyTorch version: {torch.__version__}")
    
    print("\n=== CUDA Availability ===")
    print(f"CUDA available: {torch.cuda.is_available()}")
    
    if torch.cuda.is_available():
        print(f"CUDA version: {torch.version.cuda}")
        print(f"Current CUDA device: {torch.cuda.current_device()}")
        print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
        
        # Test basic CUDA operations
        print("\n=== Testing CUDA Operations ===")
        try:
            # Create a tensor on CPU
            x = torch.rand(5, 3)
            print("Created CPU tensor:")
            print(x)
            
            # Move tensor to GPU
            cuda_x = x.cuda()
            print("\nMoved tensor to GPU:")
            print(cuda_x)
            
            # Perform a simple operation
            cuda_y = cuda_x * 2
            print("\nPerformed multiplication on GPU:")
            print(cuda_y)
            
            print("\nCUDA operations test: PASSED")
            
        except Exception as e:
            print(f"\nError during CUDA operations: {str(e)}")
    else:
        print("\nCUDA is not available. Please check your installation.")
        print("\nPossible issues to check:")
        print("1. NVIDIA GPU drivers are installed")
        print("2. CUDA toolkit is installed and in PATH")
        print("3. PyTorch is installed with CUDA support")
        print("4. Environment variables are set correctly")

if __name__ == "__main__":
    check_cuda_installation()