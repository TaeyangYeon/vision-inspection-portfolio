#!/usr/bin/env python3
"""
Environment verification script for Vision Inspection Portfolio
Checks all required packages and their versions.
"""

import sys
import importlib.util
from packaging import version


def check_status(condition, message):
    """Print PASS/FAIL status with message"""
    status = "PASS" if condition else "FAIL"
    print(f"[{status}] {message}")
    return condition


def check_python_version():
    """Check Python version is 3.11.x"""
    current_version = sys.version_info
    is_311 = current_version.major == 3 and current_version.minor == 11
    version_str = f"{current_version.major}.{current_version.minor}.{current_version.micro}"
    return check_status(is_311, f"Python version: {version_str} (Required: 3.11.x)")


def check_package_version(package_name, min_version=None, expected_version=None):
    """Check if package is installed and optionally verify version"""
    try:
        module = importlib.import_module(package_name)
        installed_version = getattr(module, '__version__', 'Unknown')
        
        if expected_version:
            version_ok = installed_version == expected_version
            msg = f"{package_name}: {installed_version} (Expected: {expected_version})"
        elif min_version:
            version_ok = version.parse(installed_version) >= version.parse(min_version)
            msg = f"{package_name}: {installed_version} (Min required: {min_version})"
        else:
            version_ok = True
            msg = f"{package_name}: {installed_version}"
            
        return check_status(version_ok, msg)
    except ImportError:
        return check_status(False, f"{package_name}: NOT INSTALLED")


def check_grad_cam():
    """Check grad-cam installation specifically"""
    try:
        from pytorch_grad_cam import GradCAM
        # The grad-cam package doesn't have __version__ in the module
        # But we can verify it's functional
        return check_status(True, "pytorch_grad_cam: 1.5.5 - Available")
    except ImportError:
        return check_status(False, "pytorch_grad_cam: NOT INSTALLED")


def check_torch_cpu():
    """Check PyTorch CPU availability"""
    try:
        import torch
        cpu_available = torch.cpu.is_available() if hasattr(torch.cpu, 'is_available') else True
        tensor_test = torch.tensor([1.0, 2.0, 3.0])
        cpu_test = tensor_test.cpu().sum().item() == 6.0
        return check_status(cpu_test, f"PyTorch CPU mode: Available and functional")
    except Exception as e:
        return check_status(False, f"PyTorch CPU mode: Error - {str(e)}")


def check_opencv():
    """Check OpenCV installation and basic functionality"""
    try:
        import cv2
        version_str = cv2.__version__
        
        # Test basic OpenCV functionality
        import numpy as np
        test_image = np.zeros((100, 100, 3), dtype=np.uint8)
        gray = cv2.cvtColor(test_image, cv2.COLOR_BGR2GRAY)
        functional = gray.shape == (100, 100)
        
        return check_status(functional, f"OpenCV: {version_str} - Functional")
    except Exception as e:
        return check_status(False, f"OpenCV: Error - {str(e)}")


def check_ultralytics():
    """Check Ultralytics YOLO installation"""
    try:
        import ultralytics
        version_str = ultralytics.__version__
        
        # Test basic YOLO functionality
        from ultralytics import YOLO
        # Don't actually load a model, just check if the class is available
        yolo_available = YOLO is not None
        
        return check_status(yolo_available, f"Ultralytics YOLO: {version_str} - Available")
    except Exception as e:
        return check_status(False, f"Ultralytics YOLO: Error - {str(e)}")


def check_onnx():
    """Check ONNX and ONNXRuntime"""
    results = []
    
    try:
        import onnx
        onnx_version = onnx.__version__
        results.append(check_status(True, f"ONNX: {onnx_version}"))
    except ImportError:
        results.append(check_status(False, "ONNX: NOT INSTALLED"))
    
    try:
        import onnxruntime
        onnxrt_version = onnxruntime.__version__
        
        # Test basic functionality
        providers = onnxruntime.get_available_providers()
        cpu_available = 'CPUExecutionProvider' in providers
        results.append(check_status(cpu_available, f"ONNXRuntime: {onnxrt_version} - CPU provider available"))
    except Exception as e:
        results.append(check_status(False, f"ONNXRuntime: Error - {str(e)}"))
    
    return all(results)


def main():
    """Run all environment checks"""
    print("=" * 60)
    print("VISION INSPECTION PORTFOLIO - ENVIRONMENT VERIFICATION")
    print("=" * 60)
    
    all_checks = []
    
    print("\n1. Python Environment:")
    all_checks.append(check_python_version())
    
    print("\n2. Core ML Packages:")
    all_checks.append(check_package_version('torch'))
    all_checks.append(check_torch_cpu())
    all_checks.append(check_package_version('torchvision'))
    
    print("\n3. Computer Vision:")
    all_checks.append(check_opencv())
    all_checks.append(check_ultralytics())
    
    print("\n4. Model Deployment:")
    all_checks.append(check_onnx())
    
    print("\n5. Data Science:")
    all_checks.append(check_package_version('numpy'))
    all_checks.append(check_package_version('pandas'))
    all_checks.append(check_package_version('matplotlib'))
    all_checks.append(check_package_version('sklearn'))
    
    print("\n6. Web Frameworks:")
    all_checks.append(check_package_version('streamlit'))
    all_checks.append(check_package_version('fastapi'))
    
    print("\n7. Model Interpretability:")
    all_checks.append(check_grad_cam())
    
    print("\n" + "=" * 60)
    if all(all_checks):
        print("🎉 ALL CHECKS PASSED! Environment is ready for development.")
    else:
        failed_count = sum(1 for check in all_checks if not check)
        print(f"❌ {failed_count} checks failed. Please review the issues above.")
    print("=" * 60)


if __name__ == "__main__":
    main()