import torch
from detection_head import DFL, DFL_FC_NoSlice_NoDiv


def test_dfl_equivalence():
    """Test that DFL and DFL_FC_NoSlice_NoDiv produce the same outputs."""

    # Initialize both models with same parameters
    c1 = 16
    num_classes = 1

    dfl_original = DFL(c1=c1, num_classes=num_classes)
    dfl_optimized = DFL_FC_NoSlice_NoDiv(c1=c1, num_classes=num_classes)

    # Set to evaluation mode
    dfl_original.eval()
    dfl_optimized.eval()

    # Create test inputs using actual shapes from the system:
    # [(1, 1600, 64), (1, 400, 64), (1, 100, 64), (1, 1600, 1), (1, 400, 1), (1, 100, 1), (1, 2100, 2)]
    torch.manual_seed(42)  # for reproducible results

    # Distance prediction inputs (first 3 tensors)
    x0 = torch.randn(1, 1600, 64)  # Shape from actual system
    x1 = torch.randn(1, 400, 64)   # Shape from actual system
    x2 = torch.randn(1, 100, 64)   # Shape from actual system

    # Class prediction inputs (next 3 tensors)
    x3 = torch.randn(1, 1600, 1)   # Shape from actual system
    x4 = torch.randn(1, 400, 1)    # Shape from actual system
    x5 = torch.randn(1, 100, 1)    # Shape from actual system

    print("Input shapes:")
    print(f"x0: {x0.shape}, x1: {x1.shape}, x2: {x2.shape}")
    print(f"x3: {x3.shape}, x4: {x4.shape}, x5: {x5.shape}")

    # Verify concatenation will work
    x_concat = torch.cat([x0, x1, x2], 1)
    y_concat = torch.cat([x3, x4, x5], 1)
    print(f"After concatenation: x={x_concat.shape}, y={y_concat.shape}")

    # Check what happens after permute
    x_permuted = torch.permute(x_concat, (0, 2, 1))
    y_permuted = torch.permute(y_concat, (0, 2, 1))
    print(f"After permute: x={x_permuted.shape}, y={y_permuted.shape}")

    # Debug the DFL expectations
    b, c, a = x_permuted.shape
    print(f"DFL will interpret as: b={b}, c={c}, a={a}")
    print(f"DFL will try to view as: ({b}, 4, {16}, {a}) = {b * 4 * 16 * a} elements")
    print(f"Actual elements in x_permuted: {x_permuted.numel()}")

    # Since original DFL seems to have wrong variable names, let's just test with the sizes it expects
    # Based on the view operation, it needs x.numel() == b * 4 * c1 * a
    # Currently we have (2, 2100, 64) -> needs to be reshaped to (2, 4, 16, 64)
    # This means a=64 (not 2100), so we need different anchor count

    # Run forward pass on both models
    with torch.no_grad():
        out1, y1 = dfl_original(x0, x1, x2, x3, x4, x5)
        out2, y2 = dfl_optimized(x0, x1, x2, x3, x4, x5)

    print("\nOutput shapes:")
    print(f"DFL output: {out1.shape}, y: {y1.shape}")
    print(f"DFL_FC_NoSlice_NoDiv output: {out2.shape}, y: {y2.shape}")

    # Compare outputs
    print("\nComparing outputs...")

    # Check if shapes match
    assert out1.shape == out2.shape, f"Output shapes don't match: {out1.shape} vs {out2.shape}"

    # Y shapes might be different due to different output conventions, check if they're equivalent
    # DFL returns (B, A, C), DFL_FC_NoSlice_NoDiv returns (B, C, A) - need to transpose one
    if y1.shape != y2.shape:
        if y1.shape == (y2.shape[0], y2.shape[2], y2.shape[1]):
            y1_normalized = y1.permute(0, 2, 1)  # Convert to same layout as y2
            print(f"Normalized y1 shape: {y1_normalized.shape} to match y2: {y2.shape}")
        else:
            raise AssertionError(f"Y shapes don't match and can't be normalized: {y1.shape} vs {y2.shape}")
    else:
        y1_normalized = y1

    # Check if values are close (allowing for small numerical differences)
    out_close = torch.allclose(out1, out2, rtol=1e-5, atol=1e-6)
    y_close = torch.allclose(y1_normalized, y2, rtol=1e-5, atol=1e-6)

    print(f"Outputs close (rtol=1e-5, atol=1e-6): {out_close}")
    print(f"Y values close (rtol=1e-5, atol=1e-6): {y_close}")

    if not out_close:
        diff = torch.abs(out1 - out2)
        print(f"Max difference in outputs: {diff.max().item()}")
        print(f"Mean difference in outputs: {diff.mean().item()}")

    if not y_close:
        diff_y = torch.abs(y1_normalized - y2)
        print(f"Max difference in y: {diff_y.max().item()}")
        print(f"Mean difference in y: {diff_y.mean().item()}")

    return out_close and y_close


def test_different_input_sizes():
    """Test with different input sizes to ensure robustness."""

    c1 = 16
    num_classes = 1

    dfl_original = DFL(c1=c1, num_classes=num_classes)
    dfl_optimized = DFL_FC_NoSlice_NoDiv(c1=c1, num_classes=num_classes)

    dfl_original.eval()
    dfl_optimized.eval()

    test_cases = [
        1, 2, 4,   # different batch sizes with same anchor structure
    ]

    for batch_size in test_cases:
        print(f"\nTesting batch_size={batch_size}")

        torch.manual_seed(42)

        # Use the same anchor structure as the real system but with different batch sizes
        x0 = torch.randn(batch_size, 1600, 64)
        x1 = torch.randn(batch_size, 400, 64)
        x2 = torch.randn(batch_size, 100, 64)

        x3 = torch.randn(batch_size, 1600, 1)
        x4 = torch.randn(batch_size, 400, 1)
        x5 = torch.randn(batch_size, 100, 1)

        with torch.no_grad():
            out1, y1 = dfl_original(x0, x1, x2, x3, x4, x5)
            out2, y2 = dfl_optimized(x0, x1, x2, x3, x4, x5)

        # Normalize y shapes if needed
        if y1.shape != y2.shape:
            if y1.shape == (y2.shape[0], y2.shape[2], y2.shape[1]):
                y1 = y1.permute(0, 2, 1)

        out_close = torch.allclose(out1, out2, rtol=1e-5, atol=1e-6)
        y_close = torch.allclose(y1, y2, rtol=1e-5, atol=1e-6)

        print(f"  Outputs equivalent: {out_close and y_close}")

        if not (out_close and y_close):
            print(f"  FAILED for batch_size={batch_size}, anchor_count={anchor_count}")
            return False

    return True


if __name__ == "__main__":
    print("Testing DFL vs DFL_FC_NoSlice_NoDiv equivalence...")

    # Test basic equivalence
    basic_test_passed = test_dfl_equivalence()
    print(f"\nBasic equivalence test: {'PASSED' if basic_test_passed else 'FAILED'}")

    # Test different input sizes
    size_test_passed = test_different_input_sizes()
    print(f"Different sizes test: {'PASSED' if size_test_passed else 'FAILED'}")

    # Final result
    all_tests_passed = basic_test_passed and size_test_passed
    print(f"\nOverall result: {'ALL TESTS PASSED' if all_tests_passed else 'SOME TESTS FAILED'}")