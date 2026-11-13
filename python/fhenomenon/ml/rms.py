"""Algebraic RMS backward composite shared by capture and explicit lowering."""


def rms_norm_backward(grad, x, reciprocal_std, dims):
    """Use the same saved reciprocal_std, including its approximation error.

    This equals grad*r - x*r**3*mean(grad*x), without forming r**3.
    The interpreter lowers this function explicitly into ciphertext arithmetic.
    """
    normalized = x * reciprocal_std
    projection = (grad * normalized).mean(dim=dims, keepdim=True)
    return (grad - normalized * projection) * reciprocal_std
