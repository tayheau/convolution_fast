 
import numpy as np 
from typing import Tuple
import matplotlib.pyplot as plt

def convolution_1D(input:np.array, kernel:np.array) -> np.array:
    """
    Naïve implementation of the 1D convolution.

    This function performs 1D convolution between the input array 
    and the kernel array by sliding the kernel over the input and computing the dot product 
    at each position.

    Args:
        input (np.array): The 1D input array over which the kernel will be applied.
        kernel (np.array): The 1D kernel (filter) that will be convolved with the input. 
                           Must be smaller than or equal in length to the input array.

    Returns:
        np.array: The 1D output array containing the result of the convolution.

    Raises:
        AssertionError: If the kernel is larger than the input.

    Example:
        >>> input = np.array([1, 2, 3, 4])
        >>> kernel = np.array([5, 6, 6])
        >>> convolution_1D(input, kernel)
        array([5, 16, 34, 52, 45, 28])
    """
    assert len(kernel) <= len(input), 'Kernel must be smaller than Input'
    kernel = kernel[::-1]
    return np.array([
        (np.dot(
            input[max(0, i):min(i+len(kernel), len(input))],
            kernel[max(-i, 0):min(len(kernel), len(kernel)-i+1)]
        )) for i in range(1-len(kernel), len(input))
    ])


def convolution_2D(input:np.array, kernel:np.array, stride:Tuple[int, int] = (1, 1)) -> np.array:
    """
    Naïve implementation of 2D cross-correlation (called convolution).

    This function slides the kernel over the input matrix, computing element-wise product and sum for each region.
    Note: this is technically cross-correlation since the kernel is not flipped.

    Args:
        input (np.array): 2D input array over which the kernel is applied.
        kernel (np.array): 2D array (filter) that is applied to the input.
                           Must be smaller than the input array.
        stride (Tuple[int, int], optional): The stride of the convolution as (rows, columns) steps. Defaults to (1, 1)

    Returns:
        np.array: 2D feature map resulting from the cross-correlation between the 'input' and the 'kernel'.
    """
    input_shape = input.shape
    kernel_shape = kernel.shape

    output_shape = ((input_shape[0] - kernel_shape[0])//stride[0] + 1,
                    (input_shape[1] - kernel_shape[1])//stride[1] + 1)
    
    new = np.zeros((output_shape[0], output_shape[1]))
    for row in range(0, output_shape[0]):
        for col in range(0, output_shape[1]):
            new[row, col] = np.sum(input[row * stride[0]:row *stride[0] + kernel_shape[0], col * stride[1]:col * stride[1]+kernel_shape[1]]*kernel).astype(np.float32)
    return new


def im2col(input: np.array, k_h: int, k_w:int, stride: Tuple[int, int] = (1, 1)) -> Tuple[int, int, np.array]:
    """
        Process to rearrange image blocks into columns in view of performing a 2d convolution by matrix multiplication.

        Args:
            input (np.array): 3D input array over which the kernel is applied. (H, W, C)
            k_w (int): kernel width
            k_h (int) : kernel height
            stride (Tuple[int, int], optinal): strinde of the convolution as (rows, columns) steps. Default to (1, 1).

        Returns:
            np.array: 2D reshaped input 

        Note:
            im2col does not improve the time complexity of Convolution, it improves real time performances as it enable the use of GEMM.
    """
    input = np.atleast_3d(input)
    h, w, c = input.shape
    new_h = (h - k_h) // stride[0] + 1
    new_w = (w - k_w) // stride[1] + 1
    res = np.zeros((new_w*new_h, k_h*k_w*c))
    for j in range(new_h):
        for i in range(new_w):
            temp = input[j*stride[0]:j*stride[0]+k_h, i*stride[1]:i*stride[1]+k_w, :]
            res[j*new_w+i,:] = np.reshape(temp, -1)
    return new_h, new_w, res


def matrix_convolution_2d(input: np.array, kernel:np.array, stride:Tuple[int, int] = (1, 1)) -> np.array:
    """
    Performs 2D convolution using matrix multiplication (GEMM) after rearranging input image with im2col.
    
    Args:
        input (np.array): 3D input array (height, width, channels).
        kernel (np.array): 2D kernel to be applied to the input.
        stride (Tuple[int, int], optional): Stride for convolution in the form (row_step, col_step). Defaults to (1, 1).

    Returns:
        np.array: 2D output of the convolution.

    Note:
        The kernel is flipped before applying the convolution, mimicking traditional cross-correlation operations.
    """
    h, w, res = im2col(input, kernel.shape[0], kernel.shape[1])
    kernel_flatten = np.flip(kernel).reshape(-1)
    return np.reshape(res @ kernel_flatten, (h, w))

def im2col_strideTrick_2D(input: np.ndarray, kernel_shape: tuple) -> Tuple[Tuple[int, int], np.ndarray]:
    """
    Use stride tricks to implement im2col without data duplication.
    
    Args:
        input (np.array): Input image array.
        kernel_shape (tuple): Kernel shape (height, width).

    Returns:
        Tuple[Tuple[int, int], np.ndarray]: Output dimensions and rearranged matrix.
    """
    input_strides = input.strides
    new = np.array(input.shape) - np.array(kernel_shape) + 1
    output_shape = tuple(new) + kernel_shape
    result =  np.lib.stride_tricks.as_strided(input, shape=output_shape ,strides=(input_strides[0] ,input_strides[1], input_strides[0], input_strides[1]))
    result = result.transpose(2, 3, 0, 1)
    return tuple(new), result.reshape(kernel_shape[0] * kernel_shape[1], -1).T


def memstrided_matrix_convolution_2d(input: np.array, kernel:np.array) -> np.array:
    """
    Perform 2D convolution using matrix multiplication with im2col implemented via stride tricks.

    Args:
        input (np.array): Input image.
        kernel (np.array): Convolutional kernel.

    Returns:
        np.array: Output after convolution.
    """
    dim, res = im2col_strideTrick_2D(input, kernel.shape)
    kernel_flatten = np.flip(kernel).reshape(-1)
    return np.reshape(res @ kernel_flatten, dim)
