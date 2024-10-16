import numpy as np 
import matplotlib.pyplot as plt 
import convolution
from typing import List
import argparse
from PIL import Image



def show_images(images: List[np.ndarray], **kwargs) -> None:
    n: int = len(images)
    f = plt.figure()
    for i in range(n):
        f.add_subplot(1, n, i + 1)
        plt.imshow(images[i], **kwargs)
    plt.show(block=True)

def get_image_greyscale(image: str) -> np.array:
    """
    grayscale = 0.2989 R + 0.5870 G + 0.1140 B
    """
    img = Image.open(image).convert('L')
    return np.array(img, dtype='float')

def convolve_visual(image:np.array, kernel:List[np.array]) -> None:
    processed = [convolution.memstrided_matrix_convolution_2d(image, i) for i in kernel]
    final = np.sqrt(np.sum(np.power(processed, 2), axis=(0,))) if len(processed) > 1 else processed[0]
    show_images([image, final], cmap='gray')



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Apply a filter through convolution to a given image.')
    parser.add_argument('--image_file', type=str, default='tayheau.jpeg', help='Input image file path')
    parser.add_argument('--filter', type=str, default='edges_extract', help='"edges_extract" to extract edges of the picture, "gaussian" to apply a gaussian blur, "invert" to invert colors')
    args = parser.parse_args()
    
    img = get_image_greyscale(args.image_file)
    #Prewitt operators
    v_edges =  np.array([[-1, 0, 1], 
                        [-1, 0, 1], 
                        [-1, 0, 1]])

    h_edges = np.array([[-1, -1, -1], 
                        [0, 0, 0], 
                        [1, 1, 1]])
    
    blurring_kernel = 1/256 * np.array([[1, 4, 6, 4, 1], 
                                   [4, 16, 24, 16, 4], 
                                   [6, 24, 36, 24, 6], 
                                   [4, 16, 24, 16, 4], 
                                   [1, 4, 6, 4, 1]])
    
    invert = np.array([[0, 0, -1], 
                     [0, 1, 0], 
                     [-1, 0, 0]])
    
    if args.filter == 'edges_extract':
        filtered_img = convolve_visual(img, [v_edges, h_edges])
    elif args.filter == 'invert':
        filtered_img = convolve_visual(img, [invert])
    elif args.filter == 'gaussian':
        filtered_img = convolve_visual(img, [blurring_kernel])
