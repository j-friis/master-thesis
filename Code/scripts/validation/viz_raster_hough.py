import argparse
import laspy
import numpy as np

from skimage.transform import (hough_line, hough_line_peaks,
                               probabilistic_hough_line)
from skimage.feature import canny

from matplotlib import cm
from mpl_toolkits.axes_grid1 import ImageGrid

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import cv2
import laspy
from os import listdir
from os.path import isfile, join

def plot_res(file: str):
    #fig, axs = plt.subplots(2, 2)

    image = cv2.imread(file, cv2.IMREAD_UNCHANGED)
    image = np.where(image >= 0, image, 0)
    image = image/np.max(image)

    image = (image*255).astype(np.uint8)
    # plt.title("Original Image")
    # plt.imshow(image, cmap='gray')
    # plt.show()

    kernel = np.ones((70,70),np.uint8)
    closing = cv2.morphologyEx(image, cv2.MORPH_CLOSE, kernel)

    # plt.title("Closing")
    # plt.imshow(closing, cmap='gray')
    # plt.show()

    # Apply edge detection method on the image
    edges = cv2.Canny(closing, 4, 160, None, 3)

    # plt.title("Edges")
    # plt.imshow(edges, cmap='gray')
    # plt.show()

    # Probabilistic Line Transform
    # min_line_length, max_line_gap

    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 100)

    lines_image = np.zeros_like(image)
    # Draw the lines
    if linesP is not None:
        for i in range(0, len(linesP)):
            l = linesP[i][0]
            cv2.line(lines_image, (l[0], l[1]), (l[2], l[3]), (255,0,0), 3)

    # plt.title("Hough Lines")
    # plt.imshow(lines_image, cmap='gray')
    # plt.show()

    kernel = np.ones((5,5),np.uint8)
    dilation = cv2.dilate(lines_image, kernel, iterations = 7)

    # plt.imshow(dilation, cmap='gray')
    # plt.title("Dialation")
    # plt.show()

    file_name = file.split('/')[-1]
    tile_name = file_name.split("_height_filtered",1)[0]
    plot_name = tile_name+".laz"

    fig, axs = plt.subplots(2, 2, figsize=(10, 10))
    fig.suptitle(plot_name, fontsize=16)
    axs[0, 0].imshow(image, cmap='gray')
    axs[0, 0].set_title('Image')
    axs[0, 1].imshow(closing, cmap='gray')
    axs[0, 1].set_title('Closing')
    axs[1, 0].imshow(edges, cmap='gray')
    axs[1, 0].set_title('Edges')
    axs[1, 1].imshow(dilation, cmap='gray')
    axs[1, 1].set_title('Dilation')

    # for ax in axs.flat:
    #     ax.set(xlabel='x-label', ylabel='y-label')

    # # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    plt.setp([a.get_xticklabels() for a in axs[0, :]], visible=False)
    plt.setp([a.get_yticklabels() for a in axs[:, 1]], visible=False)

    # Hide x labels and tick labels for top plots and y ticks for right plots.
    for ax in axs.flat:
        ax.label_outer()

    # Tight layout often produces nice results
    # but requires the title to be spaced accordingly
    fig.tight_layout()
    fig.subplots_adjust(top=0.88)
    plt.show()
    # fig = plt.figure(figsize=(4., 4.))
    # grid = ImageGrid(fig, 111,  # similar to subplot(111)
    #                 nrows_ncols=(2, 2),  # creates 2x2 grid of axes
    #                 axes_pad=0.1,  # pad between axes in inch.
    #                 )
    # i = 0
    # for ax, im in zip(grid, [image, closing, edges, dilation]):
    #     # Iterating over the grid returns the Axes.
    #     ax.imshow(im)
    #     #ax.title(i)
    #     i = i+1

    # plt.show()

    # # Simple data to display in various forms
    # x = np.linspace(0, 2 * np.pi, 400)
    # y = np.sin(x ** 2)

    # fig, axarr = plt.subplots(2, 2)
    # print(file.split('/')[-1])
    # fig.suptitle(file.split('/')[0], fontsize=16)
    # axarr[0, 0].imshow(image, cmap='gray')
    # axarr[0, 0].set_title('Image')
    # axarr[0, 1].imshow(closing, cmap='gray')
    # axarr[0, 1].set_title('closing')
    # axarr[1, 0].imshow(edges, cmap='gray')
    # axarr[1, 0].set_title('Edges')
    # axarr[1, 1].imshow(dilation, cmap='gray')
    # axarr[1, 1].set_title('Dilation')


    # # axarr[0, 0].plot(x, y)
    # # axarr[0, 0].set_title('Axis [0,0] Subtitle')
    # # axarr[0, 1].scatter(x, y)
    # # axarr[0, 1].set_title('Axis [0,1] Subtitle')
    # # axarr[1, 0].plot(x, y ** 2)
    # # axarr[1, 0].set_title('Axis [1,0] Subtitle')
    # # axarr[1, 1].scatter(x, y ** 2)
    # # axarr[1, 1].set_title('Axis [1,1] Subtitle')

    # # # Fine-tune figure; hide x ticks for top plots and y ticks for right plots
    # plt.setp([a.get_xticklabels() for a in axarr[0, :]], visible=False)
    # plt.setp([a.get_yticklabels() for a in axarr[:, 1]], visible=False)

    # # Tight layout often produces nice results
    # # but requires the title to be spaced accordingly
    # fig.tight_layout()
    # fig.subplots_adjust(top=0.88)

    # plt.show()


def viz(dir: str):

    onlyfiles = [f for f in listdir(dir) if isfile(join(dir, f)) and "max" in f]
    print(onlyfiles)

    for file in onlyfiles:
        file_name = file
        file_name = join(dir, file_name)

        out_filename = file_name.split(".")[0]
        out_filename = out_filename.replace("_hag_delaunay",'')
        out_filename = f"{out_filename}_height_filtered.laz"
        out_file = join(dir, out_filename)

        plot_res(file_name)
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Vizulize the hough line transformation.')
    parser.add_argument('folder', type=str, help='folder to Vizulize')

    args = parser.parse_args()
    dir = args.folder
    viz(dir)