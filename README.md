# CS267 Final Project - Parallelizing Stereo Vision for Driverless Applications

Contributors: Apratim Banerjee, Terence Neo, Zeid Solh

April 2024

## 1 Introduction

Stereo vision (depth perception using two cameras) plays a crucial role in self-driving cars by building a 3D understanding of the surrounding environment. 
However, processing these images in real-time, essential to quick reactions, can be computationally expensive, presenting a critical bottleneck to self-driving car technology.
One possible application of parallel computing that we will work on for the course project is the calculation of Disparity Maps for Stereo Vision. Calculating the disparity map, which represents the diﬀerences in image location of the same 3D point from two cameras, is the core of stereo vision. The algorithm has a huge potential for parallelism, and the speed at which the algorithm runs will
be extremely useful in real-time perception tasks in autonomous systems, enabling the processing of
high-resolution images and videos.
Since parallel computing can be applied to algorithm steps that are not interdependent, such as analyzing diﬀerent pixels of an image, we can distribute this workload across multiple cores or processors. We will explore parallelization techniques like SIMD, OpenMP, OpenMP SIMD, CUDA, CUDA SIMD, parallelizing on the CPU and GPU, bench-marking the algorithms, and comparing the performance improvements. Naively, we expect that adopting parallelization techniques will speed up the performance proportionately to the number of cores or processors used.

## 2 Method

Computing depth from a pair of images in a stereo camera involves matching each pixel in the image from the left camera with the corresponding pixel in the right camera. The further apart the pixels are in terms of X-Y coordinates, the closer the object is to the camera. We used a block-matching algorithm by converting input images into blocks of arrays, computing the disparity map for a given block in parallel, and then converting the disparity map to the appropriate data type and scale. We collected data on our own for testing. We conducted multiple tests, ﬁrst on images of varying resolutions, and then videos of varying resolutions of the UC Berkeley Campus. We captured the images and videos on a smartphone with two cameras (iPhone 15), simultaneously capturing images from each camera as the left and right stereo images. To process the video, we used the OpenCV library. We split the videos into multiple frames, and then, similar to how we processed images, we resized and converted them to mean-adjusted grayscale images. Then, after processing the frames, we convert the disparity maps of each frame back to a video and store them in an output folder. Beginning with no parallelization, we explore parallelizing via multi-threading using the OpenMP library, using Single Instruction Multiple Data (SIMD) instructions, and CUDA.

### 2.1 Algorithm

The further apart the pixels are in terms of their coordinates, the closer the object is to the observer. Pixels with larger disparity values (closer to the camera) are shown in a lighter color in a disparity map. In our experiments, we used the block-matching algorithm to calculate our disparity map. The algorithm takes a two step process:

1. In the ﬁrst step, it considers a square block around each pixel in the left image, comparing this square block to various locations in the right image. A loss function is then calculated for each block position. If a perfect match is found, the loss function returns zero. However, due to diﬀerences in conditions (i.e., lighting), exact matches are rare. In such instances, the pixel’s position can be estimated by ﬁtting a parabola to the three nearest matching points.

2. In the second step, it calculates the distance between the best-match pixels giving the disparity score. We will use a standard loss function of the Sum of Absolute Diﬀerences for our purposes.

As we are capturing the image using two cameras on the same phone, we will assume that the pictures are stereo-rectiﬁed, meaning that the best match in both images will have the same Y-coordinate, and the X-coordinate of the best match in the right image will be greater than or equal to the X-coordinate of the same pixel in the left image. This assumption will signiﬁcantly reduce the search space and speed up calculations. The disparity map algorithm is expressed in pseudo-code below.

### 2\.2 Order of Complexity and Computational Intensity

We hypothesized that this algorithm will run with an order of complexity of:

O(Block Size × Scan Steps × Image Resolution)

Where the block size is the square block of pixels we used to calculate the loss, scan steps is the
number of steps we scanned over to search for pixel candidates.
Next, to calculate the computational intensity of the disparity map algorithm, we’ll analyze the
number of ﬂoating-point operations (FLOPs) and the amount of data moved.
Number of Floating Point Operations (FLOPs)
• The main computation occurs in computing the loss function (Equation [1),](#br2)[ ](#br2)where the sum of
absolute diﬀerences (SAD) is computed over a block. This operation involves computing the
absolute diﬀerence between corresponding pixels in the left and right images and summing
them up.

• Each pixel comparison requires one subtraction operation and one absolute value operation,
which can be considered as two FLOPs.

• The total number of FLOPs per block comparison is thus approximately
2 × block width × block height


### Amount of Data Moved

• Data movement occurs mainly when accessing pixel values from the left and right images

• Each pixel access involves reading one byte because the pixel values in the grayscaled image

are stored as unsigned chars

• The amount of data moved per block comparison is hence approximately

2 × block width × block height bytes

Therefore, the computational intensity can be calculated as:

Number of FLOPs

2 × block width × block height = 1 

Computational Intensity = Number of FLOPs/Number of bytes moved = 2 × block width × block height/2 x block width x block height = 1

This indicates that the algorithm is balanced between computation and data movement, with approximately equal amounts of computation and data movement involved in each block comparison.

## 3 Parallelization Techniques Used

From the above pseudo-code, we can see that several algorithm parts can potentially be parallelized,
as labelled Steps A, B, and C in the pseudo-code.

• Step A: Calculation of the disparity value for each pixel in the Left image is independent of

each other

• Step B: The disparity value between a pixel in Left Image, and each pixel in the Right Image

is also independent.

• Step C: The loss value of each pixel pair in the neighborhood can also be calculated indepen-

dently.

## Results
Please refer to the [PDF: CS267_Final_project.pdf](https://github.com/Apratim08/Stereo-vision-parallel-computing-/blob/main/Report%20and%20Results) for more details on the perfromance we achieved from the implementation. 
