# camera_calibration_tool

camera_calibration_tool is used to easily calibrate and rectify camera using 
opencv-python. 

With only one line of command, you can get parameters of your 
camera or rectify your video. 

It is similar with calibration package of ROS.


# Content
* [camera_calibration_tool](#camera_calibration_tool)
* [Content](#content)
* [Usage](#usage)
    * [Calibration](#calibration)
    * [Rectify video](#rectify-video)
    * [Rectify camera](#rectify-camera)
* [Command line parameters](#command-line-parameters)

# Requirement

This code works well in the following environment:

Ubuntu: 16.04

Python: 3.5.2

OpenCV: 3.2.0

Package requirement:
- glob
- numpy
- xml

# Usage

This code can be used to calculate matrix and distortion coefficients 
of your camera, and rectify video/camera with these parameters.

## Calibration

1. Prepare more than 10 images of chessboard photoed by your camera 
in different position and direction. 
2. Be sure they are in the format of 'JPG' or 'jpg' or 'png'. 
(They are better to be in the same format, because I do not know 
if it will work or not.)
3. Put them in a folder named `chess` which should be in the same 
directory as `calibration.py`.
4. Run Terminal in current directory:
```
python3 calibration.py --image_size 1920x1080 --mode calibrate --corner 8x6 --square 20
```
**Replace the value according to your camera and chessboard.**

Your camera parameters will saved in a xml file named `camera_params.xml`.

## Rectify video

If you have already calibrated your camera and saved params in `camera_params.xml`, then
run:
```
python3 calibration.py --image_size 1920x1080 --mode rectify --video_path test.mp4
```
**Replace the value according to your video. Only `mp4` and `avi` are tested. 
Not sure about other format.**

**NOTE: image_size should be the same with chessboard images you used to calibrate.**

## Rectify camera

To rectify camera, run:
```
python3 calibration.py --image_size 1920x1080 --mode rectify --camera_id 0
```
**Replace the value according to your camera.**

**NOTE: image_size should be the same with chessboard images you used to calibrate.**

It will show the origin and rectified images of your camera.

# Command line parameters

All the parameters and their actions are listed here:

```
python3 calibration.py [option] [value]
```
option|type|help|example
------|----|----|------------|
--image_size|str|width and height of image|1920x1080
--mode|str|to calibrate or rectify| choose from calibrate/rectify
--square|int|length of chessboard square(Necessary when calibrating)|20
--corner|str|width*height of chessboard corner(Necessary when calibrating)|8x6
--video_path|str|path of video need to rectify|./test.mp4
--camera_id|int|usb camera id|0
