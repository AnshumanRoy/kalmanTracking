# Vehicle Tracking using Kalman Filter

This project aims to demonstrate an application of the **Kalman Filter** in tracking vehicles in a video capture. It could find potential use across various diverse fields like **Autonomous Vehicles**, **Surveillance and Security**, **Traffic Analysis** and **Logistics and Fleet Management**.

## Sample Output

![image](https://github.com/AnshumanRoy/kalmanTracking/assets/56593553/65d8a304-9ab2-4d61-9c8e-ac1d9ea14537)
- 🔴 Detected position from YOLO model
- 🟢 Estimated position from Kalman Filter
- 🔵 Predicted position from Kalman Filter


Still from the output video. To check the complete output video, view the **output.mp4** file included in the repository.

## Features
- Object detection using YOLO (You Only Look Once)
- Kalman Filter for position estimation and prediction
- Visualization of actual, estimated, and predicted vehicle positions

## Kalman Filter

The Kalman Filter is an efficient recursive algorithm that estimates the state of a linear dynamic system from a series of noisy measurements. In this project, the Kalman Filter is used to predict and update the positions of detected vehicles. The filter takes into account the process model (motion of the vehicle) and the measurement model (noisy measurements from object detection) to provide accurate estimates even in the presence of noise.

For a complete theoretical explanantion, this [website](https://www.kalmanfilter.net/background.html) is an excellent resource.

## Installation

1. Navigate to the desired location for the project and clone this repository to your local machine:

   ```.sh
   git clone https://github.com/AnshumanRoy/kalmanTracking.git
   cd kalmanTracking
   
3. Download the YOLO model files for the **weights**, the corresponding **config** files and the **coco class** names. These can be downloaded from the **official yolo repository**.

4. Install the required Python packages from the **requirements.txt** file.

   ```.sh
   pip install -r requirements.txt
   
5. Create a config.ini file with appropriate file paths and Kalman Filter parameters. Feel free to change the parameters used for the Kalman Filter to experiment with the model.

   ```.ini
   [Paths]
   yolo_weights = weights-file-path
   yolo_cfg = config-file-path
   coco_names = names-file-path
   video_input = video-file-path

   [KalmanFilter]
   dt = 1
   ux = 0
   uy = 0
   std_acc = 0.1
   x_measuredSD = 10
   y_measuredSD = 10

## Usage
    
  Run the program to start tracking the vehicle in the input video. Press the **'q'** key to exit the application.

   ```.sh
   python main.py  
