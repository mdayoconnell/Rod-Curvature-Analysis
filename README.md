# Rod-Curvature-Analysis
Code used for analyzing 1D rod curvature in contemporary experimental physics project, Spring 2025
Brief description of each file is included below. Note that much of this code requires OpenCV, a free image analysis software which you can learn about here https://opencv.org/releases/

**Main files for curvature analysis below:**

Video_Frame_Analysis_V4.py
  Main script where video file path, fps, and rod length in centimeters can be input. Returns plots and includes option for manually fitting linear segments to a graph.
  
Make_Skeleton.py
  Contains the function extract_skeleton(photo_file_path) which computes the 1-pixel wide curve from our photo. Crucially, this was used for videos where the rod was framed    against a black background. This allowed us to identify the rod by setting a threshhold for finding the brighest x% of pixels. For a slower but more robust method, see     
  BlackOut_Frames.py. This main() function is a debugger which helps us to set correct detection threshholds.

**Optional files - may be helpful for debugging or related projects**

Test_Video_Is_Loading.py
  Makes sure that cv2 is working correctly, loading the video, and able to export those frames to a folder.

Broken_Spaghetti.py
  Not used in our project, but a possible extension. Takes a black-on-white skeleton of the frame **after** breakage and estimates the length of each segment.

BlackOut_Frames.py
  Skeleton fitting for an earlier image analysis method where we manually blacked out areas around a rod to aid image detection. Script was used for a few max/initial     
  curvature calculations - would have been impractical for analyzing videos with ~1000 frames. skimage.morphology() section here is more robust than the version in   
  Make_Skeleton.py, but the black out process is imprecise and slow, and was only used for analyzing a few bend-release cases.

Test_Inversion_And_Skeleton_Curve.py
  Defunct script for making sure that skeleton curve is being fitted propertly. Ultimately was superseded by Make_Skeleton.py, but contains many of the same tools for image    analysis.

**Brief Outline for getting OpenCV if using Anaconda**

1. Create a new anaconda environment called "curvature-environment" or something similar. It can be tricky to get OpenCV on the base environment because of the dependencies that must be resolved
2. Activate your special curvauture analysis environment in the terminal, and run:
conda install -c conda-forge opencv
3. Start Spyder or your IDE and make sure you run it from the base environment!
4. In your IDE preferences or settings tab, set the "python interpreter" to the new environment you created
5. You should be all set to use OpenCV for this project!
