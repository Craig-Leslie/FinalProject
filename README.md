This is my code repository for my final year project *Music Control using Hand Gesture Recognition*. This project was implemented in 3 stages, which are expanded on below.

# Section 1 - Capturing Live Data
To capture hand gestures from the user the depth sensor from a Microsoft Kinect V2 was used. Depth sensors offer easier hand segmentation, and no interference from skin coloured objects, which provides a more accurate hand than RGB capture. A function is ran everytime the Kinect has a new frame, which takes the raw depth capture, performs segmentation using depth thresholds, assesses the liklihood that the segmented object is a hand, and crops the image to fit. This processed image is passed to the next section for prediction.

# Section 2 - Predicting Hand Gestures
To predict hand gestures a CNN was trained on a pre-existing database of [hand gesture depth images](https://zhou-ren.github.io/publications.html), which were all preprocessed using the same method as the live data, in order to ensure similar outputs. Once trained to a satisfactory accuracy, the model can be saved and loaded at runtime. When predicting live gestures a confidence metric is used to ensure false positives are not captured. The same gesture must be recognised 4 times in a row before it’s passed on to the music control system. 

# Section 3 - Controlling Music
To control music, an interface was developed using TKinter. This interface provides controls to the user to upload audio, and start the gesture recognition system, and to provide visual feedback, such as the live RGB feed, the different hand gestures available to the user, what hand gesture is currently recognised, etc. The system operates in 2 modes based on what kind of song is chosen, as is explained below: 
  • While in single mode, hand gestures will alter properties of the selected song, such as 
  increasing/decreasing the volume, pausing, playing and restarting the song. 
  • While in stem mode, predicted hand gestures will play different parts of the song, one hand 
  gesture may play the isolated vocals, while another may play the drum track.
