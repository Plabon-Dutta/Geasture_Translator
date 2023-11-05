# Geasture_Translator
1. Abstract: 
Sign language is one of the oldest and most natural form of language for communication, hence we have produced a real time method using neural networks for finger spelling based American sign language. Automatic human gesture recognition from camera images is an interesting topic for developing vision. We propose a convolution neural network (CNN) method to recognize hand gestures of human actions from an image captured by camera. The purpose is to recognize hand gestures of human task activities from a camera image. The position of hand and orientation are applied to obtain the training and testing data for the CNN. The hand is first passed through a filter and after the filter is applied where the hand is passed through a classifier which predicts the class of the hand gestures. Then the calibrated images are used to train CNN.

2. Motivation:

While the world has progressed in online interaction through various tools, but the D&M (Dumb & Deaf) people still face a language barrier. So, they depend on vision-based communication for interaction. If there is a common interface that converts sign language to text the hand gestures can be easily understood by the other people. So, research has been made for a vision-based interface system where D&M people can enjoy communication without really knowing each other’s language. The aim is to develop a user-friendly sign to speech conversion software where the computer understands the human sign language. There are different sign languages all over the world, namely American Sign Language (ASL), French Sign Language, British Sign Language (BSL), Indian Sign language, Japanese Sign Language and work has been done on other languages all around the world.


3. Literature Survey: 

In recent years there has been tremendous research done on hand gesture recognition. With the help of literature survey done we realized the basic steps in hand gesture recognition are: 

• Data acquisition 
The different approaches to acquire data about the hand gesture can be done in the following ways:
-	Use of sensory devices It uses electronic devices to provide exact hand configuration and position. Different glove-based approaches can be used to extract information. But it is expensive and not user friendly.

-	Vision based approach in vision-based methods computer camera is the input device for observing the information of hands or fingers. The Vision Based methods require only a camera, thus realizing a natural interaction between humans and computers without the use of any extra voices. These systems tend to complement biological vision by describing seven artificial vision systems that are implemented in software and/or hardware. The main challenge of vision-based hand detection is to cope with the large variability of human hand’s appearance due to an enormous number of hand movements, to different skin-color possibilities as well as to the variations in viewpoints, scales, and speed of the camera capturing the scene. 

• Data preprocessing and Feature extraction
-	In [1] the approach for hand detection combines threshold-based color detection with background subtraction. We can use Adaboost face detector to differentiate between faces and hands as both involve similar skin color.

-	We can also extract necessary images which are to be trained by applying a filter called Gaussian blur. The filter can be easily applied using open computer vision also known as OpenCV and is described in [3].

-	For extracting necessary image which is to be trained we can use instrumented gloves as mentioned in [4]. This helps reduce computation time for preprocessing and can give us more concise and accurate data compared to applying filters on data received from video extraction.

-	We tried doing the hand segmentation of an image using color segmentation techniques but as mentioned in the research paper skin color and tone is highly dependent on the lighting conditions due to which output, we got for the segmentation we tried to do were no so great. Moreover, we have a huge number of symbols to be trained for our project many of which look similar to each other like the gesture for symbol ‘V’ and digit ‘2’, hence we decided that in order to produce better accuracies for our large number of symbols, rather than segmenting the hand out of a random background we keep background of hand a stable single color so that we don’t need to segment it on the basis of skin color. This would help us to get better results. 

• Gesture classification
-	In [1] Hidden Markov Models (HMM) is used for the classification of the gestures. This model deals with dynamic aspects of gestures. Gestures are extracted from a sequence of video images by tracking the skin-color blobs corresponding to the hand into a body– face space centered on the face of the user. The goal is to recognize two classes of gestures: deictic and symbolic. The image is filtered using a fast look–up indexing table. After filtering, skin color pixels are gathered into blobs. Blobs are statistical objects based on the location (x, y) and the colorimetry (Y, U, V) of the skin color pixels to determine homogeneous areas [2] Naive Bayes Classifier is used which is an effective and fast method for static hand gesture recognition. It is based on classifying the different gestures according to geometric based invariants which are obtained from image data after segmentation. Thus, unlike many other recognition methods, this method is not dependent on skin color. The gestures are extracted from each frame of the video, with a static background. The first step is to segment and label the objects of interest and to extract geometric invariants from them. Next step is the classification of gestures by using a K nearest neighbor algorithm aided with distance weighting algorithm (KNNDW) to provide suitable data for a locally weighted Na¨ ıve Bayes classifier.
-	According to paper on “Human Hand Gesture Recognition Using a Con volution Neural Network” by Hsien-I Lin, Ming-Hsiang Hsu, and Wei Kai Chen graduates of Institute of Automation Technology National Taipei University of Technology Taipei, Taiwan, they construct a skin model to extract the hand out of an image and then 9 apply binary threshold to the whole image. After obtaining the threshold image they calibrate it about the principal axis to center the image about it. They input this image to a convolutional neural network model to train and predict the outputs. They have trained their model over seven hand gestures and using their model they produce an accuracy of around 95% for those seven gestures.


4. Problem Statement:

American sign language is a predominant sign language Since the only disability D&M people have been communication related and they cannot use spoken languages hence the only way for them to communicate is through sign language. Communication is the process of exchange of thoughts and messages in many ways such as speech, signals, behavior, and visuals. Deaf and dumb(D&M) people make use of their hands to express different gestures to express their ideas with other people. Gestures are the nonverbally exchanged messages and these gestures are understood with vision. This nonverbal communication of deaf and dumb people is called sign language. In our project we focus on producing a model which can recognize Fingerspelling based hand gestures to form a complete word by combining each gesture. The gestures we aim to train are as given in the image below.


![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/fa31ad8c-2f39-4f08-880e-ee1f00a5b481)
Figure 1: Hand Signs


5. Objectives & Project output:

Objectives:

•	To develop a fully functional product for individuals with hearing impairments, enabling them to connect with the world more effectively.

•	Utilizing technologies such as OpenCV, Matplotlib, Keras, Deep Learning, Python, and Heroku Hosting, our goal is to construct a web-based project that can recognize and interpret American Sign Language through machine learning techniques.


•	We aim to establish our own database for the purpose of training and testing our CNN model.

•	Our objective is to narrow the technological divide between deaf individuals and the broader population by offering a web-based application.

Project Output: 

•	User Interface for Gesture Input: A user-friendly interface allowing users to input sign language gestures through images.

•	Sign Language Recognition using CNN: Implement a Convolutional Neural Network (CNN) model to recognize and interpret American Sign Language (ASL) gestures from the input images.


•	Text Output of Recognized Sign Language: Convert the recognized ASL gestures into text format for display on the screen. This text represents the corresponding words or phrases.

•	Text-to-Speech Conversion: Incorporate a text-to-speech module that converts the recognized text into audio format. This feature enables users who do not understand sign language to hear spoken words.


•	Database of ASL Signs for Training: Create and maintain a database of ASL signs that the CNN model uses for training and recognition. Include information about the dataset sources and details of the model architecture.


4. Effect on society:

This project, designed to bridge the communication gap between sign language users and those who do not understand sign language, can have a profound effect on society. It promotes inclusivity, enabling individuals with hearing impairments to communicate effectively with the wider community. This empowers them, potentially increasing employment opportunities and reducing isolation. Additionally, it can be a valuable tool in education, healthcare, and daily interactions. By preserving and promoting sign language, the project contributes to cultural preservation. Its implementation raises awareness and fosters understanding of the challenges faced by individuals with hearing impairments, inspiring empathy, and inclusive attitudes in society. Overall, it has the potential to create a more accessible and understanding society for all.

5. Requirement Analysis:

There are two types of requirements in our project.

Basic Requirement’s:

1. Hardware Requirement:
   
![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/4168ccb0-fa23-4799-a2fb-41eeac1d38ab)


3. Software Requirements:

Platform: 
1.	Operating System: Windows 8 and Above 2. 
2.	IDE: Pychamp 
3.	 Programming Language: Python 3.6 
4.	Python libraries: OpenCV, Numpy, Matplotlib, Keras, PIL



Technical Requirement’s:

Data Acquisition:
The different approaches to acquire data about the hand gesture can be done in the following ways:
It uses electromechanical devices to provide exact hand configuration and position. Different glove-based approaches can be used to extract information. But it is expensive and not user friendly.
In vision-based methods, the computer webcam is the input device for observing the information of hands and/or fingers. The Vision Based methods require only a camera, thus realizing a natural interaction between humans and computers without the use of any extra devices, thereby reducing costs. The main challenge of vision-based hand detection ranges from coping with the large variability of the human hand’s appearance due to an enormous number of hand movements, to different skin-color possibilities as well as to the variations in viewpoints, scales, and speed of the camera capturing the scene.

![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/58ee1e80-0dbc-4372-b843-1ab41975db55)

Data pre-processing and Feature extraction:
In this approach for hand detection, firstly we detect hand from image that is acquired by webcam and for detecting a hand we used media pipe library which is used for image processing. So, after finding the hand from image we get the region of interest (Roi) then we cropped that image and convert the image to gray image using OpenCV library after we applied the gaussian blur. The filter can be easily applied using the open computer vision library also known as OpenCV. Then we converted the gray image to binary image using threshold and Adaptive threshold methods. We have collected images of different signs of different angles for sign letter A to Z.

![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/2035d73d-4f90-4464-8e91-55e327d39531)
Figure 3: Gray Scaling

In this method there are many loopholes like your hand must be ahead of clean, soft background and that is in proper lightning condition then only this method will give good accurate results but in real world we do not get good background everywhere and we do not get good lightning conditions too.
So, to overcome this situation we try different approaches then we reached at one interesting solution in which firstly we detect hand from frame using mediapipe and get the hand landmarks of hand present in that image then we draw and connect those landmarks in simple white image.

![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/83d5264d-29b8-4d27-a3ef-a35b7b60ccd8)
![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/7a1a1b1f-49e6-4844-8c1c-f3a4fb8df040) ![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/13b011ae-7b54-4467-9a38-adc16abe3365)
![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/8365dc97-247a-4433-895a-6712ef2aefa9)
Figure 4: Hand Landmark

Now we get these landmark points and draw it in plain white background using OpenCV library. By doing this we tackle the situation of background and lightning conditions because the mediapipe library will give us landmark points in any background and mostly in any lightning conditions. We have collected 180 skeleton images of Alphabets from A to Z.


![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/8a3cb90a-8f02-4dbf-b223-ce8af683c52c) ![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/89511990-7c0a-487a-93c6-7d7f561febe5) ![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/12423afe-3a5f-4a5f-ba18-333c877a88fc)







