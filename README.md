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

![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/201dd2eb-d240-4c01-834e-c7fe1e1eed81)

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

Figure 5: Hand Landmark Processing

6. Methodology:
Project Developing Resource:

We have designed our software model using UML. UML (Unified Modeling Language) diagram is a visual representation of a system or software project, designed to help developers and stakeholders understand and communicate various aspects of the system. UML diagrams provide a standardized set of symbols and notations to describe various elements of a system and their relationships. The several types of UML diagrams are used to represent various aspects of the system, such as structure, behavior, interactions, and architecture. UML diagrams can be used throughout the software development process, from requirements gathering and analysis to design, implementation, and testing. They can help to identify potential issues, clarify requirements, and communicate the system's design to stakeholders. UML diagrams provide a useful tool for software development teams to document and communicate the design and functionality of a system in a standardized and visual way.


![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/633bb050-b883-4105-9e13-2705c3f65499)

DFD Diagram:


![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/95f92998-0b32-401f-9651-417fb566f040)

The system is a vision-based approach. All the signs are represented with bare hands and so it eliminates the problem of using any artificial devices for interaction.

Data Set Generation: 

For the project we tried to find already made datasets, but we could not find datasets in the form of raw images that matched our requirements. All we could find were the datasets in the form of RGB values. Hence, we decided to create our own data set. Steps we followed to create our data set are as follows. We have decided to use Open computer vision (OpenCV) library to produce our dataset. Firstly, we are going to capture around eight hundred images of each of the symbols in ASL for training purposes and around two hundred images per symbol for testing purposes. First, we capture each frame shown by the webcam of our machine. In each frame we define a region of interest (ROI) which is denoted by a blue bound square. From this whole image we extract our ROI which is RGB and convert it into gray scale Image. Finally, we apply our gaussian blur filter to our image which helps us extract various features of our image. The image after applying gaussian blur.

Gesture Classification: 

Our approach uses two layers of algorithm to predict the final symbol of the user. 

Algorithm Layer 1: 
•	Apply gaussian blur filter and threshold to the frame taken with OpenCV to get the processed image after feature extraction. 
•	This processed image is passed to the CNN model for prediction and if a letter is detected for more than fifty frames then the letter is printed and taken into consideration for forming the word. 
•	Space between the words is considered using the blank symbol. 

Algorithm Layer 2: 
•	We detect various sets of symbols which show related results on getting detected.
•	We then classify between those sets using classifiers made for those sets only.

Layer 1: CNN Model
-	1st Convolution Layer: The input picture has resolution of 12Sx128 pixels. It is first processed in the first convolutional layer using thirty-two filter weights (3x3 pixels each). This will result in a 126x126 pixel image, one for each Filter-weight.

-	1st Pooling Layer: The pictures are downsampled using max pooling of 2x2 i.e., we keep the highest value in the 2x2 square of array- Therefore, our picture is downsampled to 63x63 Pixels.


-	2nd Convolution Layer: Now, these 63 x 63 from the output of the first pooling layer is served as an input to the second convolutional layer. It is processed in the second convolutional layer using thirty-two filter weights (3x3 pixels each). This will result in a 60 x 60-pixel image.

-	2nd Pooling Layer: The resulting Images are downsampled again using max pool of 2x2 and is reduced to 30 x 30 resolution of Images.


-	1st Densely Connected Layer: Now these images are used as an Input to a fully connected layer with 128 neurons and the output from the second convolutional layer is reshaped to an array of 30x30x32 =2SS00 values. The input to this layer is an array of 28800 values. The output of these layers is fed to the 2nd Densely Connected Layer-We are using a dropout layer of value 0.5 to avoid overfitting.

-	2nd Densely Connected Layer: Now the output from the 1st Densely Connected Layer is used as an input to a fully connected layer with ninety-six neurons.


-	Final layer: The output of the 2nd Densely Connected Layer serves as an input for the final layer which will have the number of neurons as the number of classes we are classifying (alphabets + blank symbol).

Activation Function: We have used ReLu (Rectified Linear Unit) in each of the layers (convolutional as well as fully connected neurons). ReLu calculates max(x,0) for each input pixel. This adds nonlinearity to the formula and helps to learn more complicated features. It helps in removing the vanishing gradient problem and speeding up the training by reducing the computation time.

Pooling Layer: We apply Max pooling to the input image with a pool size of (2, 2) with relu activation function. This reduces the number of parameters thus lessening the computation cost and reduces overfitting.

Dropout Layers: The problem of overfitting, where after training, the weights of the network
are so tuned to the training examples they are given that the network does not perform well when given new examples. This layer "drops out" a random set of activations in that layer by setting them to zero. The network should be able to provide the right classification or output for a specific example even if some of the activations are dropped out [5].

Optimizer: We have used Adam optimizer for updating the model in response to the
output of the loss function. Adam combines the advantages of two extensions of two stochastic gradient descent algorithms namely adaptive gradient algorithm (ADA GRAD) and root mean square propagation (RMSProp).


Layer 2:

We are using two layers of algorithms to verify and predict symbols which are more like each other so that we can get as close as we can get to detect the symbol shown. In our testing we found that following symbols were not showing properly and were giving other symbols also:
1. ForD: RandU
2. ForU: DandR
3. Forl: T, D, Kandl
4. Fors: M and N

So, to handle above cases we made three different classifiers for classifying these sets:
1. {D, R, U} 
2. {T, K, D, I} 
3. {S, M, N}



Finger spelling sentence formation Implementation:

•	Whenever the count of a letter detected exceeds a specific value and no other letter is close to it by a threshold, we print the letter and add it to the current string (ln our code we kept the value as fifty and difference threshold as 20).

•	Otherwise, we clear the current dictionary which has the count of detections of present symbol to avoid the probability of a wrong letter getting predicted.


•	Whenever the count of a blank (plain background) detected exceeds a specific value and if the current buffer is empty no spaces are detected.

•	In other case it predicts the end of word by printing a space and the current gets appended to the sentence below.

Autocorrect Feature:
A python library Hunspell_suggest is used to suggest connecting alternatives for each (incorrect) input word and we display a set of words matching the current word in which the user can select a word to append it to the current sentence. This helps in reducing mistakes committed in spelling and assists in predicting complex words.

Training and Testing:

We convert our input images (RGB) into grayscale and apply gaussian blur to remove unnecessary noise. We apply adaptive threshold to extract our hand from the background and resize our images to 128 x 128. 
We feed the input images after preprocessing to our model for training and testing after applying all the operations mentioned above. The prediction layer estimates how likely the image will fall under one of the classes. So, the output is normalized between 0 and 1 and such that the sum of each value in each class sums to one. We have achieved this using softmax function. 
First the output of the prediction layer will be far from the actual value. To make it better we have trained the networks using labeled data. Cross-entropy is a performance measurement used in classification. It is a continuous function which is positive at values which are not same as labeled value and is zero exactly when it is equal to the labeled value. Therefore, we optimized the cross-entropy by minimizing it as close to zero. To do this in our network layer we adjust the weights of our neural networks. TensorFlow has an inbuilt function to calculate the cross entropy. 
As we have found out the cross-entropy function, we have optimized it using Gradient Descent in fact with the best gradient descent optimizer is called Adam Optimizer
Software Process Model:

We have used Agile Methodology for software process models. Agile process model is the best choice for blockchain-based crowdfunding platform. As it focuses on iterative development, collaboration, and flexibility. This methodology is well-suited for blockchain projects as it allows for frequent feedback and adjustments, which is necessary for developing a complex and constantly evolving technology like blockchain.
We particularly used Scrum, which is an agile framework for managing and completing complex projects, particularly in software development. The framework is based on an iterative and incremental approach to project management, where the project is divided into smaller and more manageable parts, called sprints. The Scrum framework emphasizes flexibility, continuous improvement, and collaboration. By breaking down the project into smaller parts and regularly reviewing progress and feedback, the Scrum framework can help teams adapt to changing requirements and deliver high-quality products in a timely manner.

7. Challenges Faced:

There were many challenges faced by us during the project. The very first issue we faced was of dataset. We wanted to deal with raw images and those too square images like CNN in Keras as it was a lot more convenient working with only square images. We could not find any existing dataset for that hence we decided to make our own dataset. The second issue was to select a filter which we could apply on our images so that proper features of the images could be obtained and hence then we could provide that image as input for CNN model. We tried various filters including binary threshold, canny edge detection, gaussian blur etc. but finally we settled with gaussian blur filter. More issues were faced relating to the accuracy of the model we trained in earlier phases, which we eventually improved by increasing the input image size and by improving the dataset.


8. Result of Project:

We have achieved an accuracy of 95.8% in our model using only layer one of our algorithms and using the combination of layer one and layer two we achieve an accuracy of 98.0%, which is a better accuracy then most of the current research papers on American sign language. Most of the research papers focus on using devices like Kinect for hand detection. In [7] they build a recognition system for Flemish sign language using convolutional neural networks and Kinect and achieve an error rate of 2.5%. In [8] a recognition model is built using hidden Markov model classifier and a vocabulary of thirty words and they achieve an error rate of 10.90%. In [9] they achieve an average accuracy of 86% for forty-one static gestures in Japanese sign language. Using depth sensors map [10] achieved an accuracy of 99.99% for observed signers and 83.58% and 85.49% for new signers. They also used CNN for their recognition system. One thing that should be noted is that our model does not.
uses any background subtraction algorithm while some of the models present above do that. So, once we try to implement background subtraction in our project the accuracies may vary. On the other hand, most of the above projects use Kinect devices but our main aim was to create a project which can be used with readily available resources. A sensor like Kinect not only is not readily available but also is expensive for most of the audience to buy and our model uses a normal webcam of the laptop hence it is a great plus point.


![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/2e97975e-a26e-438d-954c-0714ca91c864)

Figure 6: Final Project Output

Project Output:

Signup Page: User Can Sign Up with his/her Details.

![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/99d568e5-d8a7-4973-84c0-5a195d1ce80f)

Login Page: User Can log in with his/her Credentials.

![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/eabbb2c3-2660-4fe8-ad87-996c4ade6e29)

Reset Password: If the user forgot his/her Password then he/she can reset the password.

![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/bda865c3-ca03-41f8-8385-a3f7162d7bf5)

Home Page: After Login User will be redirected to Home Page

![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/84908d45-1c02-44dd-b5f8-d32a2c827741)

Translate Page: If the user clicks the Translate Button, the User will be redirected to This Page. Where the User will show the sign of the Letter and That will be detected and added in the sentence. Also, Suggestions will be generated. Lastly, we can turn the text into voice by Clicking the Speak Button.

![image](https://github.com/Plabon-Dutta/Geasture_Translator/assets/79752960/efbda60a-c973-47af-bd5b-da444ba3314a)

10. Conclusion & Future work: 

Conclusion: 

In this report, a functional real-time vision-based American sign language recognition for D&M people has been developed for ASL alphabets. We achieved a final accuracy of 98.0% on our dataset. We can improve our prediction after implementing two layers of algorithms in which we verify and predict symbols that are more like each other. This way we can detect all the symbols if they are shown properly, there is no noise in the background and the lighting is adequate.


Future Work:

We are planning to achieve higher accuracy even in the case of complex backgrounds by trying out various background subtraction algorithms. We are also thinking of Improving the preprocessing to predict gestures in low-light conditions with a higher accuracy.


11. Reference:

[1] T. Yang, Y. Xu, and “A., Hidden Markov Model for Gesture Recognition”, CMU-RI-TR-94 10, Robotics Institute, Carnegie Mellon Univ., Pittsburgh, PA, May 1994. 

[2] Pujan Ziaie, Thomas M ̈uller, Mary Ellen Foster, and Alois Knoll “A Na ̈ıve Bayes Munich, Dept. of Informatics VI, Robotics and Embedded Systems, Boltzmannstr. 3, DE-85748 Garching, Germany. 

[3] https://docs.opencv.org/2.4/doc/tutorials/imgproc/gausian_median_blur_bilateral_filter/gausian_median_blur_bilateral_filter.html 

[4] Mohammed Waleed Kalous, Machine recognition of Auslan signs using PowerGloves: Towards large-lexicon recognition of sign language. 

[5] aeshpande3.github.io/A-Beginner%27s-Guide-To-Understanding-Convolutional-Neural-Networks-Part-2/

[6] http://www-i6.informatik.rwth-aachen.de/~dreuw/database.php 

[7] Pigou L., Dieleman S., Kindermans PJ., Schrauwen B. (2015) Sign Language Recognition Using Convolutional Neural Networks. In: Agapito L., Bronstein M., Rother C. (eds) Computer Vision- ECCV 2014 Workshops. ECCV 2014. Lecture Notes in Computer Science, vol 8925. Springer, Cham 

[8] Zaki, M.M., Shaheen, S.I.: Sign language recognition using a combination of new vision-based features. Pattern Recognition Letters 32(4), 572–577 (2011) 25 

[9] N. Mukai, N. Harada and Y. Chang, "Japanese Fingerspelling Recognition Based on Classification Tree and Machine Learning," 2017 Nicograph International (NicoInt), Kyoto, Japan, 2017, pp. 19-24. doi:10.1109/NICOInt.2017.9


