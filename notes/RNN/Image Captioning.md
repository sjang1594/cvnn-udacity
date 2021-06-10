# CNN-RNN models - Image Captioning. 

We want our captioning model to take in  an image as input and output a text description of that image. The input image will be processed by CNN and will connect the output of the CNN to the input of the RNN which will allow us to generate descriptive texts. 

---

**Example** : 

You have the training image of a man holding a slice of pizza. 

**Our goal** is to train a network to generate this caption given the image as input. 

**Process** : 

First, we feed this image into a CNN using a pre-trained network like VGG16 or ResNet. At the end of these neetworks, is a softmax classifier that outputs a vector of class scores. But we don't want to classify the image. Instead, we want a set of features that represents the spatial content in the image. To get that kind of spatial Information, we have to remove the final fully connected layer that classifies the image and look at it ealier layer that distills the spatial information in the image. Now the CNN is used as a feature extractor that compresses the huge amount of information contained in the original image into a smaller representation. This CNN model often called the encoder because it encodes the content of the image into a smaller feature vector.  Then this feature vector will be processed and used as an initial input to the RNN. 

---

## **The Glue, Feature Vector**

The feature vector that's extracted from the CNN will go through some processing steps to be used as input to the first cell in the RNN. It can sometimes prove useful to parse a CNN output through an additional fully-connected or lineary layer before using it as an input to the RNN. This is similar to what we've seen in other transfer learning example. The CNN we're using is pertained network and adding an untrained linearly or at the end of it allow us to tweak only this final layer as we train the entire model to generate captions. 

---

## **Tokenizing Captions**

The RNN component of the captioning network is trained on the captions in the COCO dataset. We're aiming to predict the next word of a sentence based on previous words. *But, how exactly can it train on string data?* *Neural networks do not do well with strings.* They need a well-defined numerical alpha to effectively perform back propogation and learn to produce similar output. So we have to transform the caption associated with the image into a list of tokenize words. This tokenization turns any strings into a list of integers. 

*How dose the tokenization works?*

1. Iterate through all of the training captions and create a dictionary that maps all unique words to a numerical index. So, every word we come across will have a corresponding integer value that we can find in this dictionaly. The world in this dictionaries are referred to as our vocabulary. The vocabulary typically also includes a few special tokens. 
2. Embedding layer : right before it sent to RNN, which transform each word in a caption into a vector of a vector of a desired consistent shape. 
3. After this embedding state, we're finally ready to train an RNN that can predict the most likely next word in a sentence.

---

## **Tokenizing Words**

### **Words to Vectors**

At this point, we know that you cannot directly feed words into an LSTM and expect it to be able to train or produce the correct output. These words first must be turned into a numerical representation so that a network can use normal loss functions and optimizers to calculate how "close" a predicted word and ground truth word(from a known, training caption) are. So we, typically turn a sequence of words into a sequence of numerical values; a vector of numbers where each number maps to a specific word in our vocabulary.

To process words and create a vocabulary, we'll be using the Python text processing toolkit: NLTK. 

---

## **RNN Training**

**Taking a look at how the decoder trains on a given caption**

The decoder will be made of LSTM cells, which are good at remembering lengthy sequence of words. Each LSTM cell is expecting to see the same shape of the input vector at each time-step. The very first cell is connected to the output feature vecetor of the CNN encoder. An embedding layer that transformed each input word into a vector of a certain shape before being fed as input to the RNN. We need to apply the same transformation to the ouptut feature vector of the CNN. Once this feature vector is embedded into expected input shape. We can begin that RNN training process with this as the first input. The input to the RNN for all future time steps will be the individual words of the training caption.  

**RNN has two responsibilities**:

1. Remember spatial Information from the input feature vector. 
2. Predict the next word.

We know that the very first word it produces should always be the start token, and the next word should be those in the training caption. For our caption, "a man holding a slice of pizza", we know after the start token comes a and after a, comes man, and so on. At every time step, we look at the current caption word as input, and combine it with the hidden state of the LSTM cell to produce an output. This output is then passed through a fully connected layer that produces a distribution that represents the most likely next word. It produces a list of next words scores intead of a list of calss scores. This is like how we've seen softmax apply to classification task. But in this case, it produces a list of next words scores instead of a list of class scores. We feed the next word in the caption to the network, and so on, until we reach the end token. The hidden state of an LSTM is a function of the input token to the LSTM and the previous state. This function is refered as the recurrence function. The recurrence function is defined by weight and during the training process this model uses back propagation to update these weights until the LSTM cells learn to produce the correct next word in the caption given the current input word. 

### **Training vs Testing**

During training, we have a true caption which is fixed, but during testing the caption is being actively generated(starting with `<start>`), and at each step you are getting the most likely next word and using that as input to the next LSTM cell.

### **Caption Generation, Test Data**

After the CNN sees a new, test image, the decoder should first produce the `<start>` toekn, then an output distribution at each time step that indicates the most likely next word in the sentence. We can sample the output distribution (namedy, extract the one word in the distribution with the highest probability of being the next word) to get the next word in the caption and keep this process going until we get to another special token: `<end>`, which indicates that we are done generating a sentence. 

---

## **Video Captioning**

This captioning network can also be applied to video, the one thing that has to change about this network architecture is the feature extraction step that occurs between the CNN and the RNN. The input to the pre-trained CNN will be a short video clip which can be thought of as a series of image frames. For each image frame, the CNN will produce a feature vector. But our RNN cannot accept multiple feature vectors as input. So we have to merge all of these fature vectors into one that is representative of all image frames. One way of doing that is to take an average over all of the feature vectors created by the set of image frames. This produces a single average feature vector that represents the entire video clip. Then we can proceed as usual, training the RNN, and then using the single vector as the initial input, and training on a set of tokenized video captions. 