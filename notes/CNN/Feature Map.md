## How can we see what kind of features and networks have learned to recognize ?

- This area of study is called feature visualization and it’s all about techniques that allow you to see at each layer of the model. 
- What kinds of image features and networks have learned to extract ?

---

### **Feature Map**

The feature map will consist of grids of weights, For a convolutional layer with four filters, four filtered output will be produced. Each of these filtered output images is also called a feature map or activation map because each filtered image should extract certain features from an original image and ignore other information. For a given image, each of these maps will activate in some way, displaying activated bright pixels or not in each map. 

The first convolutionaly layer in a CNN applies a set of image filters to an input image and outputs a stack of feature maps. After such a network is trained, we can get a sense of what kinds of features this first layer has learned to extract by visualizing the learned weights for each of these filters. 

---

### First Convolutional Layer

For example, AlexNetFet. The first convolutional layer of a CNN with 11 by 11 color filters, they are looking for specific kinds of geometric patterns (vertical or horizontal edges), which are represented by alteranting bars of light and dark at different angles. You can also see corner detectors and filters that appear to detect the areas of opposing colors with magent and green bunches or blue and orange lines. 

This becomes tricky as soon as you move further into the network and try to look at the second or even thrid convolutional layer. This is because these later layers are no longer directly connected to the input image. They're connected to some output that has likely been pooled and precessed by activation function. 

---

### Visualizing Activation

For intermediate layers, like the second convolutional layer in a CNN visualizing the learned weights in each filter doesn't give us easy to read information. 

**Layer Activation** : means looking at how a certain layer of feature maps activates when it sees a specific input image such as an image of a face. 