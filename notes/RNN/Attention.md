# **Attention**

## **Introduction**

"One important property of human perception is that one does not ten to process a whole scene in its entirety at once. Instead humans focus attention selectively on parts of the visual space to acquire information when and where it is needed, and combine information from different fiations over time to build up an internal representation of the scene, guiding future eye movements and decision makings" - Recurrent Models of Visual Attention, 2014

Classice sequence to sequence models, without attention, have to look at the original sentence that you want to translate one time and then use that entire input to produce every single small outputted work. 

Attention, however, allows the model to look at this small relevant parts of the input as you generate the output over time. When attention was incorporated in sequence to sequence models, they became the state of the art in neural machine translation. 

---

## **Background**

### Sequence to Sequence Models (RNN).

**Application :**

Assume that you train it on dataset where the source is an English phrase [It's a beautiful day!] and the target is a French phrase [C'est une belle journee!] and you have a lot of these examples. Train it on a dataset of news articles and their summaries and you have a smmarization bot. Train it on a dataset of questions and their answers and you have a question-answering model. 

The RNNs are used along convolutional nets and image captioning test. 

**Definition **

A sequence to sequence model takes in an input that is a sequence of items, and then it produces another sequence of items as an output. In a machine translation application, the input sequence is a series of words in one language, and the output is the translation in antoher language. 

A sequence to sequence model usually consists of an encoder and a decoder and it works by the encoder first processing all of the inputs turning the inputs into a single representation, typically a single vector. This is called the context vector, and it contains whatever information the encoder was able to capture from the input sequence. This vector is then sent to the decoder which uses it to formulate an output sequence. In machine translation senario, the encoder and decoder are both recurrent neural networks, typically LSTM cells in practice. In this scenario, the context vector is a vector of numbers encoding the information that the encoder captured from the input sequence. In real-world scenarios, this vector can have a length of 256 or 512 or more. 

The limitation of this model is that the encoder is confiend to sending a single vector no matter how long or short the input sequence is. Choosing a reasonable size for this vector makes the model have problems with long input sequences. This limitation can be resolved with Attention Network

### **Encoding**

First, the encoder processes the input sequence just like the model without attention one word at a time producing a hidden state and using the hidden state and the next step. Next, the model passes a context vector to the decoder but unlike the context vector in the model without attention, this one is not just the final hidden state, it's all of the hidden states. (Basically giving all hidden state to decoder). This gives us the benefit of having the flexibility in the context size. So longer sequences can have long contexts vectors that better capture the information from the input sequence. One additional point that's important for the intuition of attention, is that each hidden state is sort of associated the most with the part of the input sequence that preceded how that word was generated. So, the first hidden state was outputted after processing the first word, so it caputres the essence of the first word the most.

The encoder is a recurrent neural network. When creating RNN, we have to declare the number of hidden units in the RNN cell. This applies whether we have a vanilla RNN or an LSTM or GRU cell. Before we start feeding our input sequnece words to the Encoder. They have to pass through an embedding process which translates each word into a vector, which is going to be embedded input. Then ready to feed that into the encoder. Then RNN will create hidden states for each word in each time step. Then this will be passed to decoder.

## **Decoding**

At every time step, an attention decoder pays attention to the appropriate part of the input sequence using the context vector. How does the attention decoder know which of the parts of the input sequence to focus on at each step ? That process is learned during the training phase, and it's not jsut stupidly going sequentially from the first and the second to the third. It can learn some sophisticated behavior. 

For example, let's assume that we want to build the machine translation from french to english. In decoding phase, the first step, the attention decoder would pay attention to first part of the sentence. 

An attention decoder has the ability to look at the inputted words, and the decoders own the hidden states, and then it would do following; it would use a scoring function to score each hidden state in the context matrix. After scoring each context vector would end up with a certain score and if we feed these scores into a softmax function, we end up with scores that are all positive, that are all between zero and one, and that all sum up to one. These values are how much each vector will be expressed in the attention vector that the decoder will look at before producing an output. Simply multipliying each vector by its softmax score and then, summing up these vector produces an attention contexts vector, this is a basic weighted sum operation. The decoder has now looked at the input word and at the attention context vector, which focused its attention on the appropriate place in the input sequence. So, it prodcues a hidden state and it produces the first word in the output sequence. In the next time-step, the RNN takes its previous output as an intput and it generates its own context vector for that time-step, as well as the hidden state from the previous time-step, and that produces new hidden state for the decoder, and a new word inthe ouptut sequence. 

## Encoders and Decorder

The encoder and decorder do not have to be RNNs; they can be CNNs too! An LSTM is used to generate a sequencce of words; LSTMs "remember" by keeping track of the input words that they see and their own hidden state

In computer vision, we can use this kind of encoder-decoder model to generate words or captions for an input image or even to generate an image from a sequence of input words. We'll focus on the first case: generating captions for images, and you'll learn more about caption generation in the next lesson. For now know we can input an image into a CNN(encoder) and generate a descriptive caption for that image using an LSTM(decoder)

## **Multiplicative Attention**

 An Attention scoring function tends to be a function that takes in the hidden state of the decoder and the set of hidden states of the encoder. This scoring function will do at each time-step on the decoder side. The simplest method is to calculate the dot product of hidden state of the deocder and the set of the hidden state of the encoder. The dot product of two vectors produces a single number. But the important thing is the significance of this number. Geometrically, the dot product of two vectors is equal to multiplying the lengths of the two vectors by the cosine of the angle between them. This means that if we have two vectors with the same length, the smaller the angle between them, the larger the dot product becomes. In practice, however, we want to speed up the calculation by scoring all the encoder hidden states at onece, which leads us to the formal mathemtatical definition of dot product attention. It is the hidden state of the current time-step transposed times the matrix of the encoder hiddden timesteps. 

With the simplicity of this method comes the drawback of assuming the encoder and decoder have the same embdding space.  So while this might work for text summarization, where the encoder and decoder use the same language and the same embedding space.  For machine translation, however, you might find that each language tends to have its own embedding space. This is a case where we might want to use the second scoring method. It simply introduces a weight matrix between the multiplication of the decoder hidden state and the encoder hidden states. This weight matrix is a linear transformation that allows the input and outputs to use different embeddings and the result of this multiplication would be the weight vector. 

## **Additive Attention**

Commonly used scoring method is called *concat*, and the way to do it is to use a feedforward neural network. The concat scoring method is commonly done by concatenating the two vectors and making that the input to a feed forward neural network, So we merge the two vectors; the hidden state of the decoder and the set of hidden states of the encoder, and concat them into one vector, and then we pass them through a neural network. The parameter of this network are learned during the training process. While in the loop paper it uses the one from the current time step at the decoder. 

---

## **Computer Vision Application**

### **Image Captioning** :

1. [Show, Attend and Tell: Neural Image Caption Generation with Visual Attention](https://arxiv.org/pdf/1502.03044.pdf)

2. [Bottom-Up and Top-Down Attention for Image Captioning and Visual Question Answering]()
3. [Video Paragraph Captioning Using Hierarchical Recurrent Neural Network](https://www.cv-foundation.org/openaccess/content_cvpr_2016/app/S19-04.pdf)
4. [Every Moment Counts: Dense Detailed Labeling of Actions in Complex Videos](https://arxiv.org/pdf/1507.05738.pdf)
5. [Tips and Tricks for Visual Question Answering: Learnings from the 2017 Challenge](https://arxiv.org/pdf/1708.02711.pdf)
6. [Visual Question Answering: A Survey of Methods and Datasets](https://arxiv.org/pdf/1607.05910.pdf)

*Comments:* The encoder in this case will be a convolutional neural network that has produces a set of feature vectors, each of which corresponds to a part of the image or a feature of the image. To be more exact, a VGG net convolutional network trained on the ImageNet. The annotations were created from this feature map [ 14 x 14 x 512 ]. To create the annotation vector, we need to flatten each feature turning it from 14 x 14 to 196 x 1. Then, you will create 196 x 512, and this is going to be the context vector to the decoder. 

The decoder is a recurrent neural network, which uses attention to focus on the appropriate annotation vector at each time step. We plug this into the attention process we've outlinted before and that's image captioning model. 

---

## **Other Attention Method**

This paper noted that the complexity of encoder-decoder with Attention models can be simplified by adopting a new type of model that only uses Attention, no RNNs. They called this new model the *Transformer*.  In two of their expperiments on machine translation tasks, the model proved superior in quality as well as requiring significantly less time to train. The transformer takes a sequence as an input and generate a sequence, just like the squence-sequence models. The difference here, however, is it does not take the inputs one by one as in the case of an RNN. It can produce all of them together in parallel. Perhaps each element is processed by a separate GPU if we want. It then produces the output one by one but also not using an RNN. The transformer model also breaks down into an encoder and a decoder. But instead of RNNs, they use feed-forward neural networks and a concept called self-attention. This combination allows the encoder and decoder to work without RNNs, which vastly improves performance since it allows parallelization of processing that was not possible with RNNs. The transformer contains a stack of identical encoders and decoders. Six is the number the paper proposes. 

**Encoder** 

Each encoder layer contains two sublayers: a multi-headed self-attention layer and a feed-forward layer. The attention component is completely on the encoder as side as opposed to being a decoder component like the previous attention mechanism. This attention component helps the encoder comprehend its input by focusing on other parts of the input sequence that are relevant to each input element it processes. 

For example, when it comes to solve the machien reading problem. The structure of the transformer allows the encoder to not only focus on previous words in the input sequence, but also on words that appeared later in the input sequence. 

**Decoder**

The decoder contains two attention components as well. One that allows it to focus on the relevant part of the inputs and another than only pays attention to previous decoder outputs. [Feed-forward, encoder-decoder attention, self-attention]

Paper : [Attention is All You Need](https://arxiv.org/abs/1706.03762)

Video : [Attention is All You Need](https://www.youtube.com/watch?v=rBCqOTEfxvg&ab_channel=PiSchool)

---

