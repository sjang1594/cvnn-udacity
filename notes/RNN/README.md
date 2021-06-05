# **Recurrent Neural Network** 

---

RNNs are used to maintain a kind of memory by linking the output of one node to the input of the next. In the case of an LSTM, for each piece of data in a sequence (say, for a word in a given sentence), there is corresponding *hidden* state h~t~. This hidden state is a function of the pieces of data that an LSTM has seen over time; it contains some weights and, represents both the short term and long term memory components for the data that the LSM has already seen. 

So, for an LSTM that is looking at words in a sentece, **the hidden state of the LSTM will change based on each new word it sees. And, we can use the hidden state to predict the next word in a sequence** or help identify the type of word in a language model, and lots of other things! 

## Some useful blogs & videos

* [CS231n-RNN-Image Captioning, LSTM](https://www.youtube.com/watch?v=iX5V1WpxxkY&ab_channel=MachineLearner)

* [Exploring LSTMs](http://blog.echen.me/2017/05/30/exploring-lstms/)

* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)

  