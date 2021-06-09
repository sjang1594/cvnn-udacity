# **Recurrent Neural Network** 

RNNs are used to maintain a kind of memory by linking the output of one node to the input of the next. In the case of an LSTM, for each piece of data in a sequence (say, for a word in a given sentence), there is corresponding *hidden* state h<sub>t</sub>. This hidden state is a function of the pieces of data that an LSTM has seen over time; it contains some weights and, represents both the short term and long term memory components for the data that the LSM has already seen. 

So, for an LSTM that is looking at words in a sentece, **the hidden state of the LSTM will change based on each new word it sees. And, we can use the hidden state to predict the next word in a sequence** or help identify the type of word in a language model, and lots of other things! 

---

## LSTMs in Pytorch

To create and train an LSTM, you have to know how to structure the inputs, and hidden state of an LSTM. In PyTorch an LSTM can be defined as : `lstm = nn.LSTM(input_size=input_dim, hidden_size=hiddn_dim, num_layers=n_layers)`

In PyTorch, an LSTM expects all of its inputs to be a 3D tensors, with dimensions defined as follows:

* `input_dim` = the number of inputs (a dimensions of 20 could represent 20 inputs)
* `hidden_dim` = the size of the hidden state; this will be the number of outputs that each LSTM cell produces at each time step.
* `n_layers` = the number of hidden LSTM layers to use; this is typically a value between 1 and 3; a value of 1 means that each LSTM cell has one hidden state. This has a default value of 1.

### **Hidden State**

Once an LSTM has been defined with input and hidden dimensions, we can call it and retrieve the output and hidden state at every time step. `out, hidden = list(input.view(1, 1, -1), (h0, c0))`

The inputs to an LSTM are `(input, (h0, c0))`.

* `input` = a Tensor containing the values in an input sequence; this has value; (seq_len, batch, input_size)
* `h0`= a Tensor containing the initial hidden state for each element in a batch
* `c0` = a Tensor containing the initial cell memory for each elements in a batch

`h0` and `c0` will default to 0, if they are not sepcified. Their dimensions are : (n_layers, batch, hidden_dim). 

Pytorch Tutorial on LSTMs : [LSTMs in PyTorch](https://pytorch.org/tutorials/beginner/nlp/sequence_models_tutorial.html#lstm-s-in-pytorch)

* Example : we want to process a single sentence through an LSTM. If we want to run the sequence model over one sentence "Giraffes in a field", our input should look like this `1x4` row vector of individual words.  [ Giraffes in a field ]
* In this case, we know that we have **4 inputs words** and we decide how many outputs to generate at each time step, say we want each LSTM cell to generate **3 hidden state values**. We'll keep the number of layers in our LSTM at the default size of 1.
* The hidden state and cell memory will have dimensions (n_layers, batch, hidden_dim), and in this case that will be (1, 1, 3) for a 1 layer model with one batch/sequence of words to process (this one sentence) and 3 generated, hidden state values. 

---

### **Learn Gate**

It takes a short term memory and the event and it joins it.  It takes the short term memory and the event and it combines them and then it ignores a bit of it keeping the important part of it.  

The output of the *Learn Gate* is N<sub>t</sub>i<sub>t</sub>

N<sub>t</sub> = tanh(W<sub>n</sub>[STM<sub>t-1</sub>, E<sub>t</sub>] + b<sub>n</sub>)

i<sub>t</sub> = sigmoid(W<sub>i</sub>[STM<sub>t-1</sub>, E<sub>t</sub>] + b<sub>i</sub>)

**How to ignore part of it**: Multiplying the factor i<sub>t</sub>. (it multiply element-wise). **How to calculate the i<sub>t</sub> **: We use our previous information of the short term memory and the event. 

## **Forgot Gate**

It takes a long term memory and it decides what parts to keep and to forget.

LTM<sub>t-1</sub> is multiplied by the **forget factor**. which is deonted f<sub>t</sub>. 

**How does the forget factor f<sub>t</sub> is calculated **: 

f<sub>t</sub> = sigmoid(W<sub>f</sub>[STM<sub>t-1</sub>, E<sub>t</sub>] + b<sub>f</sub>)

## **Remember Gate**

It takes the long-term memory coming out of the Forget Gate and the short-term memory coming out of the Learn Gate and simply combines them together. 

LTM<sub>t</sub> = LTM<sub>t-1</sub>*f<sub>t</sub> + N<sub>t</sub>*i<sub>t</sub>

## **Use Gate**

This is the one that uses the long term memory that just came out of the foret gate and the short term memory that just came out of the learned gate to come up with a new long term memory and an output. 

U<sub>t</sub>(Output of Forget Gate) = tanh(W<sub>u</sub>LTM<sub>t-1</sub>f<sub>t</sub> + b<sub>u</sub>)

V<sub>t</sub>(STM<sub>t-1</sub> and E<sub>t</sub>) = sigmoid([STM<sub>t-1</sub>, E<sub>t</sub>] + b<sub>n</sub>)

STM<sub>t</sub> = U<sub>t</sub>*V<sub>t</sub>

## Some useful blogs & videos

* [CS231n-RNN-Image Captioning, LSTM](https://www.youtube.com/watch?v=iX5V1WpxxkY&ab_channel=MachineLearner)
* [Exploring LSTMs](http://blog.echen.me/2017/05/30/exploring-lstms/)
* [Understanding LSTM Networks](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)
* [LSTMs Tutorial](https://web.archive.org/web/20190106151528/https://skymind.ai/wiki/lstm)

