## Backpropogation Theory

**KEY** : Partial Derivatives

In the **backpropagation** process we minimize the network error slightly with each iteration, by adjusting the weights; going backwards from the output to the input while changing the weights is called **backpropagation** which is essentially stochastic gradient descent computedd using the chain rule.

---

### Goal : goal is to find a set of weights that minimizes the network errror

- Iterative process presenting the network with one input at a time from our training set.
- During the feedforward pass, we calculate the networks error. (Change the error to slightly change the weights in the correct direction until the error is small enough)

