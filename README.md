[**Day1**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6921713172592697344-xstF?utm_source=linkedin_share&utm_medium=member_desktop_web)

**💡 Bias/Variance**: 
- If a model is underfitting(such as logistic regression of non linear data), it has "high bias". If the model is overfitting, then it has "high variance".
- We can get to know whether it has high bias or variance or not by checking the error of training, dev and test, for example:
    
        1. High Variance: Training error (1%) and Dev Error (11%)
        2. High Bias: Training Error (15%) and Dev Error(14%)
        3. High Bias and High Variance: Training Error(15%) and Test Error (30%)

-  If your algorithm has a high bias:
    
        1. Try to make the Neural network bigger (size of hidden units, number of layers)
        2. Try a different model that is suitable for your data
        3. Try to run it longer
        4. Different/Advanced Optimization Algorithms
- If your algorithm has high variance:
  
        1. Add more data
        2. Try regularization
        3. Try a different model that is more suitable for the data
    
- Reference:
  - [**Deep Learning Specialization**](https://www.coursera.org/specializations/deep-learning)

[**Day2**]()

**💡 Activation Function:**

- Sigmoid function looks like an S-curve
- The output of a sigmoid function ranges between 0 and 1. Since, output values bound between 0 and 1, it normalizes the output of each neuron.
- Specially used for models where we have to predict the probability as an output. Since the probability of anything exists only between the range of 0 and 1, sigmoid is the perfect choice.
- Smooth gradient, preventing “jumps” in output values. The function is differentiable, and it gives clear predictions, i.e very close to 1 or 0.
- Tanh activation function is a hyperbolic tangent function. The curves of tanh function and sigmoid function are relatively similar.
- The output interval of tanh is [-1,1] , and the whole function is 0-centric, which is better than sigmoid.
- It turns out that the tanh activation usually works better than sigmoid activation function for hidden units because the mean of its output is closer to zero and so it centers the data better for next layer. 
- The major advantage of tanh is that the negative inputs will be mapped strongly negative and the zero in[puts will be mapped near zero in the tanh graph. 
- Sigmoid or Tanh Function disadvantage is that if the input is too small or too high, the slope will be near zero which will cause us gradient descent problem. 
