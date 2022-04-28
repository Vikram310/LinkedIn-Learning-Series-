[**Day 1**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6921713172592697344-xstF?utm_source=linkedin_share&utm_medium=member_desktop_web)

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

[**Day 2**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6922068127300149248-0q96?utm_source=linkedin_share&utm_medium=member_desktop_web)

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

[**Day 3**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6922417875035049984-99VU?utm_source=linkedin_share&utm_medium=member_desktop_web)

**💡 Activation Function:**

- The ReLU (Rectified Linear Unit) function is an activation function that is currently more popular compared to other activation functions in deep learning.
- When the input is positive, there is no gradient saturation problem as in Sigmoid and Tanh
- The calculation speed is much faster. The ReLU function has only a linear relationship. Whether it is forward or backward, it is much faster than sigmoid and tanh.
- The only problem with ReLU is Dead ReLU problem. When the input is negative, ReLU is completely inactive, which means that once a negative number is entered, ReLU will die. In this way, in the forward propagation process, it is not a problem. Some areas are sensitive and some are insensitive. 
- But in the back propagation process, if you enter a negative number, the gradient will be completely zero, which has the same problem as the sigmoid function and tanh function.
- Leaky ReLU is An activation function specifically designed to compensate for the dying ReLU problem.
- The leaky ReLU adjusts the problem of zero gradients for negative value, by giving a very small linear component of x to negative inputs(0.01x). The leak helps to increase the range of the ReLU function. Usually, the value of a is 0.01 or so.

[**Day 4**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6922786060552863744-8HUu?utm_source=linkedin_share&utm_medium=member_desktop_web)

**💡 Activation Function (ELU and PReLU):**

-  ELU is also proposed to solve the problems of ReLU. In contrast to ReLUs, ELUs have negative values which push the mean of the activations closer to zero. 
- Mean activations that are closer to zero enable faster learning as they bring the gradient closer to the natural gradient.
- In contrast to ReLUs, ELUs have negative values which allow them to push mean unit activations closer to zero like batch normalization but with lower computational complexity.  Mean shifts toward zero speed up learning by bringing the normal gradient closer to the unit natural gradient because of a reduced bias shift effect.
- ELUs saturate to a negative value with smaller inputs and thereby decrease the forward propagated variation and information.
- One small problem is that it is slightly more computationally intensive.
- PReLU is also an improved version of ReLU. Here we multiply the z with a parameter aᵢ. So, 

        1. if aᵢ=0, f becomes ReLU
        2. if aᵢ>0, f becomes leaky ReLU
        3. if aᵢ is a learnable parameter, f becomes PReLU

- In the negative region, PReLU has a small slope, which can also avoid the problem of ReLU death.
- Compared to ELU, PReLU is a linear operation in the negative region. Although the slope is small, it does not tend to 0, which is a certain advantage.

[**Day 5**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6923153672847704064-rIF8?utm_source=linkedin_share&utm_medium=member_desktop_web)

**💡Softmax Activation Function:**

- Softmax is used as the activation function for multi-class classification problems where class membership is required on more than two class labels.
-  For an arbitrary real vector of length K, Softmax can compress it into a real vector of length K with a value in the range (0, 1), and the sum of the elements in the vector is 1.
-  Softmax is different from the normal max function: the max function only outputs the largest value, and Softmax ensures that smaller values have a smaller probability and will not be discarded directly. It is a “max” that is “soft”; it can be thought to be a probabilistic or “softer” version of the argmax function.
- The major drawback in the softmax activation function is that it is 

        1. Non-differentiable at zero and ReLU is unbounded.
        2. The gradients for negative input are zero, which means for activations in that region, the weights are not updated during backpropagation. 

**💡Swish  Activation Function:**

- Swish’s design was inspired by the use of sigmoid functions for gating in LSTMs and highway networks. We use the same value for gating to simplify the gating mechanism, which is called self-gating
- The advantage of self-gating is that it only requires a simple scalar input, while normal gating requires multiple scalar inputs. This feature enables self-gated activation functions such as Swish to easily replace activation functions that take a single scalar as input (such as ReLU) without changing the hidden capacity or number of parameters.
- Swish activation function can only be implemented when your neural network is ≥ 40 layers.
- Advantages:

        1. Unboundedness is helpful to prevent the gradient from gradually approaching 0 during slow training, causing saturation.
        2. Derivative always >0
        3. Smoothness also plays an important role in optimization and generalization.  

[**Day 6**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6924242722304245761-Pxxx?utm_source=linkedin_share&utm_medium=member_desktop_web)

**💡Bias:**

-  Statistical bias refers to measurement or sampling errors that are systematic and produced by the measurement or sampling process. An important distinction should be made between errors due to random chance and errors due to bias. 
-  Bias comes in different forms, and may be observable or invisible. When a result does suggest bias, it is often an indicator that a statistical or machine learning model has been misspecified, or an important variable left out.
-  Selection bias refers to the practice of selectively choosing data consciously or unconsciously in a way that leads to a conclusion that is misleading or ephemeral.  It is sometimes referred to as the selection effect.
-  Sampling bias is systematic error due to a non-random sample of a population, causing some members of the population to be less likely to be included than others, resulting in a biased sample, defined as a statistical sample of a population in which all participants are not equally balanced or objectively represented.
-  A distinction of sampling bias (albeit not a universally accepted one) is that it undermines the external validity of a test (the ability of its results to be generalized to the rest of the population), while selection bias mainly addresses internal validity for differences or similarities found in the sample at hand. 
-  In this sense, errors occurring in the process of gathering the sample or cohort cause sampling bias, while errors in any process thereafter cause selection bias.
- Ways to avoid bias:

        1. Random Sampling
        2. Stratified sampling

[**Day 7**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6924599968301285376-AfaA?utm_source=linkedin_share&utm_medium=member_desktop_web)

**💡 Normal Distribution:**

- The normal distribution, also known as the Gaussian distribution, is the most important probability distribution in statistics for independent, random variables. Most people recognize its familiar bell-shaped curve in statistical reports.
- The normal distribution is a continuous probability distribution that is symmetrical around its mean, most of the observations cluster around the central peak, and the probabilities for values further away from the mean taper off equally in both directions. 
- The skewness and kurtosis coefficients measure how different a given distribution is from a normal distribution.
- The skewness measures the symmetry of a distribution. The normal distribution is symmetric and has a skewness of zero. The kurtosis statistic measures the thickness of the tail ends of a distribution in relation to the tails of the normal distribution.
- Features:

        1. Symmetric bell shape
        2. Mean and median both are equal and located at the centre of distribution
        3. 68% of data falls within first standard deviation of the mean 
        4. 95% of data falls within second standard deviation of the mean
        5. 99.7% of data falls within third standard deviation of the mean

[**Day 8**](https://www.linkedin.com/posts/vikram--krishna_66daysofdata-datawithvikram-datascience-activity-6924959876737630208-_AQT?utm_source=linkedin_share&utm_medium=member_desktop_web)

**💡 Students t distribution:**

- The t-distribution is a normally shaped distribution, except that it is a bit thicker and longer on the tails. It is used extensively in depicting distributions of sample statistics. 
- Distributions of sample means are typically shaped like a t-distribution, and there is a family of t-distributions that differ depending on how large the sample is. The larger the sample, the more normally shaped the t-distribution becomes.
- The t-distribution plays a role in a number of widely used statistical analyses, including Student's t-test for assessing the statistical significance of the difference between two sample means, the construction of confidence intervals for the difference between two population means, and in linear regression analysis.
- Student's t-distribution also arises in the Bayesian analysis of data from a normal family.
- The t-distribution is symmetric and bell-shaped, like the normal distribution. However, the t-distribution has heavier tails, meaning that it is more prone to producing values that fall far from its mean.
- A number of statistics can be shown to have t-distributions for samples of moderate size under null hypotheses that are of interest, so that the t-distribution forms the basis for significance tests.
- If X has a Student's t-distribution with degree of freedom "v" then X^2 has an F-distribution.
- Student’s t Distribution is used when 

        1. The sample size must be 30 or less than 30.
        2. The population standard deviation(σ) is unknown.
        3. The population distribution must be unimodal and skewed.

[**Day 9**]()

**💡 Binomial distribution:**

- Binomial Distribution has an important place in analytics, since they are often the culmination of a decision or other process such as yes/no and so on. The distribution is obtained by performing a number of Bernoulli trials.
- The binomial distribution is the frequency distribution of the number of successes (x) in a given number of trials (n) with specified probability (p) of success in each trial.
- It must meet the following 3 criteria:

        1. Number of observations or trials is fixed, i.e., you can only figure out the probability of something happening if you do it a certain number of times.
        2. Each observation or trial is independent. In other words, none of your trials have an effect on the probability of the next trial.
        3. The probability of success (tails, heads, fail or pass) is exactly the same from one trial to another.

**💡 Chi Square distribution:**

- An important idea in statistics is departure from expectation, especially with respect to category counts. Expectation is defined loosely as “nothing unusual or of note in the data”. This is also termed as "null hypothesis" or "null test".
- The chi-square statistic is a measure of the extent to which a set of observed values “fits” a specified distribution (a “goodness-of-fit” test). It is useful for determining whether multiple treatments (an “A/B/C... test”) differ from one another in their effects.
-  It has many uses:

        1. Confidence interval estimation for a population standard deviation of a normal distribution from a sample standard deviation 
        2. Independence of two criteria of classification of qualitative variables. 
        3. Relationships between categorical variables (contingency tables).
        4. Tests of deviations of differences between expected and observed frequencies (one-way tables).
        5. The chi-square test (a goodness of fit test).
