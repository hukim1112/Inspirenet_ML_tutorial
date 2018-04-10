# Week 9. April 10th 2018

## Density estimation(Probability density estimation)

- Estimating probability distribution(fundamental characteristic) of random variables from observed data

- Parametric(모수적) VS Non-Parametric(비모수적)  

  - Whether you assume specific distribution for distribution of data 

  ​

Here is a problem called Anomaly Detection which is identifying somethings look strange and uncommon.

We are going to use density estimation for finding outliers.

![week9_picture1](images/9week/week9_picture1.png)

$p(x)$ : probability of that x is okay; x are features of data

In the figure above, the point in the middle would be okay because $p(x)$ is greater than or equal to some small threshold(epsilon). Whereas the point on the right bottom would be anomalous since has lower possibility.

![week9_picture2](images/9week/week9_picture2.png)



## Gaussian Distribution(normal distribution)

![week9_picture3](images/9week/week9_picture3.png)

![week9_picture4](images/9week/week9_picture4.png)

1. The area under the curve is "one"

2. Mean affect shifting 

3. Standard deviation has an effect on width and height

   ​

![week9_picture5](images/9week/week9_picture5.png)

Let's suppose that these examples came from a Gaussian distribution.

We can get $\mu$ and $\sigma$ with formula written above.

![week9_picture6](images/9week/week9_picture6.png)

 We can write $p(x)$ more compactly with expression product of probabilities of each features.



Here is anomaly detection algorithm process.

![week9_picture7](images/9week/week9_picture7.png)

![week9_picture8](images/9week/week9_picture8.png)

$p(x)$ is the height of 3-D surface plot. 

So the region which is painted with pink color is outliers-area. 



# Developing and Evaluating an Anomaly Detection System

![week9_picture9](images/9week/week9_picture9.png)

If we have labeled data, we are able to evaluate the algorithm. 

Let's see an example.

![week9_picture10](images/9week/week9_picture10.png)

- Training set : 60% (with normal data)
- Cross Validation set : 20% (+ half of anomalous data)
- Test set : 20% (+ half of anomalous data)



![week9_picture11](images/9week/week9_picture11.png)

- Prediction value depends on threshold(epsilon)

- Evaluation with predicted label and real label -> using evaluation metrics method

   

# Anomaly Detection VS Supervised Learning

Why do we use Anomaly Detection algorithm, not using supervised learning algorithm even if we have labeled data?

![week9_picture12](images/9week/week9_picture12.png)

- Not balanced data could not fully explain many types of 'anomalies'
- New 'anomaly' examples we've never seen so far



# Choosing What Features to Use

If features doesn't follow Gaussian distribution, you could do some transformations.  

![week9_picture13](images/9week/week9_picture13.png)

![week9_picture14](images/9week/week9_picture14.png)

![week9_picture15](images/9week/week9_picture15.png)

We could make new features to capture unusually large or small values as anomalies.



# Multivariate Gaussian Distribution

![week9_picture16](images/9week/week9_picture16.png)

Anomaly Detection algorithm will fail to flag the green point as an anomaly.

In short, Anomaly Detection algorithm doesn't realize blue ellipse shape for high probability region.

![week9_picture17](images/9week/week9_picture17.png)



![week9_picture18](images/9week/week9_picture18.png)

![week9_picture19](images/9week/week9_picture19.png)

![week9_picture20](images/9week/week9_picture20.png)

![week9_picture21](images/9week/week9_picture21.png)

![week9_picture22](images/9week/week9_picture22.png)



# Anomaly Detection using the Multivariate Gaussian Distribution

![week9_picture23](images/9week/week9_picture23.png)



![week9_picture24](images/9week/week9_picture24.png)

Original model is a special case of multivariate Gaussian model which has all zero elements off the diagonal.



![week9_picture25](images/9week/week9_picture25.png)

If $\Sigma$ is non-invertible when you implement multivariate Gaussian model, check these two.

1. m(training set size) > n(the number of features)
2. check for redundant features