# Design and Train Neural Network from Scratch
Design a neural network, derive the parameter gradients with respect to loss function and update the parameter weights and update the weight parameters using the gradients without help of inbuilt libraries.

# Introduction
In this assignment. I am trying to design a neural network, derive the parameter gradients with respect to loss function and update the parameter weights and update the weight parameters using the gradients. 
I will be trying to implement the Classification task in MNIST data set. Once the required details are derived, using this detail I will try to code and implement a neural network and train the same. 

# Data Set
The MNIST data set is used in this project. The data set consists of 60000 training and 10000 test images. The images are of the dimension 28x28.

![image](https://user-images.githubusercontent.com/29349268/118030838-6244a980-b398-11eb-9f5e-c27765920ee2.png)

Fig: Above are some of the sample images from the MNIST data set.

# Neural Network Model Design

For this task, a 2 Layer MLP is used. The input dimension of the MLP will be the flattened 28 x28 image i.e. 784x1.  The output dimension will be 10x1 representing the 10 classes( 10 digits). The 1 st layer will have 1000 neurons and output layer with 10 neurons. Below is sample design of the neural network and the flow of values.

![image](https://user-images.githubusercontent.com/29349268/118030996-99b35600-b398-11eb-9b96-e66f1500948a.png)


![image](https://user-images.githubusercontent.com/29349268/118031071-b18ada00-b398-11eb-8bfd-919a3c2090c2.png)

Fig: The figure above shows the operations between the Weight and bias vectors with the given inputs in the forward propagation of the network.

**Input**

Input X to the network will be a flattened matrix of the image with dimension 1x784. This will contain the values of each pixels in the image. 

**First Layer**

The 1st layer of the neural network has 1000 neurons. The dimension of the Weight Parameter Uw matrix will be 784x1000 since the input dimension is 1x784. Each neuron will have an associated bias value. So, for 1000 neurons it will have a bias vector Ubias of dimension 1x1000. 
The output vector of the first layer Y = (X * Uw) + Ubias will have the dimension of 1x1000.

**ReLu activation Layer**

After the first layer a ReLu (Rectified Linear Unit) layer is added to introduce nonlinearity. The equation of ReLu function is _ReLU(x) = (x)+ = max(0,x)_. If the input to ReLu is less than zero, then output will be zero. For positive values output is same as input.

![image](https://user-images.githubusercontent.com/29349268/118031188-d4b58980-b398-11eb-8ed4-5fe3eab2762a.png)

Fig: The above figure demonstrates how ReLu functions for a given input.
The output of the ReLu layer ![image](https://user-images.githubusercontent.com/29349268/118031275-f3b41b80-b398-11eb-9a65-77b7157f5ca8.png) and will have the same dimension as 1x1000.

**Second Layer**

The second layer has 10 neurons which is equal to the number of classes to classify i.e. 10 digits. The dimension of the Weight Parameter Vw matrix will be 1000x10 since the input dimension ![image](https://user-images.githubusercontent.com/29349268/118031478-28c06e00-b399-11eb-9069-cd8ac833d8eb.png) is 1x1000. Each neuron will have an associated bias value. So, for 10 neurons it will have a bias vector Vbias of dimension 1x10. 
The output vector of the second layer ![image](https://user-images.githubusercontent.com/29349268/118031562-442b7900-b399-11eb-9d32-d5dacafbacd6.png) will have the dimension of 1x10. So, Z is the final predicted value of the network.

**Cross Entropy Layer**

The task is classification of 10 digits and the Loss function selected for this task is Cross entropy. Let T be the target vector and Z the prediction, then equation for the loss function is

![image](https://user-images.githubusercontent.com/29349268/118031629-56a5b280-b399-11eb-874e-2c57e5dcc535.png)

where ti is corresponding element in target vector T and P(Zi) is the probability of the  corresponding element in predicted vector Z.
In the above equation probability of the predicted values P(Zi) using the soft max function in the vector Z

![image](https://user-images.githubusercontent.com/29349268/118031803-848af700-b399-11eb-8441-0115e85073c1.png)

So, the probability of one predicted value is equal to the exponent of that value divided by the sum of exponent of all the predicted values. Therefore, if the predicted value is large then the probability of that value will be high and vice versa.
From the loss function, we can observe that the loss will be large if the probability P(Zi) is small and vice versa. The least possible Loss value is zero when P(zi) =1 and largest when P(Zi) = 0.

![image](https://user-images.githubusercontent.com/29349268/118031859-92d91300-b399-11eb-934a-f1d61b5b47db.png)

Fig: The above figure shows the value of Loss compared to the predicted probability. We can observe that the loss will be very high if the predicted probability is low and vice versa.

If the network predicts a lower value for the class that is equal to the Target class, then the Loss will be high and the network will be penalized for this and if the network predicts high value for the positive class in target vector then the loss will be small.

**Target Vector**

Target vector is the ground truth created from the label provided. The target vector is the one hot encoded version of the image label. For example if the label value is 4 then the target vector T will be [0,0,0,0,1,0,0,0,0,0] , so the index associated to digit 4 will be one and rest will be zeros.

# Gradient Derivation of Various Components

In this section, I will be deriving the gradients of each weight and bias with respect to the loss function, which will be used later to update the same weight and bias during the back propagation.

**Gradient of the Loss function L with respect to output vector Z**


The equation for the cross-entropy loss function is 


![image](https://user-images.githubusercontent.com/29349268/118032221-fc592180-b399-11eb-967c-80cdb01e13f0.png)


Here we have 2 cases where i=j and i≠j.


Case i = j then


![image](https://user-images.githubusercontent.com/29349268/118032319-1e52a400-b39a-11eb-97b3-1e9571323799.png)


Case i≠j then 


![image](https://user-images.githubusercontent.com/29349268/118032390-36c2be80-b39a-11eb-958c-c4fa39580f2f.png)


Combining both by chain rule we get 


![image](https://user-images.githubusercontent.com/29349268/118032474-4e9a4280-b39a-11eb-87e5-8495fd23dc26.png)

Splitting the 2 cases i = j and i≠j we get


![image](https://user-images.githubusercontent.com/29349268/118032573-68d42080-b39a-11eb-987c-d33a4344eb96.png)


P(Zi) is the Softmax of the predicted vector Z and ti represents the corresponding  target vector elements in T. Substituting this we will get


![image](https://user-images.githubusercontent.com/29349268/118032630-79849680-b39a-11eb-9c6a-e1236981b438.png)
-------------------------------------------------------------------------------

**Gradient of the Loss function L with respect to Vw , Vbias and YR of second layer**

In this section I will be calculating the gradient of the Loss function with respect to the second layer weight, bias parameters, and the output of the ReLu, YR. I will be calculating dL/dVw and dL/dVbias

![image](https://user-images.githubusercontent.com/29349268/118033288-3840b680-b39b-11eb-8c05-e4f77770fe84.png)
-------------------------------------------------------------------------------


**Gradient of the Loss Function L with respect to the output Y of first layer**

In this section I will be calculating the gradient of Loss function with respect to the output Y of the 1st layer i.e. dL/dY.

![image](https://user-images.githubusercontent.com/29349268/118033376-51e1fe00-b39b-11eb-9bd4-471968ab9e0e.png)
-------------------------------------------------------------------------------


**Gradient of the Loss function L with respect to Uw and Ubias of first layer**

In this section I will be calculating the gradient of the Loss function with respect to the first layer weight and bias parameter. I will be calculating dL/dUw and dL/dUbias.

![image](https://user-images.githubusercontent.com/29349268/118033417-62927400-b39b-11eb-9a9a-424ef8dd1f5d.png)


# Gradient Descent equation for updating Parameter Matrixes

Gradient descent is a first-order iterative optimization algorithm for finding a local minimum of a differentiable function. To find a local minimum of a function using gradient descent, we take steps proportional to the negative of the gradient (or approximate gradient) of the function at the current point.

![image](https://user-images.githubusercontent.com/29349268/118033530-81910600-b39b-11eb-847c-9ddeafb4b77e.png)

Fig: Above figure shows how the weight parameters are updated using the gradients in each learning steps to reduce the loss and moving towards the minimum.
The rate of decrease of the weight parameters is controlled by a hyper parameter called learning rate lr.  
Following are the equation to update the weight and bias parameters using the found gradient in each step.

![image](https://user-images.githubusercontent.com/29349268/118033473-72aa5380-b39b-11eb-888b-4021228dc2c6.png)


Updated parameters and bias give the values for the next step.

# Details of Network Training Code Implementation

Using the details from the above sections, I implemented the network using python. MNIST data set was downloaded and used it for the training and was loaded using the library PyTorch. Matrix related calculations like matrix multiplication, element wise multiplication, transpose etc. were done with the help of NumPy.

**Converting MNIST image to normalized Input**

The MNIST image data value range varies from 0 to 255 which is normalized to range of [0,1] by dividing the whole image matrix by 255. MNIST image matrix dimension is 28x28, this was flattened to 1x784 to pass it as input to the  network.

**Declaring and Initializing the networks weight and bias parameters**

Networks first and second layer weight and bias parameter matrixes Uw , Ub , Vw  and Vb were declared with the required dimensions. A good initial value should be assigned to these matrixes for good network performance. I followed a strategy similar to Xavier initialization but simpler. The weight we initialized using a normal distribution but between the range of [-1/sqrt(n) ,  1/sqrt(n))], where n is the number of inputs to that layer.

**Minibatch gradient descent**

In this assignment, I implemented mini batch gradient descent with a batch size of 100 over the data set containing 60000 training sample. Gradient descents were calculated for each sample in the minibatch and the weights were updated only once for each mini batch. This helps to increase the speed of the training and avoids over fitting. 

**One hot encoding data of labels**

The labels were one hot encoded using a function ‘get_hot_encodedLabels’ for comparing the predicted values and true label.

**Forward propagation**

Using the formatted input X and the weight and bias parameter matrices, the model predicted value Z was calculated by forward propagation. The ReLu function is implemented using the function ‘implement_ReLU’.
The predicted values were converted into probabilities using the softmax calculation. Using these probabilities, Cross Entropy Loss is calculated for each data sample. The total loss of the batch is calculated using the function ‘calculate_loss_bs’.
The percentage of data samples which were predicted wrongly for each batch is calculated using the function ‘getError’. 

**Backward Propagation**

Using the values of the matrices which was populated during the forward propagation is used to find the gradient of Loss with respect to each component. In this step, I did the step by step calculation of gradient from output to the input direction.

**Gradient Descent Step**

Using the above calculated gradients, the weight and bias parameters were updated after iterating each batch. A learning rate value of 0.001 was used for updating the values.

# Networking Training Analysis

The aim of network training is to make the network able to predict the correct class of the digit in the given image. For this the Loss of the network, i.e. the difference between the predicted and true label should be reduced. By gradient descent, we obtain this task by updating the weight and bias parameters in such a way so that the network will have minimum loss.
In this section we will be analysing the Loss and Error percentage of the network during training.

**Loss versus Epoch**

Below is the graph of the Loss values obtained after each epoch of training. So here we can observe that the loss value is decreasing after each epoch and have almost become 0 in the 100th epoch. From this graph we can conclude that the network is learning from the training.

![image](https://user-images.githubusercontent.com/29349268/118033694-b309d180-b39b-11eb-8a28-8b39c0b71638.png)

Fig: Above graph shows the relation between Network training Loss on each epoch.

**Error Percentage versus Epoch**

Below is the graph showing the percentage of images that were classified wrongly in each epoch.  As the number of epochs increases the error percentage decreases. Thus, we can conclude that the network is learning and is able to classify most of the images correctly.  The error of the 100th epoch is 0% for the network.

![image](https://user-images.githubusercontent.com/29349268/118033755-c2891a80-b39b-11eb-8f8d-c10161d86c3e.png)

Fig: Above graph shows the relation between Network training error percentage on each epoch

