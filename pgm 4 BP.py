import numpy as np
input_data = np.array([[2,9],[1,5],[3,6]], dtype=float) 
expected_output=np.array ([[92], [86], [89]], dtype=float) 
input_data = input_data/np.amax(input_data, axis=0)
expected_output=expected_output/100 
def sigmoid(x):
    return 1/(1+np.exp(-x)) 
def derivative_sigmoid(x):
    return x*(1-x)
epoch=1
learning_rate = 0.1
inputlayer_neurons = 2 
hiddenlayer_neurons=3 
outputlayer_neurons = 1 
hiddenlayer_weights=np.random.uniform(size=(inputlayer_neurons, hiddenlayer_neurons)) 
hiddenlayer_bias = np.random.uniform(size=(1, hiddenlayer_neurons)) 
outputlayer_weights=np.random.uniform(size= (hiddenlayer_neurons, outputlayer_neurons ))  
outputlayer_bias = np.random.uniform(size=(1, outputlayer_neurons)) 
for i in range(epoch):
    hiddenlayer_input = np.dot(input_data, hiddenlayer_weights) 
    hiddenlayer_input = hiddenlayer_input + hiddenlayer_bias 
    hiddenlayer_output = sigmoid(hiddenlayer_input) 
    
    outputlayer_input = np.dot(hiddenlayer_output, outputlayer_weights) 
    outputlayer_input = outputlayer_input + outputlayer_bias
    outputlayer_output=sigmoid(outputlayer_input)
     
    outputlayer_error=expected_output-outputlayer_output 
    outputlayer_gradient=derivative_sigmoid(outputlayer_output) 
    outputlayer_error_correction = outputlayer_error* outputlayer_gradient
   
    hiddenlayer_error = outputlayer_error_correction.dot (outputlayer_weights. T) 
    hiddenlayer_gradient=derivative_sigmoid(hiddenlayer_output) 
    hiddenlayer_error_correction=hiddenlayer_error*hiddenlayer_gradient
    
    outputlayer_weights += hiddenlayer_output.T.dot(outputlayer_error_correction) * learning_rate 
    hiddenlayer_weights += input_data.T.dot(hiddenlayer_error_correction)*learning_rate
    
print("Input : ", input_data)
print("Expected Output :", expected_output)
print("actual output: ",outputlayer_output)
