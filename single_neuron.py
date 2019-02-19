#Single Neuron

from PIL import Image #Library to read images
import numpy as np 
import matplotlib.pyplot as plt

#Sigmoid Activation
def sigmoid(z): 
	a = 1/(1+np.exp(-z))
	return(a) 

#Calculating Linear Function
def forward_propogation(w,b,data): 
	z = np.dot(w.T,data)+b
	a = sigmoid(z)
	return(z,a)

#Calculation of cost (Difference between actual value and predicted value)
def calculate_loss(a,y,m): 
	loss =- np.multiply(y,np.log(a))-np.multiply((1-y),np.log(1-a))
	cost = np.sum(loss)/m
	return cost

#Calculation of gradient (partial differentiations)
def caculate_gradient(a,y,m,data): 
	dz = a-y
	dw = np.sum((data*dz),axis=1,keepdims=True)/m
	db = np.sum(dz,axis=1,keepdims=True)/m
	return (dz,dw,db)

#Updating parameters in every iteration
def update_parameters(dw,db,w,b):
	learning_rate = 0.001
	w -= learning_rate*dw
	b -= learning_rate*db
	return w,b

#Master - initiates training and plot cost graph
def train_network(data,size,y,m):
	w = np.zeros((size,1))
	b = 0
	losses = []
	for iterator in range(100):
		z,a = forward_propogation(w,b,data)
		loss = calculate_loss(a,y,m)
		dz,dw,db = caculate_gradient(a,y,m,data)
		w,b = update_parameters(dw,db,w,b)
		print(loss)
		losses.append(loss)
	plt.title("Cost Graph")
	plt.plot(losses)
	plt.show()	

#Main function to read images and convert them to numpy array to feed into network
if __name__ == '__main__':
	number_of_pics = 5
	for i in range(1,number_of_pics+1):
		image = Image.open('images/'+str(i)+".jpg")
		image = image.resize((64,64),Image.ANTIALIAS)
		image = np.array(image)
		image = image.reshape(64*64*3,1)
		if i==1:
			data = np.array(image)
		else:
			data = np.column_stack((data,image))
	size = 64*64*3
	data = data/255
	y = np.array([1,1,1,1,1])
	train_network(data,size,y,number_of_pics)
