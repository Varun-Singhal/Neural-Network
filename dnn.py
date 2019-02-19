import numpy as np
from PIL import Image
import sys
import matplotlib.pyplot as plt
import pickle
import pickle
import pyautogui
import cv2
import webbrowser as wb

def initialise(xh,nx,ny,n):
	params={}
	for i in range(xh):
		if i==0:
			params['w'+str(i+1)]=np.random.randn(n[i],nx)*0.01
			params['b'+str(i+1)]=np.random.randn(n[i],1)*0.01
		else:
			params['w'+str(i+1)]=np.random.randn(n[i],n[i-1])*0.01
			params['b'+str(i+1)]=np.random.randn(n[i],1)*0.01
	params['w'+str(xh+1)]=np.random.randn(ny,n[xh-1])*0.01
	params['b'+str(xh+1)]=np.random.randn(ny,1)*0.01
	for item,value in params.items():
		print(item,value.shape)
	return(params)	



def forward(params,data,xh):
	cache={}
	cache['a0']=data
	#keep_probs=0.6
	for i in range(xh):
		cache['z'+str(i+1)]=np.dot(params['w'+str(i+1)],cache['a'+str(i)])+params['b'+str(i+1)]
		cache['a'+str(i+1)]=np.tanh(cache['z'+str(i+1)])
	cache['z'+str(xh+1)]=np.dot(params['w'+str(xh+1)],cache['a'+str(xh)])
	cache['a'+str(xh+1)]=1/(1+np.exp(-cache['z'+str(xh+1)]))
	
	

	#for item,value in cache.items():
		#print(item,value.shape,value)
	return cache
def test(xh,params):
	cap=cv2.VideoCapture(0)
	wb.open_new("http://www.google.com")
	while(True):
		ret,frame=cap.read()
		
		cv2.imshow('frame',frame)
		frame=cv2.resize(frame,(100,100))
		if cv2.waitKey(1) & 0xFF==ord('q'):
			break
		arr=np.array(frame)
		arr=arr.reshape(100*100*3,1)
		cache=forward(params,arr,xh)
		#print(cache['a'+str(xh+1)])
		
		if(cache['a'+str(xh+1)])>0.5:
			print("Jump")
			pyautogui.press('space')
		else:
			print("Stagnent")

def calculate_cost(a,y,m):
	loss=-np.multiply(y,np.log(a))-np.multiply((1-y),np.log(1-a))
	cost=np.sum(loss)/m
	return cost

def backward_propogation(cache,params,xh,m):
	ds={}
	ds['dz'+str(xh+1)]=cache['a'+str(xh+1)]-y
	ds['db'+str(xh+1)]=np.sum(ds['dz'+str(xh+1)],keepdims=True,axis=1)/m
	ds['dw'+str(xh+1)]=np.dot(ds['dz'+str(xh+1)],cache['a'+str(xh)].T)/m
	ds['da'+str(xh+1)]=np.dot(params['w'+str(xh+1)].T,ds['dz'+str(xh+1)])
	for i in range(xh,0,-1):
		ds['dz'+str(i)]=np.multiply(ds['da'+str(i+1)],(1-np.power(cache['a'+str(i)],2)))
		ds['db'+str(i)]=np.sum(ds['dz'+str(i)],keepdims=True,axis=1)/m
		ds['dw'+str(i)]=np.dot(ds['dz'+str(i)],cache['a'+str(i-1)].T)/m
		ds['da'+str(i)]=np.dot(params['w'+str(i)].T,ds['dz'+str(i)])
		assert(ds['dw'+str(i)].shape==params['w'+str(i)].shape)
		assert(ds['db'+str(i)].shape==params['b'+str(i)].shape)
		#for i,j in ds.items():
			#print(i,j)
	return ds

def update_parameters(ds,params,xh,learning_rate=0.01):
	for i in range(1,xh+1):
		params['w'+str(i)]-=learning_rate*ds['dw'+str(i)]
		params['b'+str(i)]-=learning_rate*ds['db'+str(i)]
	params['w'+str(xh+1)]-=learning_rate*ds['dw'+str(xh+1)]
	params['b'+str(xh+1)]-=learning_rate*ds['db'+str(xh+1)]
	return params

def forward_propogation(data,size,y,m):
	nx=size
	xh=int(input("Enter the number of hidden layer"))
	temp=input("Enter the dimensions of hidden layer")
	temp=temp.split(" ")
	n=[]
	for i in temp:
		n.append(int(i))
	ny=int(input("Enter the dimensions of output layer"))
	params=initialise(xh,nx,ny,n)
	costs=[]
	for i in range(20000):
		cache=forward(params,data,xh)
		cost=calculate_cost(cache['a'+str(xh+1)],y,m)
		print(cost,20000-i)
		costs.append(cost)
		ds=backward_propogation(cache,params,xh,m)
		params=update_parameters(ds,params,xh)
	plt.plot(costs)
	plt.show()
	#with open("dnn.pickle",'wb') as file:
		#pickle.dump(params,file)
	test(xh,params)
	


if __name__ == '__main__':
	with open("dnn.pickle","rb") as file:
		inp=pickle.load(file)
	size=100*100*3
	number_of_pics=inp['count']
	y=inp['target']
	data=inp['data']
	y=np.array(y)
	"""number_of_pics=int(sys.argv[1])
	for i in range(1,number_of_pics+1):
		image=Image.open('hand/'+str(i)+".jpg")
		image=image.resize((64,64),Image.ANTIALIAS)
		image=np.array(image)
		image=image.reshape(64*64*3,1)
		if i==1:
			data=np.array(image)
			print(data/255)
		else:
			data=np.column_stack((data,image))
			print(image/255)
	size=64*64*3
	data=data/255
	y=np.array([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1])"""
	print(data)
	forward_propogation(data,size,y,number_of_pics)
	
	
