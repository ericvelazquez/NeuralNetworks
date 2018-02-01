import glob
from keras.datasets import cifar10
file_values = []
outputs = ["Airplane ","Automobile ","Bird " ,"Cat " ,"Deer ","Dog ","Frog ","Horse " ,"Ship ","Truck "]
for filename in glob.glob('voting*.txt'):
	file = open(filename,"r") 
	value = file.read().splitlines()
	file_values.append(value)

print(" Networks Voting: "+ str(len(file_values)))
final = []
for i,element in enumerate(file_values[0]):
	aux = []
	for j,val in enumerate(file_values):
		aux.append(int(file_values[j][i]))
	final.append([max(set(aux), key=aux.count)])

(x_train, y_train), (x_test, y_test) = cifar10.load_data()
acc = 0
n = 0
for i,e in enumerate(final):
	n+=1
	if e == y_test[i]:
		acc +=1
accuracy = float(100*acc/n)
print("Voting System accuracy: " + str(accuracy) + "%")
