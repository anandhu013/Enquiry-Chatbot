import nltk
from nltk.stem.lancaster import LancasterStemmer
lanc=LancasterStemmer()


import numpy
import tflearn
import tensorflow
import json
import random
import pickle

with open("options.json") as file:
	data=json.load(file)


words=[]
labels=[]
docs_x=[]
docs_y=[]

for intent in data["options"]:
	for pattern in intent["questions"]:
		wrds=nltk.word_tokenize(pattern)
		words.extend(wrds)
		docs_x.append(wrds)
		docs_y.append(intent["tag"])
			
	if intent["tag"] not in labels:
		labels.append(intent["tag"])

words=[lanc.stem(w.lower()) for w in words if w !='?']
words=sorted(list(set(words)))

labels=sorted(list(set(labels)))

training=[]
output=[]

output_emp=[0 for _ in range(len(labels))]

for x,doc in enumerate(docs_x):
	bag=[]
	wrds=[lanc.stem(p.lower()) for p in doc]

	for w in words:
		if w in wrds:
			bag.append(1)
		else: 
			bag.append(0)

	output_row=output_emp[:]
	output_row[labels.index(docs_y[x])]=1
	training.append(bag)
	output.append(output_row)

training=numpy.array(training)
output=numpy.array(output)




net=tflearn.input_data(shape=[None,len(training[0])])
net=tflearn.fully_connected(net,32)
net=tflearn.fully_connected(net,16)
net=tflearn.fully_connected(net,len(output[0]),activation="softmax")
net=tflearn.regression(net)

model=tflearn.DNN(net)

model.fit(training,output,n_epoch=1000,batch_size=8,show_metric=True)
model.save("model.tflearn")


def bag_of_words(s,words):
	bag=[0 for _ in range(len(words))]

	s_words=nltk.word_tokenize(s)
	s_words=[lanc.stem(l.lower()) for l in s_words]
	for s_word in s_words:
		for i,w in enumerate(words):
			if w== s_word:
				bag[i]=1

	return numpy.array(bag)

def chat():

	print("Start the talking(type quit to end)")
	while True:
		inp=input("You :")
		if inp=="quit":
			break
		
		results=model.predict([bag_of_words(inp,words)])[0]
		res_index=numpy.argmax(results)
		tag=labels[res_index]

		if results[res_index]>0.7:

			for tg in data['options']:
				if tg['tag']==tag:
					answers=tg['answers']

			print(random.choice(answers))
		else:
			print("I did not get you.")

chat()


 