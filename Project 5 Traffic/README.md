Trial 1:
1 x CNN layer (32 neurons)
2 x CNN layer (64 neurons)
333/333 - 2s - loss: 0.1545 - accuracy: 0.9614 - 2s/epoch - 6ms/step

Trial 1 was done with only CNN layers. This already yielded a high accuracy of 96.14%.

Trial 2:
1 x CNN layer (32 neurons)
2 x CNN layer (64 neurons)
1 x hidden layer (128 neurons)
333/333 - 2s - loss: 0.1520 - accuracy: 0.9586 - 2s/epoch - 5ms/step

Trial 2 was done with the CNN layers from Trial 1, along with a hidden layer containing 128 neurons. This caused a fall in accuracy to 95.86%. We can see that a hidden layer may not be useful, or that there were too many neurons in the layer. 

Trial 3:
1 x CNN layer (32 neurons)
2 x CNN layer (64 neurons)
1 x hidden layer (64 neurons)
333/333 - 2s - loss: 0.1327 - accuracy: 0.9694 - 2s/epoch - 6ms/step

Trial 3 was done with the same CNN layers and a hidden layer with 64 neurons. This caused a rise in accuracy to 96.94%. We can see that the reason behind the fall in accuracy was the number of neurons in the hidden layer. A hidden layer has in fact helped to increase the accuracy. 

From this, I have learnt that adding more layers and more neurons might not necessarily increase the accuracy. Drop out layers are also important to ensure that the model does not overfit. 