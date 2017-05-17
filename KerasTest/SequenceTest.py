from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten
# For a single-input model with 2 classes (binary classification):

model = Sequential()
model.add(Dense(32, activation='relu', input_dim=100))
model.add(Dense(1, activation='sigmoid'))
model.compile(optimizer='rmsprop',
              loss='binary_crossentropy',
              metrics=['accuracy'])

# Generate dummy data
import numpy as np
data = np.random.random((1000, 100))
labels = np.random.randint(2, size=(1000, 1))

# Train the model, iterating on the data in batches of 32 samples
model.fit(data, labels, nb_epoch=10, batch_size=32)

data_feat = model.predict(data)

print(data_feat.shape)
print("====================")
print(data_feat)
