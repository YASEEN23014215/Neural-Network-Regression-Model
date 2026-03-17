# Developing a Neural Network Regression Model

## AIM

To develop a neural network regression model for the given dataset.

## THEORY

The objective of this project is to develop a Neural Network Regression Model that can accurately predict a target variable based on input features. The model will leverage deep learning techniques to learn intricate patterns from the dataset and provide reliable predictions.
## Neural Network Model

<img width="1147" height="793" alt="image" src="https://github.com/user-attachments/assets/6030875e-5671-4ba9-a925-9f1b596b20d1" />


## DESIGN STEPS

### STEP 1:

Loading the dataset

### STEP 2:

Split the dataset into training and testing

### STEP 3:

Create MinMaxScalar objects ,fit the model and transform the data.

### STEP 4:

Build the Neural Network Model and compile the model.

### STEP 5:

Train the model with the training data.

### STEP 6:

Plot the performance plot

### STEP 7:

Evaluate the model with the testing data.

## PROGRAM
### Name: YASEEN F
### Register Number: 212223220126
```python
class Neuralnet(nn.Module):
   def __init__(self):
        super().__init__()
        self.n1=nn.Linear(1,10)
        self.n2=nn.Linear(10,20)
        self.n3=nn.Linear(20,1)
        self.relu=nn.ReLU()
        self.history={'loss': []}
   def forward(self,x):
        x=self.relu(self.n1(x))
        x=self.relu(self.n2(x))
        x=self.n3(x)
        return x


# Initialize the Model, Loss Function, and Optimizer
sai_brain=Neuralnet()
criteria=nn.MSELoss()
optimizer=optim.RMSprop(sai_brain.parameters(),lr=0.001)

def train_model(sai_brain,x_train,y_train,criteria,optmizer,epochs=4000):
    for i in range(epochs):
        optimizer.zero_grad()
        loss=criteria(sai_brain(x_train),y_train)
        loss.backward()
        optimizer.step()
        
        sai_brain.history['loss'].append(loss.item())
        if i%200==0:
            print(f"Epoch [{i}/epochs], loss: {loss.item():.6f}")



    



```
## Dataset Information

<img width="430" height="760" alt="image" src="https://github.com/user-attachments/assets/70efad0c-19ae-42e9-bb08-1c888ac1e863" />


## OUTPUT
<img width="514" height="520" alt="image" src="https://github.com/user-attachments/assets/3a6d0d6f-48e1-4ef6-8b6d-e7b3ac704d85" />


### Training Loss Vs Iteration Plot
<img width="791" height="651" alt="image" src="https://github.com/user-attachments/assets/9a9c2862-7942-4d22-b2fb-d15eadec7b71" />



### New Sample Data Prediction
<img width="320" height="103" alt="image" src="https://github.com/user-attachments/assets/004c2dee-be31-4dc9-8c70-42180a1959d2" />


## RESULT
The neural network regression model was successfully trained and evaluated. The model demonstrated strong predictive performance on unseen data, with a low error rate.
