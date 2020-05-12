from torchvision import models
from torch import nn, optim
import torch
import json

class my_model:
    filepath = 'checkpoint.pth'
    epochs_trained = 0
    drop_p = 0.5
    arch = "vgg11"
    hidden_units = 4096
    with open('cat_to_name.json', 'r') as f:
        cat_to_name = json.load(f)

    def __init__(self, arch, hidden_units):

        self.arch = arch
        self.hidden_units = hidden_units
        
        try:
            self.model = getattr(models, arch)(pretrained=True)
        except AttributeError:
            raise NotImplementedError("Class `{}` does not implement `{}`".format(models.__class__.__name__, arch))

        for param in self.model.parameters():
            param.requires_grad = False    
        self.criterion = nn.NLLLoss()
    
        layers = [25088, int(hidden_units), 102]
        self.model.classifier = nn.Sequential(
                              (nn.Linear(layers[0], layers[1])),
                              (nn.ReLU()),
                              (nn.Dropout(p=self.drop_p)),
                              (nn.Linear(layers[1], layers[2])),
                              (nn.LogSoftmax(dim=1))
                              )
    
    def validation(self, validloader, gpu):
        accuracy = 0
        validaion_loss = 0
        device = "cpu"
        if gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(device);
        # Model in inference mode, dropout is off
        self.model.eval()
        # Turn off gradients for validation, will speed up inference
        with torch.no_grad():
            for images, labels in validloader:
                images, labels = images.to(device), labels.to(device)
                #images = images.resize_(images.size()[0], 784)
                output = self.model.forward(images)
                validaion_loss += self.criterion(output, labels).item()
                ## Calculating the accuracy 
                # Model's output is log-softmax, take exponential to get the probabilities
                ps = torch.exp(output)
                # Class with highest probability is our predicted class, compare with true label
                equality = (labels.data == ps.max(1)[1])
                # Accuracy is number of correct predictions divided by all predictions, just take the mean
                accuracy += equality.type_as(torch.FloatTensor()).mean()

        return validaion_loss, accuracy


    def train(self, trainloader, validloader, epochs=5, gpu=True, learning_rate=0.003):
        device = "cpu"
        if gpu:
            device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        optimizer = optim.Adam(self.model.classifier.parameters(), lr=learning_rate)
        self.model.to(device)
        steps = 0
        running_loss = 0
        for e in range(int(epochs)):
            # Model in training mode, dropout is on
            self.model.train()
            for images, labels in trainloader:
                images, labels = images.to(device), labels.to(device)
                steps += 1
                optimizer.zero_grad()
                output = self.model.forward(images)
                loss = self.criterion(output, labels)
                loss.backward()
                optimizer.step()
                running_loss += loss.item()
                print_every = 5
                if steps % print_every == 0:
                    validaion_loss, accuracy = self.validation(validloader, gpu)
                    print("Epoch: {}/{}.. ".format(e+1, epochs),
                          "Training Loss: {:.3f}.. ".format(running_loss/print_every),
                          "Validation Loss: {:.3f}.. ".format(validaion_loss/len(validloader)),
                          "Validation Accuracy: {:.3f}".format(accuracy/len(validloader)))
                    running_loss = 0
        self.epochs_trained += epochs

    def save(self, save_dir, class_to_idx):
        
        # DONE: Save the checkpoint
        checkpoint = {
                      'epochs_trained': self.epochs_trained,
                      'class_to_idx': class_to_idx,
                      'arch': self.arch,
                      'hidden_units': self.hidden_units,
                      'state_dict': self.model.state_dict()}
        torch.save(checkpoint, save_dir+self.filepath)


    def getModel(self):
        return self.model