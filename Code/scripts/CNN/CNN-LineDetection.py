import pandas as pd
import numpy as np
import glob
import laspy
import cv2
import matplotlib.pyplot as plt
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from PIL import Image


def GetPathRelations(full_path_to_data):        
    ground_removed_image_paths = []
    laz_point_cloud_paths = []
        
    # Find full path to all images
    for path in glob.glob(full_path_to_data+'/ImagesGroundRemovedSmall/*'):
        ground_removed_image_paths.append(path)
    
    # Find full path to all laz files
    for path in glob.glob(full_path_to_data+'/LazFilesWithHeightRemoved/*'):
        laz_point_cloud_paths.append(path)
            
    ground_removed_image_paths.sort()
    laz_point_cloud_paths.sort()
    assert(len(ground_removed_image_paths)==len(laz_point_cloud_paths))
    return ground_removed_image_paths, laz_point_cloud_paths


def MaxMinNormalize(arr):
    return (arr - np.min(arr))/(np.max(arr)-np.min(arr))

def CastAllXValuesToImage(arr, x_pixels):
    return (MaxMinNormalize(arr))*x_pixels

def CastAllYValuesToImage(arr, y_pixels):
    return (1-MaxMinNormalize(arr))*y_pixels



all_path_relations = GetPathRelations("/home/nxw500/data")
path_tuples = list(zip(*all_path_relations))

# Normalize to -1 and 1
transform_img_gray = transforms.Compose(
    [transforms.Resize((256,256)),
     transforms.ToTensor(),
     transforms.Normalize((0.5,), (0.5,))])

# Open images
trainingImages = []
labelImages = []
    
for num, path in enumerate(path_tuples):
    image_path, laz_path = path
    print(image_path)
    
    # Image to training set
    image = cv2.imread(image_path, cv2.IMREAD_UNCHANGED)
    image = np.where(image >= 0, image, 0)
    image = image/np.max(image)
    image = (image*255).astype(np.uint8)
    
    image = Image.fromarray(image)
    transformed_image = transform_img_gray(image)
    _, x_pixels, y_pixels = transformed_image.shape
    trainingImages.append(transform_img_gray(image))
    
    # Generate labels 
    las = laspy.read(laz_path, laz_backend=laspy.compression.LazBackend.LazrsParallel)
    
    y_values = np.rint(CastAllXValuesToImage(las.X, y_pixels)).astype(np.int32)
    x_values = np.rint(CastAllYValuesToImage(las.Y, x_pixels)).astype(np.int32)
    
    powerline_mask = (las.classification == 14)
    x_powerline_values = x_values[powerline_mask]
    x_powerline_values = np.where(x_powerline_values < x_pixels, x_powerline_values, x_pixels-1)
    x_powerline_values = np.where(x_powerline_values >= 0, x_powerline_values, 0)
    
    y_powerline_values = y_values[powerline_mask]
    y_powerline_values = np.where(y_powerline_values < y_pixels, y_powerline_values, y_pixels-1)
    y_powerline_values = np.where(y_powerline_values >= 0, y_powerline_values, 0)
    
    labels = np.zeros((x_pixels, y_pixels)).astype(np.uint8)
    for i in range(len(x_powerline_values)):
        labels[x_powerline_values[i], y_powerline_values[i]] = 255
    
    # Create kernel
    kernel = np.ones((3, 3), np.uint8)
    #lines_image = cv2.morphologyEx(lines_image, cv2.MORPH_CLOSE, kernel)
    lines_image = cv2.dilate(labels, kernel, iterations=1)
    lines_image = cv2.erode(lines_image, kernel, iterations=1)
    
    #fig, (ax0, ax1, ax2) = plt.subplots(1, 3, figsize=(15,15))
    #ax0.set_title('Image')
    #ax0.imshow(transformed_image[0], cmap='gray')
    #ax1.set_title('Labels')
    #ax1.imshow(labels, cmap='gray')
    #ax2.set_title('Hough Line')
    #ax2.imshow(lines_image, cmap='gray')
    #fig.savefig("image_"+str(num)+".png", dpi=200)
    
    lines_image = Image.fromarray(lines_image)
    labelImages.append(transform_img_gray(lines_image))
    

X_train, X_test, Y_train, Y_test = train_test_split(trainingImages, labelImages, test_size=0.10)
#X_train, X_val, Y_train, Y_val = train_test_split(X_train, Y_train, test_size=0.1)

print(len(X_train), len(Y_train))
print(len(X_test), len(Y_test))
#print(len(X_val), len(Y_val))

# Setting up sets for trainlodader and validation loader
training_set = []
for i in range(len(X_train)):
    training_set.append([X_train[i], Y_train[i]])
    
#validation_set = []
#for i in range(len(Y_val)):
#    validation_set.append([X_val[i], Y_val[i]])
    
test_set = []
for i in range(len(Y_test)):
    test_set.append([X_test[i], Y_test[i]])

trainloader = torch.utils.data.DataLoader(training_set, batch_size=8, shuffle=True, num_workers=4)
#valloader = torch.utils.data.DataLoader(validation_set, batch_size=8, shuffle=True, num_workers=4)
testloader = torch.utils.data.DataLoader(test_set, batch_size=8, shuffle=True, num_workers=4)

#torch.save(valloader, "valloader.pt")
torch.save(testloader, "testloader.pt")
torch.save(trainloader, "trainloader.pt")


class ConvNetRGB(nn.Module):
    def __init__(self):
        super(ConvNetRGB, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(in_channels=1, out_channels=64, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=3, padding = 1),            
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
            ).cuda()
        
        self.layer2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)).cuda()
        
        self.layer3 = nn.Sequential(
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)).cuda()
        
        self.layer4 = nn.Sequential(
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=512, out_channels=512, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)).cuda()
        
        self.layer5 = nn.Sequential(
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=3, padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)).cuda()


        self.down1 = nn.Sequential(nn.ConvTranspose2d(in_channels=1024, out_channels=1024,
                                        stride = 2, kernel_size=3, padding = 1, output_padding=1),
                                   nn.ConvTranspose2d(in_channels=1024, out_channels=512,
                                        stride = 1, kernel_size=3, padding = 1)
                                  ).cuda()
        
        self.down2 = nn.Sequential(nn.ConvTranspose2d(in_channels=512, out_channels=512,
                                        stride = 2, kernel_size=3, padding = 1, output_padding=1),
                                   nn.ConvTranspose2d(in_channels=512, out_channels=256,
                                        stride = 1, kernel_size=3, padding = 1)
                                  ).cuda()
        
        self.down3 = nn.Sequential(nn.ConvTranspose2d(in_channels=256, out_channels=256,
                                        stride = 2, kernel_size=3, padding = 1, output_padding=1),
                                   nn.ConvTranspose2d(in_channels=256, out_channels=128,
                                        stride = 1, kernel_size=3, padding = 1)
                                  ).cuda()
        
        self.down4 = nn.Sequential(nn.ConvTranspose2d(in_channels=128, out_channels=128,
                                        stride = 2, kernel_size=3, padding = 1, output_padding=1),
                                   nn.ConvTranspose2d(in_channels=128, out_channels=64,
                                        stride = 1, kernel_size=3, padding = 1)
                                  ).cuda()
        
        self.down5 = nn.Sequential(nn.ConvTranspose2d(in_channels=64, out_channels=64,
                                        stride = 2, kernel_size=3, padding = 1, output_padding=1),
                                   nn.ConvTranspose2d(in_channels=64, out_channels=1,
                                        stride = 1, kernel_size=3, padding = 1)
                                  ).cuda()
        

    def forward(self, x):
        out = self.layer1(x)
        #print(f"self.layer1 {out.shape}")
        out = self.layer2(out)
        #print(f"self.layer2 {out.shape}")
        out = self.layer3(out)
        #print(f"self.layer3 {out.shape}")
        out = self.layer4(out)
        #print(f"self.layer4 {out.shape}")
        out= self.layer5(out)
        #print(f"self.layer5 {out.shape}")
        out = self.down1(out)
        #print(f"self.down1 {out.shape}")
        out = self.down2(out)
        #print(f"self.down2 {out.shape}")
        out = self.down3(out)
        #print(f"self.down3 {out.shape}")
        out = self.down4(out)
        #print(f"self.down4 {out.shape}")
        out = self.down5(out)
        #print(f"self.down5 {out.shape}")
        #out = torch.sigmoid(out)
        return out
    


def ConvNetTraining(trainloader, valloader, Conv, lossFunction, learning_rate, epochs):
    model = Conv
    num_epochs = epochs
    criterion = lossFunction.cuda()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    
    scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.995)
    
    # loss arrays for figures
    TrainingLossArray = []
    ValidationLossArray = []
    
    early_stopping = 2500
    notImproved = 0
    bestLoss = None
    bestModel = None
    
    for epoch in range(num_epochs):  # loop over the dataset multiple times
        print(f"LR = {scheduler.get_last_lr()[0]:.10f}")
        running_loss = 0.0
        for j, data in enumerate(trainloader):
            # get the input
            inputs, labels = data
            # zero the parameter gradients
            optimizer.zero_grad()
            # forward + backward + optimize            
            outputs = model(inputs.cuda())
            loss = criterion(outputs.cuda(), labels.cuda())
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
        
        # Appending the mean running loss
        TrainingLossArray.append(running_loss/(j+1))
        
        # print("Training loss: ", running_loss/j)
        # Finding validation loss
        validation_loss = 0
        with torch.no_grad():
            for i, data in enumerate(valloader):
                # get the inputs
                inputs, labels = data
                
                #Calculates loss
                outputs = model(inputs.cuda())            
                loss = criterion(outputs.cuda(), labels.cuda())      
                validation_loss += loss.item()
        # Appending the mean validation loss
        ValidationLossArray.append(validation_loss/(i+1))
        # print("Validation loss: ", validation_loss/i)
        print(f"epoch = {epoch}, Validation loss: {validation_loss/(i+1):.10f}, Training loss: {running_loss/(j+1):.10f}")
        
        # Initialising params for early stopping
        if bestLoss == None:
            bestLoss = validation_loss
        
        # Checks for early stopping        
        if validation_loss <= bestLoss:
            notImproved = 0
            bestLoss = validation_loss
            bestModel = model
            torch.save(bestModel, "bestModel.pth")
        else:
            notImproved +=1
        # Converges if the training has not improved for a certain amount of iterations
        if notImproved >= early_stopping:
            break
        scheduler.step()

    torch.save(model, "latestModel.pth")    
    return bestModel, ValidationLossArray, TrainingLossArray


bestModel, ValidationLossArray, TrainingLossArray = ConvNetTraining(trainloader, testloader, ConvNetRGB(), nn.MSELoss(), 0.001, 5000)

with open('valLoss.npy', 'wb') as f:
    np.save(f, np.array(ValidationLossArray))

with open('trainLoss.npy', 'wb') as f:
    np.save(f, np.array(TrainingLossArray))
