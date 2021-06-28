import torch
from PIL import Image
from torchvision import transforms
from model import VGG
from torch import optim
from loss import calculate_loss
from torchvision.utils import save_image
from tqdm import tqdm


#defing a function that will load the image and perform the required preprocessing and put it on the GPU
def image_loader(path):
    image=Image.open(path)
    #defining the image transformation steps to be performed before feeding them to the model
    loader=transforms.Compose([transforms.Resize((512,512)), transforms.ToTensor()])
    #The preprocessing steps involves resizing the image and then converting it to a tensor
    image=loader(image).unsqueeze(0)
    return image.to(device,torch.float)

if __name__ == "__main__":
    #Loading the original and the style image
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    original_image = image_loader("content_image.jpg")
    style_image = image_loader("style_image.jpg")
    #Creating the generated image from the original image
    generated_image=original_image.clone().requires_grad_(True)

    model=VGG().to(device).eval() 

    #initialize the paramerters required for fitting the model
    epoch=100
    lr=0.004
    

    #using adam optimizer and it will update the generated image not the model parameter 
    optimizer=optim.Adam([generated_image],lr=lr)

    for e in tqdm(range(epoch)):
        #extracting the features of generated, content and the original required for calculating the loss
        gen_features=model(generated_image)
        orig_feautes=model(original_image)
        style_featues=model(style_image)
        
        #iterating over the activation of each layer and calculate the loss and add it to the content and the style loss
        total_loss=calculate_loss(gen_features, orig_feautes, style_featues)
        #optimize the pixel values of the generated image and backpropagate the loss
        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()
        #print the image and save it after each 100 epoch
        if(e/100):
            #print(total_loss)
            
            save_image(generated_image,"gen.png")

    print("completed!!!")
