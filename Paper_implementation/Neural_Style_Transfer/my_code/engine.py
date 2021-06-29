from model import VGG
import torch
from torchvision import transforms
from PIL import Image
from torchvision.utils import save_image
from loss import total_loss
from tqdm import tqdm

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def preprocess(image):
    transform = transforms.Compose([
        transforms.Resize((512, 512)),
        transforms.ToTensor()
        #transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
        
    ])
    
    image = transform(image)
    image = image.unsqueeze(0)
    image = image.to(device)

    return image

if __name__ == "__main__":

    content_path = r"C:\Users\yasht\Desktop\neural_style_transfer\content_image.jpg"
    style_path = r"C:\Users\yasht\Desktop\neural_style_transfer\style_image.jpg"
    content_image = Image.open(content_path)   
    style_image = Image.open(style_path)  
    #prepare images
    content_image = preprocess(content_image)
    style_image = preprocess(style_image)
    generated_image = content_image.clone().requires_grad_(True)
    
    #model
    model = VGG().to(device).eval()
    #config
    epochs = 100
    
    #training
    optimizer = torch.optim.SGD([generated_image], lr = 0.003)
    for epoch in tqdm(range(epochs)):
        content_features = model(content_image)
        style_features = model(style_image)
        generated_features = model(generated_image)

        loss = total_loss(content_features, style_features, generated_features)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

save_image(generated_image, "generated_image_2.jpg")

    



    
