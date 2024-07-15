import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms # 用于转Tensor
from my_net import mynet
from utils import *

class myDataset(Dataset):
    def __init__(self, img_dir, transform = None):
        self.img_dir = img_dir
        self.image_filenames = [f for f in os.listdir(img_dir)]
        self.transform = transform
    
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, idx):
        img_name = os.path.join(self.img_dir, self.image_filenames[idx])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        return image
    
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor()
])

def main():
    # Params
    lr = 1e-3
    epochs = 100
    batch_size = 16
    noiseL = 25

    # Load training set
    print("loading dataset...\n")
    train_img_dir = "D:\Dataset\BSDS\BSDS500\images\\train\data"
    training_set = myDataset(train_img_dir, transform=transform)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(training_set)))

    # Load validation set
    val_img_dir = "D:\Dataset\BSDS\BSDS500\images\\val\data"
    val_set = myDataset(val_img_dir, transform=transform)

    # Build Model
    net = mynet(channels = 3)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss()    

    # GPU
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device_ids = [0]
    model = nn.DataParallel(net,device_ids=device_ids).cuda()
    criterion.cuda()

    # 
    optimizer = torch.optim.Adam(model.parameters(), lr = lr)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5, factor=0.5, verbose=True)

    # Train
    for epoch in range(epochs):
        epoch_loss = 0.0
        for i, data in enumerate(train_loader, 0):
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data.to(device)

            noise = torch.FloatTensor(img_train.size()).normal_(mean=0, std = noiseL/255.).to(device)

            imgn_train = img_train + noise

            out_train = model(imgn_train)
            loss = criterion(out_train, img_train) 
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Evaluation
            model.eval()
            psnr_train = batch_PSNR(out_train, img_train, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train:  %.4f" % (epoch+1, i+1, len(train_loader), loss.item(), psnr_train))
        
        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss)

        # Validate
        psnr_val = 0
        model.eval()
        with torch.no_grad():
            for k in range(len(val_set)):
                # Get the image from the dataset
                img_val = val_set[k].to(device)
                
                # Add a batch dimension
                img_val = torch.unsqueeze(img_val, 0)
                
                # Add noise
                noise = torch.FloatTensor(img_val.size()).normal_(mean=0, std=noiseL/255.).to(device)
                imgn_val = img_val + noise
                
                # Forward pass
                out_val = model(imgn_val)
                
                # Calculate PSNR
                psnr_val += batch_PSNR(out_val, img_val, 1.)
                
        psnr_val /= len(val_set)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
    
    torch.save(model.state_dict(), 'mynet.pth')



if __name__ == "__main__":
    main()