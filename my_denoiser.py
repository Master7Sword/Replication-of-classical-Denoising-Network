import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torch.autograd import Variable
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
    train_noises = [0,15,75]

    # Load training set
    print("loading dataset...\n")
    train_img_dir = "D:\Dataset\BSDS\BSDS500\images\\train\data"
    training_set = myDataset(train_img_dir, transform=transform)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    print("# of training samples: %d\n" % int(len(training_set)))

    # Load validation set
    val_img_dir = "D:\Dataset\BSDS\BSDS500\images\\val\data"
    val_set = myDataset(val_img_dir, transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)

    # Build Model
    net = mynet()
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
        for i, data in enumerate(train_loader, 0): # 多加个 0 可以获得迭代的索引
            for int_noise_sigma in train_noises:
                model.train()
                model.zero_grad()
                optimizer.zero_grad()

                noise_sigma = int_noise_sigma / 255
                new_images = add_batch_noise(data,noise_sigma)
                noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(new_images.shape[0])]))

                new_images = new_images.cuda()
                noise_sigma = Variable(noise_sigma).cuda()

                out_train = model(new_images, noise_sigma)
                loss = criterion(out_train, data.to(device)) 
                epoch_loss += loss.item()
                loss.backward()
                optimizer.step()

                # Evaluation
                model.eval()
                psnr_train = batch_PSNR(out_train, data.to(device), 1.)
                print("[epoch %d][%d/%d] [Noise_Sigma: %d]  loss: %.4f PSNR_train: %.4f" % (epoch+1, i+1, len(train_loader),int_noise_sigma, loss.item(), psnr_train))
            
        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss)

        # Validate
        val_noises = [0,30,60]
        psnr_val = 0
        count = 0
        model.eval()
        with torch.no_grad():
            for i, val_data in enumerate(val_loader,):
                for val_noise in val_noises:
                    
                    noise_sigma = val_noise / 255
                    new_images = add_batch_noise(val_data, noise_sigma)
                    noise_sigma = torch.FloatTensor(np.array([noise_sigma for idx in range(new_images.shape[0])]))
                    new_images = new_images.cuda()
                    noise_sigma = Variable(noise_sigma).cuda()
                    
                    out_val = model(new_images, noise_sigma)
                    
                    psnr_val += batch_PSNR(out_val, val_data, 1.)   
                    count += 1

        psnr_val /= count
        print("\n[epoch %d] PSNR_val: %.4f\n" % (epoch+1, psnr_val))
    
    torch.save(model.state_dict(), 'FFDNet.pth')


if __name__ == "__main__":
    main()