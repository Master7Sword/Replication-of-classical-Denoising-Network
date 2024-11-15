import os
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms # 用于转Tensor
from my_net import mynet
from utils import *

class SIDD_Dataset(Dataset):
    def __init__(self, root_dir, transform=None, patch_size=512, stride=384):
        self.root_dir = root_dir
        self.transform = transform
        self.image_pairs = []
        self.patch_size = patch_size
        self.stride = stride
        self.patches = []

        # 遍历所有文件夹，收集图像对
        for subdir in os.listdir(root_dir):
            subdir_path = os.path.join(root_dir, subdir)
            if os.path.isdir(subdir_path):
                images = sorted(os.listdir(subdir_path))
                if len(images) == 2:
                    ground_truth_path = os.path.join(subdir_path, images[0])
                    noisy_image_path = os.path.join(subdir_path, images[1])
                    self.image_pairs.append((ground_truth_path, noisy_image_path))

        # 提取所有图像对的块
        self._extract_patches()

    def _extract_patches(self):
        for gt_path, noisy_path in self.image_pairs:
            ground_truth = Image.open(gt_path).convert('RGB')
            noisy_image = Image.open(noisy_path).convert('RGB')
            ground_truth = np.array(ground_truth)
            noisy_image = np.array(noisy_image)

            h, w, _ = ground_truth.shape
            for i in range(0, h - self.patch_size + 1, self.stride):
                for j in range(0, w - self.patch_size + 1, self.stride):
                    gt_patch = ground_truth[i:i+self.patch_size, j:j+self.patch_size]
                    noisy_patch = noisy_image[i:i+self.patch_size, j:j+self.patch_size]
                    self.patches.append((gt_patch, noisy_patch))

    def __len__(self):
        return len(self.patches)

    def __getitem__(self, idx):
        gt_patch, noisy_patch = self.patches[idx]

        gt_patch = Image.fromarray(gt_patch)
        noisy_patch = Image.fromarray(noisy_patch)

        if self.transform:
            gt_patch = self.transform(gt_patch)
            noisy_patch = self.transform(noisy_patch)

        return gt_patch, noisy_patch

# 使用自定义数据集类
transform = transforms.Compose([
    transforms.ToTensor()
])

def main():
    # Params
    lr = 1e-3
    epochs = 100
    batch_size = 16

    # Load training set
    print("loading dataset...\n")
    train_img_dir = "D:\\Dataset\\SIDD_Small_sRGB_Only\\train"
    training_set = SIDD_Dataset(train_img_dir, transform=transform)
    train_loader = DataLoader(training_set, batch_size=batch_size, shuffle=True)
    print("# of training samples: %d\n" % len(training_set))

    # Load validation set (如有必要)
    val_img_dir = "D:\\Dataset\\SIDD_Small_sRGB_Only\\val"
    val_set = SIDD_Dataset(val_img_dir, transform=transform)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=True)
    print("# of validating samples: %d\n" % len(val_set))

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
            im_gt, im_noisy = [x.cuda() for x in data]

            out_train = model(im_noisy)
            loss = criterion(out_train, im_gt) 
            epoch_loss += loss.item()
            loss.backward()
            optimizer.step()

            # Evaluation
            model.eval()
            psnr_train = batch_PSNR(out_train, im_gt, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train:  %.4f" % (epoch+1, i+1, len(train_loader), loss.item(), psnr_train))
        
        epoch_loss /= len(train_loader)
        scheduler.step(epoch_loss)

        # Validate
        psnr_val = 0
        model.eval()
        with torch.no_grad():
            for i, data in enumerate(val_loader, 0):
                im_gt, im_noisy = [x.cuda() for x in data]
                
                # Forward pass
                out_val = model(im_noisy)
                
                # Calculate PSNR
                psnr_val += batch_PSNR(out_val, im_gt, 1.)
                
        psnr_val /= len(val_set)
        print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))


if __name__ == "__main__":
    main()