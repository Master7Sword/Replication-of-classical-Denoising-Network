import os
import sys
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms # 用于转Tensor
from pathlib import Path
from my_net import *
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

def train_step_P(net, x, y, optimizerP):
    alpha = 0.5

    net['P'].zero_grad()

    # Real data
    real_data = torch.cat([x,y], 1)
    real_loss = net['P'](real_data).mean()

    # Generator fake data
    with torch.autograd.no_grad():
        fake_y = sample_generator(net['G'], x)
        fake_y_data = torch.cat([x, fake_y], 1)
    fake_y_loss = net['P'](fake_y_data.data).mean()
    grad_y_loss = gradient_penalty(real_data, fake_y_data, net['P'], lambda_gp=10)
    loss_y = alpha * (fake_y_loss - real_loss)
    loss_yg = alpha * grad_y_loss

    # Denoiser fake data
    with torch.autograd.no_grad():
        fake_x = y - net['D'](y) # Denoiser预测的是噪声
        fake_x_data = torch.cat([fake_x, y], 1)
    fake_x_loss = net['P'](fake_x_data).mean()
    grad_x_loss = gradient_penalty(real_data, fake_x_data, net['P'], lambda_gp=10)
    loss_x = (1-alpha) * (fake_x_loss - real_loss)
    loss_xg = (1-alpha) * grad_x_loss

    loss = loss_x + loss_xg + loss_y + loss_yg

    # Backward
    loss.backward()
    optimizerP.step()

    return loss, loss_x, loss_xg, loss_y, loss_yg

def train_step_G(net ,x, y, optimizerG):
    alpha = 0.5
    tau_G = 10

    net['G'].zero_grad()

    fake_y = sample_generator(net['G'], x)
    loss_mean = tau_G * mean_match(x, y, fake_y, kernel.to(x.device), 3)
    fake_y_data = torch.cat([x, fake_y], 1)
    fake_y_loss = net['P'](fake_y_data).mean()
    loss_y = -alpha * fake_y_loss
    loss = loss_y + loss_mean

    # Backward
    loss.backward()
    optimizerG.step()

    return loss, loss_y, loss_mean, fake_y.data

def train_step_D(net, x, y, optimizerD):
    alpha = 0.5
    tau_D = 1000

    net['D'].zero_grad()

    # print(f"[S] {y.shape}")
    fake_x = y - net['D'](y)
    mae_loss = F.l1_loss(fake_x, x, reduction='mean')
    fake_x_data = torch.cat([fake_x, y], 1)
    fake_x_loss = net['P'](fake_x_data).mean()
    loss_x = -(1-alpha) * fake_x_loss
    loss_e = tau_D * mae_loss
    loss = loss_x + loss_e

    # Backward
    loss.backward()
    optimizerD.step()

    return loss, loss_x, loss_e, mae_loss, fake_x.data

def train_epoch(net, train_loader, val_loader, optimizer, scheduler, batch_size):
    epochs = 70
    best_psnr = 0
    # train 
    for epoch in range(epochs):
        net['D'].train()
        net['G'].train()
        net['P'].train()
        
        for i, data in enumerate(train_loader):
            im_gt, im_noisy = [x.cuda() for x in data]

            PL, Px, Pxg, Py, Pyg = train_step_P(net,im_gt, im_noisy, optimizer['P'])

            if (i+1) % 3 == 0:
                DL, Dx, DE, DAE, im_denoise = train_step_D(net, im_gt, im_noisy, optimizer['D'])
                GL, Gy, GMean, im_generate = train_step_G(net, im_gt, im_noisy, optimizer['G'])

                GErr = F.l1_loss(im_generate, im_gt, reduction='mean')
                TGErr = F.l1_loss(im_noisy, im_gt, reduction='mean')

                if (i+1) % 100 == 0:
                    template = '[Epoch:{:>2d}/{:<3d}] {:s}:{:0>5d}, PLx:{:>6.2f},'+\
                                         ' PLy:{:>6.2f}/{:4.2f}, DL:{:>6.2f}/{:.1e}, DAE:{:.2e}, '+\
                                                            'GL:{:>6.2f}/{:<5.2f}, GErr:{:.1e}/{:.1e}'
                    print(template.format(epoch+1, epochs, 'train', i+1,
                                     Pxg.item(), Py.item(), Pyg.item(), Dx.item(), DE.item(),
                                    DAE.item(), Gy.item(), GMean.item(), GErr.item(), TGErr.item()))
                        
        print('-'*100)

        # test
        net['D'].eval()
        psnr_per_epoch = ssim_per_epoch = 0
        
        for i, data in enumerate(val_loader):
            im_gt, im_noisy = [x.cuda() for x in data]
            # print(f"[S] {im_noisy.shape}")
            with torch.set_grad_enabled(False):
                im_denoise = im_noisy - net['D'](im_noisy)
            mae = 0
            mae_iter = F.l1_loss(im_denoise, im_gt)
            mae += mae_iter
            im_denoise.clamp_(0.0, 1.0)
            if im_gt.dim() == 3:
                im_gt = im_gt.unsqueeze(0)
            # print(f"[S] {im_denoise.shape}  {im_gt.shape}")
            psnr_iter = batch_PSNR(im_denoise, im_gt)
            psnr_per_epoch += psnr_iter
            ssim_iter = batch_SSIM(im_denoise, im_gt)
            ssim_per_epoch += ssim_iter
        
            if (i+1) % 50 == 0:
                log_str = '[Epoch:{:>2d}/{:<2d}] {:s}:{:0>3d}, mae={:.2e}, ' + 'psnr={:4.2f}, ssim={:5.4f}'
                print(log_str.format(epoch+1, epochs, 'val', i+1, mae_iter, psnr_iter, ssim_iter))
        
        psnr_per_epoch /= (i+1)
        ssim_per_epoch /= (i+1)
        mae /= (i+1)
        print('{:s}: mae={:.3e}, PSNR={:4.2f}, SSIM={:5.4f}'.format('val', mae, psnr_per_epoch, ssim_per_epoch))

        print('-'*100)

        # adjust the learning rate
        scheduler['D'].step()
        scheduler['G'].step()
        scheduler['P'].step()

        # 保存模型，如果当前PSNR更高
        if psnr_per_epoch > best_psnr:
            best_psnr = psnr_per_epoch
            torch.save(net['D'].state_dict(), 'best_model_D.pth')
            torch.save(net['G'].state_dict(), 'best_model_G.pth')
            torch.save(net['P'].state_dict(), 'best_model_P.pth')
            print(f"Saved best model with PSNR: {best_psnr:.4f}")

    # 最后保存最终的模型
    torch.save(net['D'].state_dict(), 'final_model_D.pth')
    torch.save(net['G'].state_dict(), 'final_model_G.pth')
    torch.save(net['P'].state_dict(), 'final_model_P.pth')
    print("Saved final model")
    
        
def main():
    # Params
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
    print("# of validating samples: %d\n" % len(val_set))

    # Build Model
    netD = UNetD(in_channels=3).cuda()
    netG = UNetG(in_channels=3).cuda()
    netP = DiscriminatorLinear(in_channels=6).cuda() # 每次输入一对RGB图像，共3+3=6 channels
    net = {'D':netD, 'G':netG, 'P':netP}

    # Optimizer
    optimizerD = torch.optim.Adam(netD.parameters(), lr = 1e-4)
    optimizerG = torch.optim.Adam(netG.parameters(), lr = 1e-4)
    optimizerP = torch.optim.Adam(netP.parameters(), lr = 2e-4)
    optimizer = {'D':optimizerD, 'G':optimizerG, 'P':optimizerP}

    # Scheduler
    milestones = [20, 30, 40, 50, 60, 65, 70]
    schedulerD = torch.optim.lr_scheduler.MultiStepLR(optimizerD, milestones=milestones, gamma=0.5)
    schedulerG = torch.optim.lr_scheduler.MultiStepLR(optimizerG, milestones=milestones, gamma=0.5)
    schedulerP = torch.optim.lr_scheduler.MultiStepLR(optimizerP, milestones=milestones, gamma=0.5)
    scheduler = {'D':schedulerD, 'G':schedulerG, 'P':schedulerP}

    global kernel
    kernel = get_gausskernel(kernel_size=5, channels=3)

    train_epoch(net, train_loader, val_set, optimizer, scheduler, batch_size)


def print_tensor_shape(frame, event, args):
    f_code = frame.f_code
    if not f_code.co_name == "forward" or not args.__class__.__name__ == "Tensor":
        return
    filename = f_code.co_filename
    # tensor_shape = [t.shape for t in args]
    if event == "call":
        print(f"[D] {args.shape}")
    elif event == "return":
        print(f"[D] {args.shape}")
    return



if __name__ == "__main__":
    # sys.setprofile(print_tensor_shape)
    main()
    # sys.setprofile(None)