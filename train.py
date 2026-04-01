import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchvision import transforms
from PIL import Image
from torch.utils.data import Subset
import os
import numpy as np

# --- 1. 模型架构定义 ---

class PPM(nn.Module):
    def __init__(self, in_dim, reduction_dim, bins):
        super(PPM, self).__init__()
        self.features = nn.ModuleList([
            nn.Sequential(
                nn.AdaptiveAvgPool2d(bin),
                nn.Conv2d(in_dim, reduction_dim, kernel_size=1, bias=False),
                nn.BatchNorm2d(reduction_dim),
                nn.ReLU(inplace=True)
            ) for bin in bins
        ])

    def forward(self, x):
        x_size = x.size()
        out = [x]
        for f in self.features:
            out.append(F.interpolate(f(x), x_size[2:], mode='bilinear', align_corners=True))
        return torch.cat(out, 1)

class DINO_PSPNet(nn.Module):
    def __init__(self, num_classes=21):
        super(DINO_PSPNet, self).__init__()
        # 自动使用本地缓存
        self.backbone = torch.hub.load(
            repo_or_dir='/root/autodl-tmp/Documents/dinov3',
            model='dinov3_vits16',
            weights='/root/autodl-tmp/dinov3_vits16_pretrain_lvd1689m-08c60483.pth',
            source='local'
        )
        self.backbone.train() 
        
        fea_dim = 384 # ViT-S 维度
        self.ppm = PPM(fea_dim, int(fea_dim/4), (1, 2, 3, 6))
        self.cls = nn.Sequential(
            nn.Conv2d(fea_dim * 2, 512, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=0.1),
            nn.Conv2d(512, num_classes, kernel_size=1)
        )

    def forward(self, x):
        input_shape = x.shape[2:]
        # 提取最后一层特征 [B, 256, 384]
        features = self.backbone.get_intermediate_layers(x, n=1)[0]
        B, N, C = features.shape
        grid = int(N**0.5)
        x = features.reshape(B, grid, grid, C).permute(0, 3, 1, 2) # [B, 384, 16, 16]
        
        x = self.ppm(x)
        x = self.cls(x)
        return F.interpolate(x, size=input_shape, mode='bilinear', align_corners=True)

class VOCSegDataset(Dataset):
    def __init__(self, root, image_set='train',augment=True):
        self.root = root
        self.augment = augment
        image_dir = os.path.join(root, 'JPEGImages')
        mask_dir  = os.path.join(root, 'SegmentationClass')
        split_f   = os.path.join(root, 'ImageSets/Segmentation', f'{image_set}.txt')
        with open(split_f, "r") as f:
            file_names = [x.strip() for x in f.readlines()]
        self.images = [os.path.join(image_dir, x + ".jpg") for x in file_names]
        self.masks  = [os.path.join(mask_dir,  x + ".png") for x in file_names]

    def __getitem__(self, index):
        img    = Image.open(self.images[index]).convert("RGB")
        target = Image.open(self.masks[index])

        if self.augment:
            # 随机增强，只有训练集走这里
            scale = np.random.uniform(0.5, 2.0)
            new_h, new_w = int(480*scale), int(480*scale)
            img    = transforms.Resize((new_h, new_w))(img)
            target = transforms.Resize((new_h, new_w),
                         interpolation=transforms.InterpolationMode.NEAREST)(target)
            pad_h = max(0, 480 - new_h)
            pad_w = max(0, 480 - new_w)
            if pad_h > 0 or pad_w > 0:
                img    = transforms.Pad((0, 0, pad_w, pad_h), fill=0)(img)
                target = transforms.Pad((0, 0, pad_w, pad_h), fill=255)(target)
            i, j, h, w = transforms.RandomCrop.get_params(img, output_size=(480, 480))
            img    = transforms.functional.crop(img,    i, j, h, w)
            target = transforms.functional.crop(target, i, j, h, w)
            if np.random.random() > 0.5:
                img    = transforms.functional.hflip(img)
                target = transforms.functional.hflip(target)
            angle  = np.random.uniform(-10, 10)
            img    = transforms.functional.rotate(img, angle)
            target = transforms.functional.rotate(target, angle,
                         interpolation=transforms.InterpolationMode.NEAREST, fill=255)
            if np.random.random() > 0.5:
                img = transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.0))(img)
        else:
            # 验证和测试只做固定 resize
            img    = transforms.Resize((480, 480))(img)
            target = transforms.Resize((480, 480),
                         interpolation=transforms.InterpolationMode.NEAREST)(target)

        img    = transforms.ToTensor()(img)
        img    = transforms.Normalize([0.485, 0.456, 0.406],
                                       [0.229, 0.224, 0.225])(img)
        target = torch.from_numpy(np.array(target)).long()
        return img, target

    def __len__(self):
        return len(self.images)

def compute_miou(model, loader, num_classes, device):
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    with torch.no_grad():
        for imgs, labels in loader:
            imgs = imgs.to(device)
            preds = model(imgs).argmax(dim=1).cpu().numpy()
            labels = labels.numpy()
            mask = (labels >= 0) & (labels < num_classes)
            combined = num_classes * labels[mask] + preds[mask]
            confusion += np.bincount(combined, minlength=num_classes**2).reshape(num_classes, num_classes)
    
    VOC_CLASSES = ['background','aeroplane','bicycle','bird','boat','bottle',
                   'bus','car','cat','chair','cow','diningtable','dog','horse',
                   'motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    
    iou_list = []
    for i in range(num_classes):
        tp = confusion[i, i]
        fp = confusion[:, i].sum() - tp
        fn = confusion[i, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            iou = tp / denom
            iou_list.append(iou)
            print(f"  {VOC_CLASSES[i]:15s}: {iou:.4f}")
        else:
            print(f"  {VOC_CLASSES[i]:15s}: N/A (该类不存在)")
    
    miou = float(np.mean(iou_list))
    print(f"\n  mIoU: {miou:.4f}")
    return miou

# --- 3. 训练主函数 ---

def train():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"✅ 使用设备: {device}")
    
    model = DINO_PSPNet(num_classes=21).to(device)
    
    voc_root = "/root/autodl-tmp/Datasets/VOC2012"

    base    = VOCSegDataset(voc_root, image_set='train', augment=False)
    total   = len(base)
    indices = list(range(total))
    np.random.seed(42)
    np.random.shuffle(indices)
    val_size  = int(total * 0.2)
    train_idx = indices[val_size:]
    val_idx   = indices[:val_size]
    
    # 同样的索引，不同的增强设置
    train_data = Subset(VOCSegDataset(voc_root, image_set='train', augment=True),  train_idx)
    val_data   = Subset(VOCSegDataset(voc_root, image_set='train', augment=False), val_idx)

    train_loader = DataLoader(train_data,   batch_size=16, shuffle=True,  num_workers=4)
    val_loader   = DataLoader(val_data,     batch_size=16, shuffle=False, num_workers=4)

    max_iter = 30000
    
    optimizer = torch.optim.AdamW([
    {'params': model.backbone.parameters(), 'lr': 1e-6},   # backbone 更小
    {'params': list(model.ppm.parameters()) + list(model.cls.parameters()), 'lr': 1e-4} ], weight_decay=1e-4) # decoder 正常

    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda iteration: (1 - iteration / max_iter) ** 0.9
    )
    
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    model.train()
    print(" 开始训练...")

    iteration = 200;
    current_iter = 0
    best_miou = 0.0
    for epoch in range(iteration):
        for i, (imgs, masks) in enumerate(train_loader):
            imgs, masks = imgs.to(device), masks.to(device)
            
            optimizer.zero_grad()
            outputs = model(imgs)
            loss = criterion(outputs, masks)
            loss.backward()
            optimizer.step()
            scheduler.step()
            current_iter += 1
            
            if i % 10 == 0:
                print(f"Epoch [{epoch}] Batch [{i}] Loss: {loss.item():.4f}")
        print("==========================================")
            
        model.eval()
        miou = compute_miou(model, val_loader, 21, device)
        print(f"Epoch [{epoch}] mIoU: {miou:.4f}")
        if miou > best_miou:
            best_miou = miou
            torch.save(model.state_dict(), 'best_model.pth')
        model.train()

if __name__ == "__main__":
    train()