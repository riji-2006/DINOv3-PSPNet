import torch
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import matplotlib.pyplot as plt
from train import DINO_PSPNet  
import glob
import os

def decode_segmap(label_mask, plot=False):
    """将类别索引映射回 VOC 官方彩色标签"""
    label_colors = np.array([
        (0, 0, 0),  # 0=background
        (128, 0, 0), (0, 128, 0), (128, 128, 0), (0, 0, 128), (128, 0, 128),
        (0, 128, 128), (128, 128, 128), (64, 0, 0), (192, 0, 0), (64, 128, 0),
        (192, 128, 0), (64, 0, 128), (192, 0, 128), (64, 128, 128), (192, 128, 128),
        (0, 64, 0), (128, 64, 0), (0, 192, 0), (128, 192, 0), (0, 64, 128)
    ])
    r = label_mask.copy()
    g = label_mask.copy()
    b = label_mask.copy()
    for l in range(0, 21):
        r[label_mask == l] = label_colors[l, 0]
        g[label_mask == l] = label_colors[l, 1]
        b[label_mask == l] = label_colors[l, 2]
    rgb = np.stack([r, g, b], axis=2)
    return rgb

def load_images():
    images = []
    root = "/root/autodl-tmp/Datasets/VOC2012/"
    img_path = "/root/autodl-tmp/Datasets/VOC2012/JPEGImages/"
    if not os.path.exists(img_path):
        print(f"文件路径不存在：{img_path}")
        exit(-1)
    
    split_val = os.path.join(root,'ImageSets/Segmentation','val.txt')
    with open(split_val,'r') as f:
        filenames = [x.strip() for x in f.readlines()]
        
    for imgname in filenames:
        images.extend(glob.glob(os.path.join(img_path,imgname+".jpg")))
    
    print(f"找到{len(images)}张图片")
    return images

def save_images(raw_image,pred):
    filename = os.path.basename(raw_image)
    label_filename = filename.replace('.jpg', '.png')
    
    gt_path = "/root/autodl-tmp/Datasets/VOC2012/SegmentationClass/"
    gt_name = os.path.join(gt_path,label_filename)

    if not os.path.exists(gt_name):
        print(f"警告：找不到标签文件 {gt_name}")
        return 
    
    gt_mask = Image.open(gt_name)
    
    gt_resized = gt_mask.resize((480, 480), resample=Image.NEAREST)
    gt_rgb = decode_segmap(np.array(gt_resized))

    rgb_mask = decode_segmap(pred)

    raw_image_display = Image.open(raw_image).convert("RGB")
    
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    axes[0].imshow(raw_image_display.resize((480, 480)))
    axes[0].set_title("Original Image")
    axes[0].axis("off")
    
    axes[1].imshow(rgb_mask)
    axes[1].set_title("DINO + PSPNet Prediction")
    axes[1].axis("off")

    axes[2].imshow(gt_rgb)
    axes[2].set_title("right Image")
    axes[2].axis("off")

    save_path = "/root/autodl-tmp/Documents/results/"
    save_full = os.path.join(save_path,label_filename)
    plt.savefig(save_full)
    plt.close()  # 加这一行
    print(f" 预测对比图已保存至: {save_full}")


def evaluate_from_preds(confusion):
    """根据混淆矩阵计算并保存评估结果"""
    VOC_CLASSES = ['background','aeroplane','bicycle','bird','boat','bottle',
                   'bus','car','cat','chair','cow','diningtable','dog','horse',
                   'motorbike','person','pottedplant','sheep','sofa','train','tvmonitor']
    print(f"sofa(18) tp={confusion[18,18]}, row_sum={confusion[18,:].sum()}, col_sum={confusion[:,18].sum()}")
    print(f"horse(13) tp={confusion[13,13]}, row_sum={confusion[13,:].sum()}, col_sum={confusion[:,13].sum()}")
    lines    = []
    iou_list = []
    for i in range(21):
        tp    = confusion[i, i]
        fp    = confusion[:, i].sum() - tp
        fn    = confusion[i, :].sum() - tp
        denom = tp + fp + fn
        if denom > 0:
            iou = tp / denom
            iou_list.append(iou)
            line = f"  {VOC_CLASSES[i]:15s}: {iou:.4f}"
        else:
            line = f"  {VOC_CLASSES[i]:15s}: N/A"
        print(line)
        lines.append(line)

    miou    = float(np.mean(iou_list))
    summary = f"\n  mIoU: {miou:.4f}"
    print(summary)
    lines.append(summary)

    txt_path = "/root/autodl-tmp/Documents/results/evaluation.txt"
    os.makedirs(os.path.dirname(txt_path), exist_ok=True)
    print(f"准备写入的内容: {lines[:3]}") 
    with open(txt_path, 'w') as f:
        f.write("===== DINOv3 + PSPNet 测试集评估结果 =====\n")
        f.write('\n'.join(lines))
    print(f"\n评估结果已保存至: {txt_path}")
    return miou

def predict(model, raw_image, transform, device):
    input_tensor = transform(raw_image).unsqueeze(0).to(device)

    # 3. 推理
    with torch.no_grad():
        output = model(input_tensor)
        pred = torch.argmax(output, dim=1).squeeze(0).cpu().numpy()
    return pred

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    model = DINO_PSPNet(num_classes=21).to(device)
    model.load_state_dict(torch.load("best_model.pth"))
    model.eval()
    print("成功加载训练的权重")
    
    transform = transforms.Compose([
        transforms.Resize((480, 480)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    
    raw_images = load_images()

    confusion = np.zeros((21, 21), dtype=np.int64)
    for img_file in raw_images:
        raw_image = Image.open(img_file).convert("RGB")
        pred = predict(model, raw_image, transform, device)
        save_images(img_file,pred)
        # 更新混淆矩阵
        label_filename = os.path.basename(img_file).replace('.jpg', '.png')
        gt_name = os.path.join("/root/autodl-tmp/Datasets/VOC2012/SegmentationClass/", label_filename)
        if not os.path.exists(gt_name):
            continue
        gt = np.array(Image.open(gt_name).resize((480, 480), resample=Image.NEAREST)).astype(np.int64)
            
        mask     = (gt >= 0) & (gt < 21)
        combined = 21 * gt[mask] + pred[mask]
        confusion += np.bincount(combined, minlength=21**2).reshape(21, 21)
        
    evaluate_from_preds(confusion) 

if __name__ == "__main__":
    main()