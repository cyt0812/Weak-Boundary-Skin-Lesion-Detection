from torch.utils.data import DataLoader

from dataset import ISICDataset


image_dir = (
    r"D:\GraduationProject\Skin-Lesion-Detection"
    r"\datasets\ISIC2018\segmentation"
    r"\ISIC2018_Task1-2_Training_Input"
)

mask_dir = (
    r"D:\GraduationProject\Skin-Lesion-Detection"
    r"\datasets\ISIC2018\segmentation"
    r"\ISIC2018_Task1_Training_GroundTruth"
)


dataset = ISICDataset(
    image_dir=image_dir,
    mask_dir=mask_dir,
    image_size=(256, 256)
)


dataloader = DataLoader(
    dataset,
    batch_size=4,
    shuffle=True,
    num_workers=0
)


images, masks = next(iter(dataloader))


print("Dataset 总数量：", len(dataset))
print("Batch 图片 shape：", images.shape)
print("Batch Mask shape：", masks.shape)
print("Image 数据类型：", images.dtype)
print("Mask 数据类型：", masks.dtype)