import os,glob,math
import matplotlib.pyplot as plt
import numpy as np
import segmentation_models_pytorch as smp
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchmetrics
from joblib import Parallel, delayed
from osgeo import gdal, gdal_array, ogr
from pytorch_lightning import LightningModule, Trainer
from pytorch_lightning.callbacks import Callback
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader, Dataset, random_split
from matplotlib.colors import ListedColormap

# Import ListedColormap

def PreProcessing_BW(filename, label):
    InputImage = filename
    base_name = os.path.splitext(os.path.basename(filename))[0]
    output_dir = rf"/home/mahmoud/Preproces_Wetland_v2/{base_name}"
    Tensor_name = f"{base_name}_Tensor"
    filename = InputImage
    base_name = os.path.splitext(os.path.basename(filename))[0]
    
    tensor_output_dir = os.path.join(output_dir, Tensor_name)
    
    if os.path.exists(tensor_output_dir):
        print(f"Tensor folder already exists for {Tensor_name}, skipping processing.")
        return
    
    inRasterPath = InputImage
    data = gdal.Open(inRasterPath)
    geo_transform = data.GetGeoTransform()
    x_min = geo_transform[0]
    y_max = geo_transform[3]
    x_max = x_min + geo_transform[1] * data.RasterXSize
    y_min = y_max + geo_transform[5] * data.RasterYSize
    y_l = gdal.Open(label)
    nodatay = abs(y_l.GetRasterBand(1).GetNoDataValue())
    print(nodatay)
    
    pixel_width = geo_transform[1]
    vrt_options = gdal.BuildVRTOptions(outputBounds=(x_min, y_min, x_max, y_max))
    vrt_options2 = gdal.BuildVRTOptions(outputBounds=(x_min, y_min, x_max, y_max), xRes=0.5, yRes=0.5)
    
    Ydal1 = gdal.BuildVRT(rf"{base_name}.vrt", y_l, options=vrt_options)
    Ydal2 = gdal.BuildVRT(rf"{base_name}.vrt", y_l, options=vrt_options2)
    Xdal2 = gdal.BuildVRT(rf"{base_name}.vrt", data, options=vrt_options2)
    
    Y1 = Ydal1.ReadAsArray()
    if np.all(Y1 != 1):
        print(f"No valid patches in image {InputImage}, skipping...")
        return
    
    Y2 = Ydal2.ReadAsArray()
    print(np.unique(Y2))    
    X1 = data.ReadAsArray()
    X2 = Xdal2.ReadAsArray()

    tensor_x1 = torch.Tensor(X1)
    tensor_y1 = torch.Tensor(Y1).to(torch.uint8)
    tensor_x2 = torch.Tensor(X2)
    tensor_y2 = torch.Tensor(Y2).to(torch.uint8)
    
    ph1 = (((X1.shape[1] // 512)) * 512) - X1.shape[1]
    pw1 = (((X1.shape[2] // 512)) * 512) - X1.shape[2]
    ph2 = (((X2.shape[2] // 512)) * 512) - X2.shape[2]
    pw2 = (((X2.shape[2] // 512)) * 512) - X2.shape[2]
    
    tensor_x1 = torch.nn.functional.pad(tensor_x1, (0, pw1, 0, ph1), mode="constant")
    tensor_y1 = torch.nn.functional.pad(tensor_y1, (0, pw1, 0, ph1), mode="constant")
    tensor_x2 = torch.nn.functional.pad(tensor_x2, (0, pw2, 0, ph2), mode="constant")
    tensor_y2 = torch.nn.functional.pad(tensor_y2, (0, pw2, 0, ph2), mode="constant")
    
    print(tensor_x1.shape)
    print(tensor_y1.shape)
    print(tensor_x2.shape)
    print(tensor_y2.shape)
    
    patches_x1 = tensor_x1.unfold(1, 512, 256).unfold(2, 512, 256)
    patches_x1 = torch.reshape(patches_x1, (3, patches_x1.size(1) * patches_x1.size(2), 512, 512)).permute(1, 0, 2, 3)

    patches_y1 = tensor_y1.unfold(0, 512, 256).unfold(1, 512, 256)
    patches_y1 = torch.reshape(patches_y1, (patches_y1.size(0) * patches_y1.size(1), 512, 512))

    patches_x2 = tensor_x2.unfold(1, 512, 256).unfold(2, 512, 256)
    patches_x2 = torch.reshape(patches_x2, (3, patches_x2.size(1) * patches_x2.size(2), 512, 512)).permute(1, 0, 2, 3)

    patches_y2 = tensor_y2.unfold(0, 512, 256).unfold(1, 512, 256)
    patches_y2 = torch.reshape(patches_y2, (patches_y2.size(0) * patches_y2.size(1), 512, 512))

    patches_x = torch.cat((patches_x1, patches_x2), dim=0)
    patches_y = torch.cat((patches_y1, patches_y2), dim=0)

    mask = (patches_x[:, 0] == 0).any(dim=1).any(dim=1)
    patches_x = patches_x[~mask]
    patches_y = patches_y[~mask]
    print("Mask 1 done")
    
    mask = (patches_y == nodatay).any(dim=1).any(dim=1)
    patches_x = patches_x[~mask]
    patches_y = patches_y[~mask]
    print("Mask 2 done")
    
    mask = (patches_y != 1).all(dim=1).all(dim=1)
    patches_x = patches_x[~mask]
    patches_y = patches_y[~mask]
    
    print(torch.unique(patches_y))
    
    if patches_x.numel() == 0:
        print(f"No valid patches in image {InputImage}, skipping...")
        return
    
    patches_x, patches_y = D_Augm(patches_x, patches_y)
    
    print('Unique values are', torch.unique(patches_y))
    os.makedirs(os.path.join(output_dir, Tensor_name), exist_ok=True)
    
    njobs = math.ceil(len(patches_x) / 32)
    Parallel(n_jobs=njobs, prefer="threads")(
        delayed(write_tensor)(patches_x, patches_y, output_dir, base_name, i)
        for i in range(len(patches_x))
    )
    return



def D_Augm(patches_x, patches_y):
    augmented_x = []
    augmented_y = []
    for i in range(patches_x.shape[0]):
        # append original patch
        augmented_x.append(patches_x[i])
        augmented_y.append(patches_y[i])

        # rotate by 45 degrees
        x_rot45 = torch.rot90(patches_x[i], k=1, dims=(1, 2))
        y_rot45 = torch.rot90(patches_y[i], k=1, dims=(0, 1))
        augmented_x.append(x_rot45)
        augmented_y.append(y_rot45)

        # rotate by 90 degrees
        x_rot90 = torch.rot90(patches_x[i], k=2, dims=(1, 2))
        y_rot90 = torch.rot90(patches_y[i], k=2, dims=(0, 1))
        augmented_x.append(x_rot90)
        augmented_y.append(y_rot90)

        # rotate by 135 degrees
        x_rot135 = torch.rot90(patches_x[i], k=3, dims=(1, 2))
        y_rot135 = torch.rot90(patches_y[i], k=3, dims=(0, 1))
        augmented_x.append(x_rot135)
        augmented_y.append(y_rot135)

        # flip horizontally
        x_flipped_h = torch.flip(patches_x[i], dims=(2,))
        y_flipped_h = torch.flip(patches_y[i], dims=(1,))
        augmented_x.append(x_flipped_h)
        augmented_y.append(y_flipped_h)

        # flip vertically
        x_flipped_v = torch.flip(patches_x[i], dims=(1,))
        y_flipped_v = torch.flip(patches_y[i], dims=(0,))
        augmented_x.append(x_flipped_v)
        augmented_y.append(y_flipped_v)

    augmented_x = torch.stack(augmented_x)
    augmented_y = torch.stack(augmented_y)
    return augmented_x, augmented_y


class MyCall(Callback):
    def __init__(self):
        super().__init__()

    def on_train_start(self, trainer, pl_module):
        print("Starting to train")

    def on_train_end(self, trainer, pl_module):
        print("Training Done")

class UNET_BW(LightningModule):
    def __init__(self, num_classes, learning_rate, dataset, batch_size, endcoder):
        super().__init__()
        self.num_classes = num_classes
        self.encoder = endcoder
        self.learning_rate = learning_rate
        self.batch_size = batch_size
        self.dataset = dataset
        self.crit = nn.BCELoss ()
        # Update metrics to binary classification metrics
        self.train_acc = torchmetrics.Accuracy(task="binary")
        self.train_F1 = torchmetrics.F1Score(task="binary")
        self.IoU = torchmetrics.JaccardIndex(task="binary")
        # Split the dataset into training and validation sets
        total_size = len(self.dataset)
        train_size = int(0.7 * total_size)  # 70% for training
        validate_size = int(0.15 * total_size)  # 15% for validation
        test_size = total_size - train_size - validate_size  # Remaining for testing
        self.train_dataset, self.validate_dataset, self.test_dataset = random_split(
            self.dataset, [train_size, validate_size, test_size]
        )
        # Model Structure
        self.model = smp.Unet(
            encoder_name=f"{self.encoder}",
            encoder_weights='imagenet',
            classes=num_classes,
            activation='sigmoid'
            
        )
        # Freeze the weights of the encoder
        for param in self.model.encoder.parameters():
            param.requires_grad = False
        self.save_hyperparameters()

    def forward(self, x):
        return self.model(x)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.learning_rate)
        return optimizer
        # Loaders

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset, batch_size=self.batch_size, num_workers=12,persistent_workers=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.validate_dataset, batch_size=self.batch_size, num_workers=12,persistent_workers=True
        )

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=self.batch_size, num_workers=12,persistent_workers=True
        )

    # Common Step
    def _common_step(self, batch, batch_idx):
        x, y = batch
        x = x.float() / 255.0  # Normalize the data by dividing by 255
        y = y.float()
        y_hat = self.forward(x).squeeze(1)
        loss = self.crit(y_hat, y)
        return loss, y_hat, y

    def training_step(self, batch, batch_idx):
        x, y = batch
        loss, y_hat, y = self._common_step(batch, batch_idx)
        preds = y_hat
        t_acc = self.train_acc(preds, y)
        t_f1 = self.train_F1(preds, y)
        t_IoU = self.IoU(preds, y)
        # Logging
        self.log_dict(
            {
                "train_loss": loss,
                "train_accuracy": t_acc,
                "train_f1_score": t_f1,
                "IoU": t_IoU,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return {"loss": loss, "x": x, "y": y}

    def validation_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {"valid_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def test_step(self, batch, batch_idx):
        loss, y_hat, y = self._common_step(batch, batch_idx)
        self.log_dict(
            {"test_loss": loss},
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            sync_dist=True,
        )
        return loss

    def predict_step(self, batch, batch_idx, dataloader_idx=0):
        x = batch
        x = x / 255.0
        x = x.float()
        return self(batch)
    




def write_tensor(patches_x, patches_y, output_dir, base_name, i):
    # Save each tensor pair as a separate file
    tensor_name = f"{base_name}_{i}.pt"  # unique name for each tensor pair
    tensor_path = os.path.join(output_dir, rf"{base_name}_Tensor", tensor_name)
    torch.save(
        {"tensor_x": patches_x[i].clone(), "tensor_y": patches_y[i].clone()},
        tensor_path,
    )


class CustomDataset(Dataset):
    def __init__(self, data_root):
        self.data_paths = glob.glob(f"{data_root}/**/*.pt", recursive=True)

    def __len__(self):
        return len(self.data_paths)

    def __getitem__(self, idx):
        data_path = self.data_paths[idx]
        data = torch.load(data_path)

        x = data['tensor_x'].float()  # data loaded from file is already a tensor
        # Scale the data to the range [0, 1]
        y = data['tensor_y'].long()  # data loaded from file is already a tensor

        return x,y



def Plotting_Dataset(dataset):
    # Randomly select 16 indices from the dataset
    num_samples_to_display = 16
    selected_indices = np.random.choice(len(dataset), num_samples_to_display, replace=False)

    # Define the color map for the labels
    label_to_color = {
        0: [0.5, 0.5, 0.5],    # Grey
        1: [0, 1, 0],          # Green
        2: [0, 0, 1],          # Blue
        3: [1, 0, 0],          # Red
        4: [1, 0.5, 0],        # Orange
        5: [1, 1, 0],          # Yellow
    }

    # Plot the patches overlaid with colors
    fig, axes = plt.subplots(4, 4, figsize=(10, 10))

    displayed_count = 0

    for idx in selected_indices:
        data_item = dataset[idx]
        if data_item is not None:
            x_patch = data_item[0]  # First layer of x
            y_labels = data_item[1]

            overlay = np.zeros((*x_patch.shape, 3), dtype=np.float32)

            for row in range(y_labels.shape[0]):
                for col in range(y_labels.shape[1]):
                    y_label = y_labels[row, col]
                    y_color = label_to_color[y_label.item()]
                    overlay[row, col] = y_color

            ax = axes[displayed_count // 4, displayed_count % 4]
            ax.imshow(x_patch, cmap='gray')  # Plot the grayscale patch
            ax.imshow(overlay, alpha=0.5)  # Overlay with colors
            ax.axis('off')

            displayed_count += 1

        if displayed_count >= num_samples_to_display:
            break

    plt.tight_layout()
    plt.show()
    return


import os
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from pytorch_lightning import Trainer
from osgeo import gdal, gdal_array
import numpy as np

def Prediction(args):
    Image, model_chkpt, opdir, device_queue = args
    device_id = device_queue.get()  # Get an available device
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device_id)
    
    print(f'Using CUDA device {device_id} for image {Image}')
    
    try:
        base_name = os.path.splitext(os.path.basename(Image))[0]
        if not os.path.exists(opdir):
            os.makedirs(opdir)
        output_file = os.path.join(opdir, f'{base_name}_OP.tif')
        
        if os.path.exists(output_file):
            print(f"Output file {output_file} already exists. Skipping...", flush=True)
            return
        
        # Enable TF32 for A100
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
        
        # Load the U-Net model checkpoint
        model = UNET_BW.load_from_checkpoint(rf"{model_chkpt}")

        # Open the image
        INP = gdal.Open(Image)
        gt = INP.GetGeoTransform()
        pr = INP.GetProjection()
        array = INP.ReadAsArray()

        # If single-layer array, replicate it to 3 channels
        if array.ndim == 2:
            array = np.stack([array] * 3)

        print(array.shape)
        
        # Padding to avoid missing pixels
        ph = ((array.shape[0] // 512) + 1) * 512 - array.shape[0]
        pw = ((array.shape[1] // 512) + 1) * 512 - array.shape[1]
        
        tensor_x = torch.tensor(array).float()
        tensor_x = torch.nn.functional.pad(tensor_x, (0, ph, 0, pw), mode='reflect')
        C, H, W = tensor_x.shape
        
        # Unfold the image into patches
        patch_size_256 = 512
        stride_256 = 256
        patches_256 = tensor_x.unfold(1, patch_size_256, stride_256).unfold(2, patch_size_256, stride_256)
        patches_256 = patches_256.reshape(3, patches_256.size(1) * patches_256.size(2), patch_size_256, patch_size_256).permute(1, 0, 2, 3)
        patches_256 = patches_256 / 255.0
        
        # Rotate patches by 90, 180, and 270 degrees
        rotated_patches_90 = torch.rot90(patches_256, k=1, dims=(2, 3))
        rotated_patches_180 = torch.rot90(patches_256, k=2, dims=(2, 3))
        rotated_patches_270 = torch.rot90(patches_256, k=3, dims=(2, 3))
        
        # Concatenate all rotations
        all_patches = torch.cat((patches_256, rotated_patches_90, rotated_patches_180, rotated_patches_270), dim=0)
        
        # Create DataLoader for the patches
        dl_all = DataLoader(all_patches, batch_size=160, shuffle=False, num_workers=12)
        
        # Define the trainer for prediction
        trainer = Trainer(devices='1', logger=False, precision='16-mixed')
        
        # Make predictions
        with torch.no_grad():
            predictions_all = trainer.predict(model, dl_all)
        
        y_all = torch.cat(predictions_all)
        
        # Split predictions back into individual rotations
        y = y_all[:len(patches_256)]
        y_90_rot = y_all[len(patches_256):2 * len(patches_256)]
        y_180_rot = y_all[2 * len(patches_256):3 * len(patches_256)]
        y_270_rot = y_all[3 * len(patches_256):]
        
        # Rotate predictions back to original orientations
        y_90 = torch.rot90(y_90_rot, k=-1, dims=(2, 3))
        y_180 = torch.rot90(y_180_rot, k=-2, dims=(2, 3))
        y_270 = torch.rot90(y_270_rot, k=-3, dims=(2, 3))
        
        # Average the predictions
        Y2_avg = (y_90 + y_180 + y_270 + y) / 4.0
        ones_tensor = torch.ones_like(Y2_avg)

        B, C, H, W = 1, 1, tensor_x.shape[1], tensor_x.shape[2]
        Y2_avg = Y2_avg.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous().view(B, C, -1, patch_size_256 * patch_size_256).permute(0, 1, 3, 2).contiguous().view(C, 1 * patch_size_256 * patch_size_256, -1)
        output_Y2 = F.fold(Y2_avg, output_size=(H, W), kernel_size=patch_size_256, stride=stride_256).squeeze(1)
        Y2_ones = ones_tensor.unsqueeze(0).permute(0, 2, 1, 3, 4).contiguous().view(B, C, -1, patch_size_256 * patch_size_256).permute(0, 1, 3, 2).contiguous().view(C, 1 * patch_size_256 * patch_size_256, -1)
        Y2_ones = F.fold(Y2_ones, output_size=(H, W), kernel_size=patch_size_256, stride=stride_256).squeeze(1)
        output_avg = output_Y2 / Y2_ones
        
        print(output_avg.shape)
        print("Max probability:", torch.max(output_avg))
        
        # Binary thresholding for the final output
        Y_F = (output_avg > 0.5) * 6
        Y_F = torch.squeeze(Y_F, dim=1)
        
        original_H, original_W = array.shape[1], array.shape[2]
        Y_F = Y_F[:, :original_H, :original_W]
        
        Y_F_NP = Y_F.cpu().numpy().astype(np.uint8)

        # Save the output as a georeferenced TIFF
        options = ["COMPRESS=LZW"]
        Y = gdal_array.OpenArray(Y_F_NP)
        gdal_array.CopyDatasetInfo(INP, Y)
        gdal.Translate(rf'{opdir}/{base_name}_OP.tif', Y, creationOptions=options)
        
        Y = None
        del tensor_x, patches_256, all_patches, rotated_patches_90, rotated_patches_180, rotated_patches_270, predictions_all, y_all
        torch.cuda.empty_cache()

    finally:
        device_queue.put(device_id)  # Return the device to the queue


def plot_predictions(model, dataloader, num_images=4):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    model.eval()  # Set the model to evaluation mode
    with torch.no_grad():
        batch = next(iter(dataloader))
        x, y = batch
        x = x.float() / 255.0
        x = x.to(device)
        y = y.to(device)
        y_hat = model(x)

        # Convert predictions to class indices
        y_hat_indices = (y_hat > 0.5)
        y_hat_indices = torch.squeeze(y_hat_indices, dim=1)

        # Custom colormap: 0 -> gray (background), 1 -> pink (label)
        cmap = ListedColormap(['gray', 'pink'])

        # Plotting
        fig, axes = plt.subplots(2, num_images, figsize=(12, 8))

        # First row: Ground Truth
        for i in range(min(num_images, x.size(0))):
            axes[0, i].imshow(x[i, 0].cpu(), cmap='gray')  # Original image in grayscale
            axes[0, i].imshow(y[i].cpu(), cmap=cmap, vmin=0, vmax=1, alpha=0.4)  # Ground truth with 40% transparency
            axes[0, i].axis('off')

        # Add centered title for the first row (Ground Truth)
        fig.text(0.5, 0.88, 'Ground Truth', ha='center', fontsize=18, weight='bold')

        # Second row: Predicted Images
        for i in range(min(num_images, x.size(0))):
            axes[1, i].imshow(x[i, 0].cpu(), cmap='gray')  # Original image in grayscale
            axes[1, i].imshow(y_hat_indices[i].cpu(), cmap=cmap, vmin=0, vmax=1, alpha=0.4)  # Prediction with 40% transparency
            axes[1, i].axis('off')

        # Add centered title for the second row (Predicted Images)
        fig.text(0.5, 0.45, 'Predicted Images', ha='center', fontsize=18, weight='bold')

        plt.tight_layout(rect=[0, 0, 1, 0.93])  # Adjust layout to fit titles
        plt.show()