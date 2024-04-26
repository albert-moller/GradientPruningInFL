from skimage.metrics import peak_signal_noise_ratio as psnr
from torchmetrics.image import StructuralSimilarityIndexMeasure as ssim
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.transforms import ToTensor 
import torch
from torchvision import transforms
from PIL import Image
import cv2

class Metrics:

    def __init__(self, ground_truth_imgs, reconstructed_imgs):
        self.ground_truth_imgs = ground_truth_imgs
        self.reconstructed_imgs = reconstructed_imgs
        self.to_tensor = ToTensor()

    def to_tensor_and_unsqueeze(self, image):
        if not isinstance(image, Image.Image):
            if isinstance(image, np.ndarray):
                image = Image.fromarray(image)
            elif isinstance(image, torch.Tensor):
                return image.unsqueeze(0) if image.ndim == 3 else image
        return transforms.ToTensor()(image).unsqueeze(0)

    def _ensure_tensor(self, img):
        if not isinstance(img, torch.Tensor):
            img = self.to_tensor(img)
        return img
    
    def _tensor_to_np(self, tensor):
        return tensor.cpu().detach().numpy()
    
    def _to_numpy(self, image):
        if isinstance(image, torch.Tensor):
            image = image.cpu().numpy()
        elif not isinstance(image, np.ndarray):
            image = np.array(image)
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:
            image = image.transpose(1, 2, 0)
        return image
        
    def compute_mse(self):
        mse_vals = []
        for gt_img, rec_img in zip(self.ground_truth_imgs, self.reconstructed_imgs):
            gt_tensor = self._ensure_tensor(gt_img)
            rec_tensor = self._ensure_tensor(rec_img)
            mse_vals.append(torch.mean((gt_tensor - rec_tensor) ** 2).item())
        average_mse = np.mean(mse_vals)
        return average_mse

    def compute_ssim(self):
        ssim_scores = []
        ssim_ = ssim()

        for gt_img, rec_img in zip(self.ground_truth_imgs, self.reconstructed_imgs):
            gt_tensor = self.to_tensor_and_unsqueeze(gt_img)
            rec_tensor = self.to_tensor_and_unsqueeze(rec_img)

            if rec_tensor.dtype != gt_tensor.dtype:
                gt_tensor = gt_tensor.to(rec_tensor.dtype)

            ssim_score = ssim_(rec_tensor, gt_tensor) 
            ssim_scores.append(ssim_score.item())

        average_ssim = np.mean(ssim_scores) if ssim_scores else float('nan')
        return average_ssim
   
    def compute_psnr(self):
        psnr_vals = []
        for gt_img, rec_img in zip(self.ground_truth_imgs, self.reconstructed_imgs):
            gt_tensor = self._ensure_tensor(gt_img)
            rec_tensor = self._ensure_tensor(rec_img)
            gt_np = self._tensor_to_np(gt_tensor)
            rec_np = self._tensor_to_np(rec_tensor)
            psnr_val = psnr(gt_np, rec_np, data_range=rec_np.max() - rec_np.min())
            psnr_vals.append(psnr_val)
        
        psnr_vals_array = np.array(psnr_vals)
        psnr_vals_filtered = psnr_vals_array[np.isfinite(psnr_vals_array)]
        average_psnr = np.mean(psnr_vals_filtered) if psnr_vals_filtered.size > 0 else float('nan')
        return average_psnr
    

    







