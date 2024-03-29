from skimage.metrics import structural_similarity as ssim
from skimage.metrics import peak_signal_noise_ratio as psnr
import numpy as np
from torchvision.transforms.functional import to_pil_image, to_tensor
from torchvision.transforms import ToTensor
import torch
import cv2

class Metrics:

    def __init__(self, ground_truth_imgs, reconstructed_imgs, ground_truth_labels, extracted_labels):
        self.ground_truth_imgs = ground_truth_imgs
        self.reconstructed_imgs = reconstructed_imgs
        self.ground_truth_labels = ground_truth_labels
        self.extracted_labels = extracted_labels
        self.to_tensor = ToTensor()

    def _ensure_tensor(self, img):
        if not isinstance(img, torch.Tensor):
            img = self.to_tensor(img)
        return img
    
    def _tensor_to_np(self, tensor):
        return tensor.cpu().detach().numpy()
    
    def _to_numpy(self, image):
        """Converts a PIL Image or PyTorch tensor to a numpy array."""
        if isinstance(image, torch.Tensor):
            # If the image is a tensor, ensure it's in CPU and convert it to numpy
            image = image.cpu().numpy()
        elif not isinstance(image, np.ndarray):
            # Convert PIL Image to numpy array
            image = np.array(image)
        # Ensure the numpy array is in the correct format (H, W, C) for OpenCV
        if image.ndim == 3 and image.shape[0] in [1, 3, 4]:  # CHW format
            image = image.transpose(1, 2, 0)  # Convert to HWC format
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

        for gt_img, rec_img in zip(self.ground_truth_imgs, self.reconstructed_imgs):
            gt_np = self._to_numpy(gt_img)
            rec_np = self._to_numpy(rec_img)
            
            if gt_np.ndim == 3:
                gt_gray = cv2.cvtColor(gt_np, cv2.COLOR_BGR2GRAY)
            else:
                gt_gray = gt_np

            if rec_np.ndim == 3:
                rec_gray = cv2.cvtColor(rec_np, cv2.COLOR_BGR2GRAY)
            else:
                rec_gray = rec_np

            if gt_gray.dtype == np.float32 or gt_gray.dtype == np.float64:
                data_range = 1.0
            else:
                data_range = 255

            ssim_score = ssim(gt_gray, rec_gray, data_range=data_range)
            ssim_scores.append(ssim_score)

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
        average_psnr = np.mean(psnr_vals)
        return average_psnr
    
    def compute_label_accuracy(self):
        correct_label_extraction = sum(1 for true_label, extracted_label in zip(self.ground_truth_labels, self.extracted_labels) if true_label == extracted_label)
        num_labels = len(self.ground_truth_labels)
        accuracy = (correct_label_extraction / num_labels) * 100
        return accuracy

    







