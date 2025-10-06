import os
import random
import argparse
import torch
from PIL import Image
from torchvision import transforms
import torchvision.transforms.functional as F
from glob import glob

# for dark set
from torchvision.utils import save_image
from data_transform import Compose, ToTensor, Scale

import torch.multiprocessing as mp
mp.set_start_method('spawn', force=True)

def parse_args_paired_training(input_args=None):
    """
    Parses command-line arguments used for configuring an paired session (pix2pix-Turbo).
    This function sets up an argument parser to handle various training options.

    Returns:
    argparse.Namespace: The parsed command-line arguments.
   """
    parser = argparse.ArgumentParser()

    #dark data
    parser.add_argument("--dark_first_n", default=0, type=int)
    parser.add_argument("--patch_size", default=128, type=int)

    # args for the loss function
    parser.add_argument("--gan_disc_type", default="vagan_clip")
    parser.add_argument("--gan_loss_type", default="multilevel_sigmoid_s")
    parser.add_argument("--lambda_gan", default=0.5, type=float)
    parser.add_argument("--lambda_lpips", default=5, type=float)
    parser.add_argument("--lambda_l2", default=1.0, type=float)
    parser.add_argument("--lambda_clipsim", default=5.0, type=float)
    parser.add_argument("--lambda_ssim", default=5.0, type=float)

    # dataset options
    parser.add_argument("--dataset_folder", default=".", type=str)
    parser.add_argument("--train_image_prep", default="resized_crop_512", type=str)
    parser.add_argument("--test_image_prep", default="resized_crop_512", type=str)

    # validation eval args
    parser.add_argument("--eval_freq", default=100, type=int)
    parser.add_argument("--track_val_fid", default=False, action="store_true")
    parser.add_argument("--num_samples_eval", type=int, default=80, help="Number of samples to use for all evaluation")

    parser.add_argument("--viz_freq", type=int, default=100, help="Frequency of visualizing the outputs.")
    parser.add_argument("--tracker_project_name", type=str, default="train_pix2pix_turbo", help="The name of the wandb project to log to.")

    # details about the model architecture
    parser.add_argument("--pretrained_model_name_or_path")
    parser.add_argument("--revision", type=str, default=None,)
    parser.add_argument("--variant", type=str, default=None,)
    parser.add_argument("--tokenizer_name", type=str, default=None)
    parser.add_argument("--lora_rank_unet", default=16, type=int)
    parser.add_argument("--lora_rank_vae", default=8, type=int)

    # training details
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--cache_dir", default=None,)
    parser.add_argument("--seed", type=int, default=None, help="A seed for reproducible training.")
    parser.add_argument("--resolution", type=int, default=512,)
    parser.add_argument("--train_batch_size", type=int, default=4, help="Batch size (per device) for the training dataloader.")
    parser.add_argument("--num_training_epochs", type=int, default=25000)
    parser.add_argument("--max_train_steps", type=int, default=500_000,)
    parser.add_argument("--checkpointing_steps", type=int, default=500,)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=1, help="Number of updates steps to accumulate before performing a backward/update pass.",)
    parser.add_argument("--gradient_checkpointing", action="store_true",)
    parser.add_argument("--learning_rate", type=float, default=5e-6)
    parser.add_argument("--lr_scheduler", type=str, default="constant",
        help=(
            'The scheduler type to use. Choose between ["linear", "cosine", "cosine_with_restarts", "polynomial",'
            ' "constant", "constant_with_warmup"]'
        ),
    )
    parser.add_argument("--lr_warmup_steps", type=int, default=500, help="Number of steps for the warmup in the lr scheduler.")
    parser.add_argument("--lr_num_cycles", type=int, default=1,
        help="Number of hard resets of the lr in cosine_with_restarts scheduler.",
    )
    parser.add_argument("--lr_power", type=float, default=1.0, help="Power factor of the polynomial scheduler.")

    parser.add_argument("--dataloader_num_workers", type=int, default=0,)
    parser.add_argument("--adam_beta1", type=float, default=0.9, help="The beta1 parameter for the Adam optimizer.")
    parser.add_argument("--adam_beta2", type=float, default=0.999, help="The beta2 parameter for the Adam optimizer.")
    parser.add_argument("--adam_weight_decay", type=float, default=1e-2, help="Weight decay to use.")
    parser.add_argument("--adam_epsilon", type=float, default=1e-08, help="Epsilon value for the Adam optimizer")
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument("--allow_tf32", action="store_true",
        help=(
            "Whether or not to allow TF32 on Ampere GPUs. Can be used to speed up training. For more information, see"
            " https://pytorch.org/docs/stable/notes/cuda.html#tensorfloat-32-tf32-on-ampere-devices"
        ),
    )
    parser.add_argument("--report_to", type=str, default="wandb",
        help=(
            'The integration to report the results and logs to. Supported platforms are `"tensorboard"`'
            ' (default), `"wandb"` and `"comet_ml"`. Use `"all"` to report to all integrations.'
        ),
    )
    parser.add_argument("--mixed_precision", type=str, default=None, choices=["no", "fp16", "bf16"],)
    parser.add_argument("--enable_xformers_memory_efficient_attention", action="store_true", help="Whether or not to use xformers.")
    parser.add_argument("--set_grads_to_none", action="store_true",)

    if input_args is not None:
        args = parser.parse_args(input_args)
    else:
        args = parser.parse_args()

    return args

def build_transform(image_prep):
    """
    Constructs a transformation pipeline based on the specified image preparation method.

    Parameters:
    - image_prep (str): A string describing the desired image preparation

    Returns:
    - torchvision.transforms.Compose: A composable sequence of transformations to be applied to images.
    """
    if image_prep == "center_crop_512":
        T = transforms.Compose([
            transforms.CenterCrop(512),
        ])    
    if image_prep == "resized_crop_512":
        T = transforms.Compose([
            transforms.Resize(512, interpolation=transforms.InterpolationMode.LANCZOS),
            transforms.CenterCrop(512),
        ])
    elif image_prep == "randomcrop_256":
        T = transforms.Compose([
            transforms.RandomCrop((256, 256))
        ])
    elif image_prep == "randomcrop_128":
        T = transforms.Compose([
            transforms.RandomCrop((128, 128))
        ])        
    elif image_prep == "randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])  
    elif image_prep == "randomcrop_128x128_hflip":
        T = transforms.Compose([
            transforms.RandomCrop((128, 128)),
            transforms.RandomHorizontalFlip(),
        ])           
    elif image_prep == "resize_286_randomcrop_256x256_hflip":
        T = transforms.Compose([
            transforms.Resize((286, 286), interpolation=Image.LANCZOS),
            transforms.RandomCrop((256, 256)),
            transforms.RandomHorizontalFlip(),
        ])
    elif image_prep == "resize_512_randomcrop_512x512_hflip":
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS),
            transforms.RandomHorizontalFlip(),
        ])        
    elif image_prep in ["resize_256", "resize_256x256"]:
        T = transforms.Compose([
            transforms.Resize((256, 256), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_512", "resize_512x512"]:
        T = transforms.Compose([
            transforms.Resize((512, 512), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_1024", "resize_1024x1024"]:
        T = transforms.Compose([
            transforms.Resize((1024, 1024), interpolation=Image.LANCZOS)
        ])
    elif image_prep in ["resize_176_216"]:
        T = transforms.Compose([
            transforms.Resize((176, 216), interpolation=Image.LANCZOS)
        ])               
    elif image_prep == "no_resize":
        T = transforms.Lambda(lambda x: x)
    return T

class SHIQData(torch.utils.data.Dataset):
    """ Dataset -- SHIQ dataset."""

    def __init__(self, root, transform, tokenizer):

        #print("!*************************!")
        self.transform = transform
        self.A_files = glob(os.path.join(root,'*_A.png'))
        self.root = root
        self.tokenizer = tokenizer


    def __getitem__(self, index):
        imname = self.A_files[index]
        lind = imname.rfind("/")
        rind = imname.rfind("_")
        imname = imname[lind+1:rind]

        #ind = imname.find('_')
        #imname = imname[ind+1:]
        A_img_path = os.path.join(self.root, imname+'_A.png')
        D_img_path = os.path.join(self.root, imname+'_D.png')


        A = Image.open(A_img_path)
        D = Image.open(D_img_path)

        # Prepare caption tokens
        caption = "remove specular highlight from image"
        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids  

        A,D = self.transform((A,D))

        return {
            "output_pixel_values": F.normalize(D, mean=[0.5], std=[0.5]),  # Normalize GT
            "conditioning_pixel_values": A,  # No normalization for input
            "caption": caption,
            "input_ids": input_ids,
            "image_name": imname
        }

    def __len__(self):
        """Return the number of images."""
        return len(self.A_files)
    
def generate_sshr_training_data_list(data_dir, data_list_file):
    # shapenet_specular training dataset
    # training_data_dir = 'dataset/shapenet_specular_1500/training_data'
    # training_data_list_file = 'dataset/shapenet_specular_1500/train_tc.lst'

    random.seed(1)

    path_i = [] # input
    path_d = [] # diffuse
    with open(data_list_file, 'r') as f:
        image_list = [x.strip() for x in f.readlines()]
    random.shuffle(image_list)
    for name in image_list:
        path_i.append(os.path.join(data_dir, name.split()[0])) # input
        path_d.append(os.path.join(data_dir, name.split()[4])) # diffuse

    num = len(image_list)
    path_i = path_i[:int(num)]
    path_d = path_d[:int(num)]

    path_list = {'path_i': path_i,  'path_d': path_d}
    return path_list

def generate_sshr_testing_data_list(data_dir, data_list_file):
    # shapenet_specular testing data
    # data_dir = 'dataset/shapenet_specular_1500/testing_data'
    # data_list_file = 'dataset/shapenet_specular_1500/test_tc.lst'

    path_i = [] # input
    path_d = [] # diffuse
    with open(data_list_file, 'r') as f:
        image_list = [x.strip() for x in f.readlines()]
    image_list.sort()
    for name in image_list:
        path_i.append(os.path.join(data_dir, name.split()[0])) # input
        path_d.append(os.path.join(data_dir, name.split()[4])) # diffuse


    num = len(image_list)
    path_i = path_i[:int(num)]
    path_d = path_d[:int(num)]

    path_list = {'path_i': path_i, 'path_d': path_d}

    return path_list

class ImageTransform():
    def __init__(self, size=512):
        self.data_transform = {'train': Compose([Scale(size=size),
                                                 ToTensor()]),

                                'test': Compose([ToTensor()])}

    def __call__(self, phase, img):
        return self.data_transform[phase](img)

class SSHRDataset(torch.utils.data.Dataset):
    def __init__(self, img_list, img_transform, phase, tokenizer):
        self.img_list = img_list
        self.img_transform = img_transform
        self.phase = phase
        self.tokenizer = tokenizer        

    def __len__(self):
        return len(self.img_list['path_i'])

    def __getitem__(self, index):
        img_path = self.img_list['path_i'][index].split('/')
        image_name = img_path[-2] + "_" + img_path[-1].split('.')[0]
        # print(image_name)
        inp = Image.open(self.img_list['path_i'][index]).convert('RGB')
        gt_diffuse = Image.open(self.img_list['path_d'][index]).convert('RGB')

        # data pre-processing
        inp, gt_diffuse = self.img_transform(self.phase, [inp, gt_diffuse])

        # Prepare caption tokens
        caption = "remove specular highlight from image"
        input_ids = self.tokenizer(
            caption, max_length=self.tokenizer.model_max_length,
            padding="max_length", truncation=True, return_tensors="pt"
        ).input_ids 

        return {
            "output_pixel_values": F.normalize(gt_diffuse, mean=[0.5], std=[0.5]),  # Normalize GT
            "conditioning_pixel_values": inp,  # No normalization for input
            "caption": caption,
            "input_ids": input_ids,
            "image_name": image_name
        }
