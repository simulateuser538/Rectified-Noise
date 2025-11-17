# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

"""
A minimal training script for SiT using PyTorch DDP.
"""
import torch
# the first flag below was False when we tested this script but True makes A100 training a lot faster:
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler
from torchvision.datasets import ImageFolder
from torchvision import transforms
import numpy as np
from collections import OrderedDict
from PIL import Image
from copy import deepcopy
from glob import glob
from time import time
import argparse
import logging
import os

from models import SiT, SiTF1, SiTF2, CombinedModel
from models import SiT_models
from download import find_model
from transport import create_transport, Sampler
from diffusers.models import AutoencoderKL
from train_utils import parse_transport_args



#################################################################################
#                             Training Helper Functions                         #
#################################################################################

@torch.no_grad()
def update_ema(ema_model, model, decay=0.9999):
    """
    Step the EMA model towards the current model.
    """
    ema_params = OrderedDict(ema_model.named_parameters())
    model_params = OrderedDict(model.named_parameters())

    for name, param in model_params.items():
        # TODO: Consider applying only to params that require_grad to avoid small numerical changes of pos_embed
        ema_params[name].mul_(decay).add_(param.data, alpha=1 - decay)


def requires_grad(model, flag=True):
    """
    Set requires_grad flag for all parameters in a model.
    """
    for p in model.parameters():
        p.requires_grad = flag


def cleanup():
    """
    End DDP training.
    """
    dist.destroy_process_group()


def create_logger(logging_dir):
    """
    Create a logger that writes to a log file and stdout.
    """
    if dist.get_rank() == 0:  # real logger
        logging.basicConfig(
            level=logging.INFO,
            format='[\033[34m%(asctime)s\033[0m] %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(), logging.FileHandler(f"{logging_dir}/log.txt")]
        )
        logger = logging.getLogger(__name__)
    else:  # dummy logger (does nothing)
        logger = logging.getLogger(__name__)
        logger.addHandler(logging.NullHandler())
    return logger


def center_crop_arr(pil_image, image_size):
    """
    Center cropping implementation from ADM.
    https://github.com/openai/guided-diffusion/blob/8fb3ad9197f16bbc40620447b2742e13458d2831/guided_diffusion/image_datasets.py#L126
    """
    while min(*pil_image.size) >= 2 * image_size:
        pil_image = pil_image.resize(
            tuple(x // 2 for x in pil_image.size), resample=Image.BOX
        )

    scale = image_size / min(*pil_image.size)
    pil_image = pil_image.resize(
        tuple(round(x * scale) for x in pil_image.size), resample=Image.BICUBIC
    )

    arr = np.array(pil_image)
    crop_y = (arr.shape[0] - image_size) // 2
    crop_x = (arr.shape[1] - image_size) // 2
    return Image.fromarray(arr[crop_y: crop_y + image_size, crop_x: crop_x + image_size])


#################################################################################
#                                  Training Loop                                #
#################################################################################

def main(args):
    """
    Trains a new SiT model.
    """
    assert torch.cuda.is_available(), "Training currently requires at least one GPU."

    dist.init_process_group("nccl")
    assert args.global_batch_size % dist.get_world_size() == 0, f"Batch size must be divisible by world size."
    rank = dist.get_rank()
    device = rank % torch.cuda.device_count()
    seed = args.global_seed * dist.get_world_size() + rank
    torch.manual_seed(seed)
    torch.cuda.set_device(device)
    print(f"Starting rank={rank}, seed={seed}, world_size={dist.get_world_size()}.")
    local_batch_size = int(args.global_batch_size // dist.get_world_size())
    learn_mu = args.learn_mu
    depth = args.depth
    # Setup an experiment folder:
    if rank == 0:
        os.makedirs(args.results_dir, exist_ok=True) 
        experiment_index = len(glob(f"{args.results_dir}/*"))
        model_string_name = args.model.replace("/", "-") 
        if learn_mu==True:
            experiment_name = f"1New-depth-mu-{depth}-{experiment_index:03d}-{model_string_name}-" \
                            f"{args.path_type}-{args.prediction}-{args.loss_weight}"
        else:
            experiment_name = f"1New-depth-sigma-{depth}-{experiment_index:03d}-{model_string_name}-" \
                            f"{args.path_type}-{args.prediction}-{args.loss_weight}"           
        experiment_dir = f"{args.results_dir}/{experiment_name}"  
        checkpoint_dir = f"{experiment_dir}/checkpoints" 
        os.makedirs(checkpoint_dir, exist_ok=True)
        logger = create_logger(experiment_dir)
        logger.info(f"Experiment directory created at {experiment_dir}")

    else:
        logger = create_logger(None)

    # Create models:
    assert args.image_size % 8 == 0, "Image size must be divisible by 8 (for the VAE encoder)."
    latent_size = args.image_size // 8
    sitf1 = SiTF1(
        input_size=latent_size,
        patch_size=2,  
        in_channels=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=args.num_classes,
        learn_sigma=False
    ).to(device)
    sit = SiT(
        input_size=latent_size,
        patch_size=2,
        in_channels=4,
        hidden_size=768,
        depth=12,
        num_heads=12,
        mlp_ratio=4.0,
        class_dropout_prob=0.1,
        num_classes=args.num_classes,
        learn_sigma=False
    ).to(device)
    sitf2 = SiTF2(
        hidden_size=768,
        out_channels=8, 
        patch_size=2,
        num_heads=12,
        mlp_ratio=4.0,
        depth=depth,
        learn_sigma=True,
        learn_mu=learn_mu
    ).to(device)
    sitf2_ema = deepcopy(sitf2).to(device)
    combined_model = CombinedModel(sitf1, sitf2).to(device)

    if args.ckpt is not None:
        ckpt_path = args.ckpt
        state_dict = find_model(ckpt_path)
        try:
            sitf1.load_state_dict(state_dict["model"], strict=False)
            sit.load_state_dict(state_dict["model"], strict=False)
        except:
            sitf1.load_state_dict(state_dict, strict=False)
            sit.load_state_dict(state_dict, strict=False)            
        

    requires_grad(sitf1, False)
    requires_grad(sit, False)
    requires_grad(sitf2, True)

    opt = torch.optim.AdamW(sitf2.parameters(), lr=1e-4, weight_decay=0)
    sitf2 = DDP(sitf2, device_ids=[rank])
    combined_model = DDP(combined_model, device_ids=[rank])

    transport = create_transport(
        args.path_type,
        args.prediction,
        args.loss_weight,
        args.train_eps,
        args.sample_eps
    )  # default: velocity; 
    transport_sampler = Sampler(transport)
    vae = AutoencoderKL.from_pretrained(f"stabilityai/sd-vae-ft-{args.vae}").to(device)
    logger.info(f"Combined_model Parameters: {sum(p.numel() for p in combined_model.parameters()):,}")

    grad_params = [(n, p.numel()) for n, p in combined_model.named_parameters() if p.requires_grad]
    logger.info(f"Total trainable parameters: {sum(cnt for _, cnt in grad_params):,}")

    # Setup data:
    transform = transforms.Compose([
        transforms.Lambda(lambda pil_image: center_crop_arr(pil_image, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5], inplace=True)
    ])
    dataset = ImageFolder(args.data_path, transform=transform)
    sampler = DistributedSampler(
        dataset,
        num_replicas=dist.get_world_size(),
        rank=rank,
        shuffle=True,
        seed=args.global_seed
    )
    loader = DataLoader(
        dataset,
        batch_size=local_batch_size,
        shuffle=False,
        sampler=sampler,
        num_workers=args.num_workers,
        pin_memory=True,
        drop_last=True
    )
    logger.info(f"Dataset contains {len(dataset):,} images ({args.data_path})")
    update_ema(sitf2_ema, sitf2.module, decay=0)
    sitf1.eval()  
    sit.eval()
    sitf2.train()
    sitf2_ema.eval()

    train_steps = 0
    log_steps = 0
    running_loss = 0
    start_time = time()
    ys = torch.randint(1000, size=(local_batch_size,), device=device)
    use_cfg = args.cfg_scale > 1.0
    n = ys.size(0)
    zs = torch.randn(n, 4, latent_size, latent_size, device=device)
    if use_cfg:
        zs = torch.cat([zs, zs], 0)
        y_null = torch.tensor([1000] * n, device=device)
        ys = torch.cat([ys, y_null], 0)
        sample_model_kwargs = dict(y=ys, cfg_scale=args.cfg_scale)
        model_fn = sitf1.forward_with_cfg
    else:
        sample_model_kwargs = dict(y=ys)
        model_fn = sitf1.forward
    def combined_sampling_model(x, t, y=None, **kwargs):
        with torch.no_grad():
            sit_out = sit.forward(x, t, y)
            combined_out = combined_model.forward(x, t, y)
            return sit_out + combined_out
    logger.info(f"Training for {args.epochs} epochs...")
    for epoch in range(args.epochs):
        sampler.set_epoch(epoch)
        logger.info(f"Beginning epoch {epoch}...")
        for x, y in loader:
            x = x.to(device)
            y = y.to(device)
            with torch.no_grad():
                x_latent = vae.encode(x).latent_dist.sample().mul_(0.18215)
            model_kwargs = dict(y=y)
            loss_dict = transport.training_losses(sit,x_latent,model_noise=combined_model,model_kwargs=model_kwargs)
            loss = loss_dict["loss"].mean()
            opt.zero_grad()
            loss.backward()
            opt.step()
            update_ema(sitf2, sitf2)
            running_loss += loss.item()
            log_steps += 1
            train_steps += 1
            if train_steps % args.log_every == 0:
                torch.cuda.synchronize()
                end_time = time()
                steps_per_sec = log_steps / (end_time - start_time)
                avg_loss = torch.tensor(running_loss / log_steps, device=device)
                dist.all_reduce(avg_loss, op=dist.ReduceOp.SUM)
                avg_loss = avg_loss.item() / dist.get_world_size()
                logger.info(f"(step={train_steps:07d}) Train Loss: {avg_loss:.4f}, Train Steps/Sec: {steps_per_sec:.2f}")
                running_loss = 0
                log_steps = 0
                start_time = time()
            if train_steps % args.ckpt_every == 0 and train_steps > 0:
                print(train_steps)
                if rank == 0:
                    checkpoint = {
                        "model": sitf2.state_dict(),
                        "ema": sitf2.state_dict(),
                        "opt": opt.state_dict(),
                        "args": args
                    }
                    checkpoint_path = f"{checkpoint_dir}/{train_steps:07d}.pt"
                    torch.save(checkpoint, checkpoint_path)
                    logger.info(f"Saved checkpoint to {checkpoint_path}")
                dist.barrier()
            
            if (train_steps % args.sample_every == 0 )and train_steps > 0:
                logger.info("Generating EMA samples...")
            if epoch == args.epochs:
                break

    sitf1.eval()  
    sit.eval()
    sitf2.eval()
    logger.info("Final sampling done.")

    logger.info("Done!")
    cleanup()


def save_samples_grid(out_samples, epoch, experiment_index, args, experiment_name, rank):
    if rank == 0:
        import os
        import numpy as np
        from PIL import Image
        parent_dir = os.path.dirname(args.results_dir)
        pic_dir = os.path.join(parent_dir, "pic")
        os.makedirs(pic_dir, exist_ok=True)
        experiment_pic_dir = os.path.join(pic_dir, experiment_name)
        os.makedirs(experiment_pic_dir, exist_ok=True)
        samples_np = torch.clamp(127.5 * out_samples + 128.0, 0, 255).permute(0, 2, 3, 1).to("cpu", dtype=torch.uint8).numpy()
        n_samples = samples_np.shape[0]
        grid_size = int(np.ceil(np.sqrt(n_samples)))
        canvas_size = grid_size * args.image_size
        canvas = np.zeros((canvas_size, canvas_size, 3), dtype=np.uint8)
        for i, sample in enumerate(samples_np):
            row = i // grid_size
            col = i % grid_size
            canvas[row*args.image_size:(row+1)*args.image_size, col*args.image_size:(col+1)*args.image_size] = sample
        combined_image = Image.fromarray(canvas)
        combined_image.save(os.path.join(experiment_pic_dir, f"epoch_{epoch:04d}_combined.png"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-path", type=str, required=True)
    parser.add_argument("--results-dir", type=str, default="results")
    parser.add_argument("--model", type=str, choices=list(SiT_models.keys()), default="SiT-B/2")
    parser.add_argument("--image-size", type=int, choices=[256, 512], default=256)
    parser.add_argument("--num-classes", type=int, default=3)
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--global-batch-size", type=int, default=128)
    parser.add_argument("--global-seed", type=int, default=0)
    parser.add_argument("--vae", type=str, choices=["ema", "mse"], default="ema")  # Choice doesn't affect training
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--log-every", type=int, default=100)
    parser.add_argument("--ckpt-every", type=int, default=10000)
    parser.add_argument("--sample-every", type=int, default=55192)
    parser.add_argument("--cfg-scale", type=float, default=1.0)
    parser.add_argument("--ckpt", type=str, default='/gemini/space/gzy/w_w_last/afhq/w_w_sit_1/0200000.pt',
                        help="Optional path to a custom SiT checkpoint")
    parser.add_argument("--learn-mu", type=bool, default=True,
                        help="Whether to learn mu parameter")
    parser.add_argument("--depth", type=int, default=1,
                        help="Depth parameter for SiTF2 model")

    parse_transport_args(parser)
    args = parser.parse_args()
    main(args)
