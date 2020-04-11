import argparse
import math
import os

import torch

from torch import optim
from torch.nn import functional as F
print(torch.__version__)

from torchvision import transforms

print("-")
from PIL import Image
print("-")
from tqdm import tqdm
print("-")
import lpips
print("-")
from model import *
print("-")
from dataset import FairDataset
print("-")
import numpy as np
# from models2 import VariationalAutoEncoderLite
import cv2
from skimage import exposure


# def noise_regularize(noises):
#     loss = 0
    
#     for noise in noises:
#         # size = noise.shape[2]
#         size = list(noise.shape[2:])

#         while True:
#             loss = loss + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2) \
#                         + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            
#             if size[1] <= 8:
#                 break
                
#             noise = noise.reshape([1, 1, size[0] // 2, 2, size[1] // 2, 2])
#             noise = noise.mean([3, 5])
#             size[0] //= 2
#             size[1] //= 2
            
#     return loss
def noise_regularize(noises):

    loss = 0
    for noise in noises:
        size = noise.shape[3]
        bs = noise.shape[0]        
        while True:
            # loss = loss + (noise * torch.roll(noise, shifts=1, dims=3)).mean().pow(2) \
            #             + (noise * torch.roll(noise, shifts=1, dims=2)).mean().pow(2)
            horizontal = (noise * torch.roll(noise, shifts=1, dims=3)).reshape(bs, -1).mean(-1).pow(2)
            vertical = (noise * torch.roll(noise, shifts=1, dims=2)).reshape(bs, -1).mean(-1).pow(2)
            loss = loss + horizontal + vertical
                
            if size <= 8:
                break
                
            noise = noise.reshape([bs, 1, 1, size // 2, 2, size // 2, 2])
            noise = noise.mean([4, 6])
            size //= 2
            
    return loss


def noise_normalize_(noises):
    for noise in noises:
        mean = noise.mean()
        std = noise.std()
        
        noise.data.add_(-mean).div_(std)
        
        
def get_lr(t, initial_lr, rampdown=0.25, rampup=0.05):
    lr_ramp = min(1, (1 - t) / rampdown)
    lr_ramp = 0.5 - 0.5 * math.cos(lr_ramp * math.pi)
    lr_ramp = lr_ramp * min(1, t / rampup)
    
    return initial_lr * lr_ramp


def latent_noise(latent, strength):

    noise = torch.randn_like(latent) * strength
    
    return latent + noise


def make_image(tensor):

    tmp = tensor.detach().permute(0, 2, 3, 1).to('cpu').numpy()

    im = tmp - np.amin(tmp)
    im = im / np.amax(im)
    im = im * 255


    return  im

def d_feat_loss(imgs, gen_imgs):
    img_gen_feat = discriminator(img_gen, return_features=True)
    imgs_feat = discriminator(batch_imgs, return_features=True)        


    p_loss = F.l1_loss(img_gen_feat, imgs_feat, reduction="none")

    return p_loss




if __name__ == '__main__':
    device = 'cuda'
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--ckpt', type=str, required=True)
    parser.add_argument('--size', type=int, default=512)
    parser.add_argument('--batch_size', type=int, default=1)
    parser.add_argument('--lr_rampup', type=float, default=0.05)
    parser.add_argument('--nodule_intensity', type=float, default=1)
    parser.add_argument('--nodule_size', type=float, default=10)
    parser.add_argument('--run', type=int, default=0)
    parser.add_argument('--n_gpu', type=int, default=1)
    parser.add_argument('--lr_rampdown', type=float, default=0.25)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--noise', type=float, default=0.05)
    parser.add_argument('--noise_ramp', type=float, default=0.75)
    parser.add_argument('--step', type=int, default=1000)
    parser.add_argument('--noise_regularize', type=float, default=1e5)
    parser.add_argument('--mse', type=float, default=0)
    parser.add_argument('--mask', type=int, default=0)
    parser.add_argument('--mask_val', type=str, default="zero")

    parser.add_argument('--feature_extractor', type=str, default="d")
    parser.add_argument('--latent_space', type=str, default="w")    
    parser.add_argument('--w_plus', action='store_true')

    parser.add_argument('files', metavar='FILES', nargs='+')

    
    args = parser.parse_args()

    ckpt_run = args.ckpt.split("/")[-1]

    data_mean = 0.175
    data_std = 0.17
    min_v = - data_mean / data_std
    max_v = (1 - data_mean) / data_std


    n_mean_latent = 10000

    
    # resize = min(args.size, 256)
    
    transform = transforms.Compose([
            # transforms.Resize(resize),
           # transforms.CenterCrop(resize),
           transforms.ToTensor()
           # transforms.Normalize([0.5, 0.5, 0.5],
                               # [0.5, 0.5, 0.5])
        ])
    print("loading data")
    test_data = FairDataset(None, transform=transform, reg=False, resolution=512, split="test", run=args.run, intensity=args.nodule_intensity, size=args.nodule_size, nodule_mask=args.mask)
    print("loaded data")
    imgs = []
    
    # for imgfile in args.files:
    #     # img = transform(Image.open(imgfile).convert('RGB'))

    #     img = transform(Image.fromarray(np.load(imgfile)))
    #     imgs.append(img)
    for i in range(len(test_data)):
        imgs.append(test_data[i])


    imgs = torch.stack(imgs, 0).to(device).to(torch.float) # tmp
    print(imgs.shape)
    if len(imgs.shape) == 3:
        imgs = imgs.unsqueeze(0)

    
    g_ema = Generator(args.size, 512, 8, 2)
    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'])
    g_ema.eval()
    g_ema = g_ema.to(device)
    g_ema = MyDataParallel(g_ema, device_ids=range(args.n_gpu))


    if args.feature_extractor == "d":
        discriminator = Discriminator(
            args.size, channel_multiplier=2
        )
        discriminator.load_state_dict(torch.load(args.ckpt)['d'])
        discriminator.eval()
        discriminator = discriminator.to(device)
        discriminator = MyDataParallel(discriminator, device_ids=range(args.n_gpu))
        percept = d_feat_loss

    elif args.feature_extractor == "vgg":
        percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=device.startswith('cuda'), gpu_ids=range(args.n_gpu))

    # checkpoint = torch.load("/scratch/gobi2/lechang/trained_vae.pth")
    # vae = VariationalAutoEncoderLite(5)
    # vae.load_state_dict(checkpoint['ae'])
    # vae = vae.to(device)
    # vae.eval()



    print("loaded models")
    if args.latent_space == "w":
        with torch.no_grad():
            noise_sample = torch.randn(n_mean_latent, 512, device=device)
            latent_out = g_ema.style(noise_sample)
            
            latent_mean = latent_out.mean(0)

            # optimize W
            latent_std = (((latent_out - latent_mean).pow(2).sum() / n_mean_latent) ** 0.5).item()

            # optimize z
            # latent_std = 1
        np.save("latent_mean.npy", latent_mean.cpu().numpy())
        print("saved latent mean")
    elif args.latent_space == "z":
        latent_mean = torch.zeros(1, 512, device=device)
        latent_std = 1


    test_latents = torch.zeros((len(imgs), g_ema.n_latent, 512))
    test_noises = g_ema.make_noise(len(imgs))
    p_losses = torch.zeros(len(imgs))
    n_losses = torch.zeros(len(imgs))
    mse_losses = torch.zeros(len(imgs))
    for batch_start in range(0, len(imgs), args.batch_size):

        batch_imgs = imgs[batch_start: batch_start + args.batch_size]

        noises = g_ema.make_noise(batch_imgs.shape[0])
        # latent_in = latent_mean.detach().clone().unsqueeze(0).repeat(2, 1)


        # log_size = int(math.log(256, 2))

        # optimize W
        if args.latent_space == "w":
            latents = latent_mean.detach().clone().unsqueeze(0).unsqueeze(0).repeat(batch_imgs.shape[0], g_ema.n_latent, 1) 
        elif args.latent_space == "z":
            latents = torch.zeros((batch_imgs.shape[0], g_ema.n_latent, 512), device=device)

        # optimize Z
        # latents = [torch.zeros((1, 512)).to(device)
                    # for _ in range(2 * log_size - 3)]



        # if args.w_plus:
        #     latent_in.unsqueeze(1).repeat(1, g_ema.n_latent, 1)
        # for l in latents:
            # l.requires_grad = True
        
        latents.requires_grad = True

        for noise in noises:
            # noise.requires_grad = True
            noise.requires_grad = True
            
        optimizer = optim.Adam([latents] + noises, lr=args.lr)
        
        pbar = tqdm(range(args.step))
        # latent_path = []

        query = batch_imgs.clone()
        # get nodule masks
        if args.mask:
            masks = []
            mask_means = []

            for i in range(batch_start, batch_start + len(batch_imgs)):
                m = torch.from_numpy(test_data.get_nodule_mask(i)).float().unsqueeze(0).unsqueeze(0)
                masks.append(m)

            masks = torch.cat(masks).to(device)


            masks = 1 - masks





            if args.mask_val == "zero":
                batch_imgs[masks==0] = test_data.min_val
            elif args.mask_val == "dataset_mean":
                batch_imgs = batch_imgs * masks

            elif args.mask_val == "patch_mean":
                patch_means = batch_imgs[masks == 0].reshape(len(batch_imgs), -1).mean(-1).unsqueeze(1)
                print(patch_means.shape, "patch mean shape")


                mean_mask = (torch.ones_like(batch_imgs[masks==0]).reshape(len(batch_imgs), -1) * patch_means).reshape(batch_imgs[masks==0].shape)
                batch_imgs[masks==0] = mean_mask


        for i in pbar:
            t = i / args.step
            lr = get_lr(t, args.lr)
            optimizer.param_groups[0]['lr'] = lr
            noise_strength = latent_std * args.noise * max(0, 1 - t / args.noise_ramp) ** 2




            latent_n = latent_noise(latents, noise_strength)
            img_gen, _ = g_ema(latent_n, input_is_latent=(args.latent_space == "w"), noise=noises, multiple_latents=True)

            if args.mask:
                if args.mask_val == "zero":
                    img_gen[masks==0] = test_data.min_val
                elif args.mask_val == "dataset_mean":
                    img_gen = img_gen * masks

                elif args.mask_val == "patch_mean":

                    img_gen[masks==0] = mean_mask
                        
            # what does this do ??
            # if height > 256:
            #     factor = height // 256
                
            #     img_gen = img_gen.reshape(batch, channel, height // factor, factor, width // factor, factor)
            #     img_gen = img_gen.mean([3, 5])



            # ==========================

            

            p_loss = percept(img_gen, batch_imgs).reshape(batch_imgs.shape[0], -1).mean(-1)
            
            n_loss = noise_regularize(noises).reshape(batch_imgs.shape[0], -1).mean(-1)

            mse_loss = F.l1_loss(img_gen, batch_imgs, reduction="none")
            mse_loss = mse_loss.reshape(mse_loss.shape[0], -1).mean(-1)


            loss = p_loss.mean() + args.noise_regularize * n_loss.mean() + args.mse * mse_loss.mean()

            # loss = p_loss + args.mse * mse_loss

                
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            noise_normalize_(noises)
            
            # if (i + 1) % 1 == 0:
                # latent_path.append(latent_in.detach().clone())
                
            # pbar.set_description((f'perceptual: {p_loss.item():.4f}; noise regularize: {n_loss.item():.4f};'
            #                      f' mse: {mse_loss.item():.4f}; lr: {lr:.4f}'))
            pbar.set_description((f'perceptual: {p_loss.mean().item():.4f};'
                                 f' mse: {mse_loss.mean().item():.4f}; lr: {lr:.4f}'))


        p_losses[batch_start: batch_start + args.batch_size] = p_loss.reshape(-1).detach()
        n_losses[batch_start: batch_start + args.batch_size] = n_loss.reshape(-1).detach()
        mse_losses[batch_start: batch_start + args.batch_size] = mse_loss.reshape(-1).detach()

        

        # optimize W
        img_gen, _ = g_ema(latents, input_is_latent=args.latent_space == "w", noise=noises, multiple_latents=True)

        test_latents[batch_start: batch_start + args.batch_size] = latents.detach()
        for i in range(len(test_noises)):
            test_noises[i][batch_start: batch_start + args.batch_size] = noises[i].detach()


        filename = os.path.splitext(os.path.basename(args.files[0]))[0] + '.pt'


        print(torch.max(batch_imgs))
        print(torch.max(img_gen))


        img_gen = torch.clamp(img_gen, min_v, max_v)

        img_ar = make_image(batch_imgs)
        gen_ar = make_image(img_gen)
        query_ar = make_image(query)

        print(np.amin(img_ar), np.amax(img_ar))
        print(np.amin(gen_ar), np.amax(gen_ar))
        
        for n in range(10):

            # result_file[input_name] = {'img': img_gen[i], 'latent': latent_in[i]} 
            img_name = "gen_imgs/test_diseased_{}_{}_{}_{}_{}_{}_{}_{}_{}_{}.png".format(args.nodule_intensity, args.nodule_size, args.run, args.feature_extractor, ckpt_run, args.latent_space, args.mse, args.mask, args.mask_val, batch_start + n)
            # img_name = "gen_imgs/diseased_{}_{}_{}_{}_{}_{}_{}_project.png".format(args.run, args.nodule_intensity, args.feature_extractor, batch_start + n, args.latent_space, args.mask)

            # print(img_ar.shape, img_ar.dtype, np.amax(img_ar), np.amin(img_ar))
            # found = exposure.rescale_intensity(img_ar[i])
            # query = exposure.rescale_intensity(img_ar[-1])

            cv2.imwrite(img_name, np.concatenate((query_ar[n], img_ar[n], gen_ar[n]), axis=1))



            # n_latent = g_ema.n_latent

            # w_sum = 0
            # noise_sum = 0
            # for j in range(n_latent):
            #     w_d = torch.mean(latents[n, j] ** 2).item()
            #     if j < g_ema.n_latent - 2:
            #         noise_d = torch.mean(noises[j][n] ** 2).item()
            #     else:
            #         noise_d = 0
            #     print("latent {} === w distance: {}   noise distance: {}".format(j, w_d, noise_d))

            #     w_sum += w_d
            #     noise_sum += noise_d


            # print("average w distance:", w_sum/g_ema.n_latent)
            # print("average noise distance:", noise_sum/len(noises))

            # for j in range(100):
            #     psi = 0.99 - 0.01 * j

            #     more_likely_latents = [psi * latent for latent in latents]
            #     with torch.no_grad():
            #         im, _ = g_ema(more_likely_latents, input_is_latent=True, noise=noises)


            #     cv2.imwrite("test/more_likely_{}".format(j) + img_name, make_image(im)[0])
   
    result_file = {
        "noises": test_noises,
        "latents": test_latents,
        "p_losses": p_losses,
        "n_losses": n_losses,
        "mse_losses": mse_losses
    }
    

    # test_noises = [noise.cpu().numpy() for noise in test_noises]
    # np.save("latents/test_diseased_{}_{}_{}_{}_{}_{}_latents.npy".format(args.nodule_intensity, args.run, args.feature_extractor, ckpt_run, args.latent_space, args.mse), 
    #     test_latents.detach().cpu().numpy())




    # np.savez("latents/test_diseased_{}_{}_{}_{}_{}_{}_noises.npz".format(args.nodule_intensity, args.run, args.feature_extractor, ckpt_run, args.latent_space, args.mse),
    #     noise1=test_noises[0],
    #     noise2=test_noises[1],
    #     noise3=test_noises[2],
    #     noise4=test_noises[3],
    #     noise5=test_noises[4],
    #     noise6=test_noises[5],
    #     noise7=test_noises[6],
    #     noise8=test_noises[7],
    #     noise9=test_noises[8],
    #     noise10=test_noises[9],
    #     noise11=test_noises[10],
    #     noise12=test_noises[11],
    #     noise13=test_noises[12],
    #     noise14=test_noises[13],
    #     noise15=test_noises[14]
    #     )

        
    torch.save(result_file,
        "latents/test_diseased_{}_{}_{}_{}_{}_{}_{}_{}_{}_losses".format(args.nodule_intensity, args.nodule_size, args.run, args.feature_extractor, ckpt_run, args.latent_space, args.mse, args.mask, args.mask_val))