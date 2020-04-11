import numpy as np

import cv2
import matplotlib.pyplot as plt
from skimage import exposure, color

import random
from tqdm import tqdm
import pandas as pd
from sklearn.linear_model import LogisticRegression


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
import scipy.misc

from sklearn.metrics import roc_curve, auc, roc_auc_score


parser = argparse.ArgumentParser()
parser.add_argument('--ckpt', type=str, required=True)
parser.add_argument('--ckptE', type=str)
parser.add_argument('--gan', type=str, default="stylegan2")


# parser.add_argument('--run', type=int, required=True)
# parser.add_argument('--mse', type=float, required=True)

# parser.add_argument('--healthy', type=str, required=True)




args = parser.parse_args()


# args.ckpt = "test_diseased_0.0_10.0_2_vgg_reswgangp_lambda20.0_lrG0.0001_lrD0.0002_iter21000.pt"


w_mean = np.load("latent_mean.npy")

metadata = pd.read_csv("../data/preproc_chest_metadata.csv")

# latents_healthy = np.load("latents/test_diseased_0.0_0_vgg_reg_chest_031275.pt_latents.npy")
# noises_healthy = np.load("latents/test_diseased_0.0_0_vgg_reg_chest_031275.pt_latents.npy")

# healthy_dist = np.sum(latents_healthy ** 2, axis=-1)


device = "cuda"

if args.gan == "stylegan2":
    g_ema = Generator(512, 512, 8, 2)
    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'])
    g_ema.eval()
    g_ema = g_ema.to(device)

    discriminator = Discriminator(
        512, channel_multiplier=2
    )
    discriminator.load_state_dict(torch.load(args.ckpt)['d'])
    discriminator.eval()
    discriminator = discriminator.to(device)

elif args.gan == "wgan-gp":

    discriminator = ResidualDiscriminator(1, dim=128).to(device)
    discriminator.eval()
    discriminator.load_state_dict(torch.load(args.ckpt)['d'])

    g_ema = ResidualGenerator(512, dim=128).to(device)
    g_ema.load_state_dict(torch.load(args.ckpt)['g_ema'])
    g_ema.eval()

    # encoder = Encoder(1, dim=128).to(device)

    # encoder.load_state_dict(torch.load(args.ckptE))
    # encoder.eval()



percept = lpips.PerceptualLoss(model='net-lin', net='vgg', use_gpu=True)


def get_npy_idx(patient_n):
    # print(test_metadata[test_metadata["patient_n"].isin(patient_n)])
    return np.arange(95)[test_metadata["patient_n"].isin(patient_n)]


def get_lr_features(losses, query_images, anomaly_score=False,):

    latents = losses["latents"]

    # noises = [noises[key] for key in noises.files]
    # dist = np.sum(latents ** 2, axis=-1)

    percept_losses = []
    pixel_losses = []
    for i in range(len(latents)):
        with torch.no_grad():
            if args.gan == "wgan-gp":
                latent = latents[i].unsqueeze(0).to(device)
                img_gen = g_ema(latent)
            else:

                for j in range(100):
                    # noise = g_ema.make_noise(1)   
                    noises = losses["noises"]
                    noise = [n[i].unsqueeze(0).to(device) for n in noises]
                    latent = latents[i].unsqueeze(0).to(device)
                    # latent[-2:] = g_ema.style(torch.randn((2, 512)))

                    img_gen, _ = g_ema(latent, input_is_latent=True, noise=noise, multiple_latents=True)

                    #         # print(img_gen.shape)
                    im = img_gen.squeeze(0).squeeze(0).cpu().numpy()
                    im = im - np.amin(im.flatten())
                    im = im / np.amax(im.flatten()) * 255
                    
                    cv2.imwrite("test/test{}_same.png".format(j), im)
                    

            break
    


            query= query_images[i]
            query = query.unsqueeze(0).to(device)


            if test_mask:
                mask_ar = torch.from_numpy(query_images.get_nodule_mask(i)).unsqueeze(0).unsqueeze(0).to(device).float()

                query = query * mask_ar
                img_gen = img_gen * mask_ar



            # img_name = "gen_imgs/diseased_{}_{}_{}_{}_{}_{}_{}_project.png".format(args.run, args.nodule_intensity, args.feature_extractor, batch_start + n, args.latent_space, args.mask)
            # cv2.imwrite(img_name, np.concatenate((img_ar[n], gen_ar[n])))





            if fe == "vgg":
                percept_loss = percept(img_gen, query).mean()
            elif fe == "d":
            # disc_loss = F.l1_loss(gen_feat, query_feat)
                gen_feat = discriminator(img_gen, return_features=True)
                query_feat = discriminator(query, return_features=True)
                percept_loss = F.l1_loss(gen_feat, query_feat)

            percept_losses.append(percept_loss.item())

            # pixel_loss = F.l1_loss(img_gen, query)


            pixel_loss = F.l1_loss(img_gen, query)
            pixel_losses.append(pixel_loss.item())

            # plt.imshow(query_images[i].squeeze(0))
            # plt.show()

            # plt.imshow(img_gen.cpu().numpy())
            # plt.show()
    # print(img_gen.shape)

    percept_losses = np.array(percept_losses)[..., np.newaxis]
    pixel_losses = np.array(pixel_losses)[..., np.newaxis]

    features = np.concatenate((percept_losses, pixel_losses), axis=-1)
    # print(dis t.shape, features.shape)
    if anomaly_score:
        print(features[:, -1].reshape(-1, 1).shape, features[:, -1].reshape(-1, 1))        
        return features[:, -1].reshape(-1, 1)

    return features

# def get_lr_features_wgan(losses, query_images, anomaly_score=False,):

#     latents = losses["latents"]
#     noises = losses["noises"]

#     # noises = [noises[key] for key in noises.files]
#     # dist = np.sum(latents ** 2, axis=-1)

#     percept_losses = []
#     pixel_losses = []
#     for i in range(len(query_images)):
#         with torch.no_grad():

#             z = encoder(query_images[i].unsqueeze(0).to(device))

#             img_gen = g_ema(z)

#             # print(img_gen.shape)

#             if test_mask:
#                 mask_ar = torch.from_numpy(query_images.get_nodule_mask(i)).unsqueeze(0).unsqueeze(0).to(device).float()

#                 query = query * mask_ar
#                 img_gen = img_gen * mask_ar

#             im = img_gen.squeeze(0).squeeze(0).cpu().numpy()
#             im = im - np.amin(im.flatten())
#             im = im / np.amax(im.flatten()) * 255
#             cv2.imwrite('test/test{}.png'.format(i), im)



#             # img_name = "gen_imgs/diseased_{}_{}_{}_{}_{}_{}_{}_project.png".format(args.run, args.nodule_intensity, args.feature_extractor, batch_start + n, args.latent_space, args.mask)
#             # cv2.imwrite(img_name, np.concatenate((img_ar[n], gen_ar[n])))





#             if fe == "vgg":
#                 percept_loss = percept(img_gen, query).mean()
#             elif fe == "d":
#             # disc_loss = F.l1_loss(gen_feat, query_feat)
#                 gen_feat = discriminator(img_gen, return_features=True)
#                 query_feat = discriminator(query, return_features=True)
#                 percept_loss = F.l1_loss(gen_feat, query_feat)

#             percept_losses.append(percept_loss.item())

#             # pixel_loss = F.l1_loss(img_gen, query)


#             pixel_loss = F.l1_loss(img_gen, query)
#             pixel_losses.append(pixel_loss.item())

#             # plt.imshow(query_images[i].squeeze(0))
#             # plt.show()

#             # plt.imshow(img_gen.cpu().numpy())
#             # plt.show()
#     # print(img_gen.shape)

#     percept_losses = np.array(percept_losses)[..., np.newaxis]
#     pixel_losses = np.array(pixel_losses)[..., np.newaxis]

#     features = np.concatenate((percept_losses, pixel_losses), axis=-1)
#     # print(dis t.shape, features.shape)
#     if anomaly_score:
#         print(features[:, -1].reshape(-1, 1).shape, features[:, -1].reshape(-1, 1))        
#         return features[:, -1].reshape(-1, 1)

#     return features

def split(patient_list, healthy_features, diseased_features, train_ratio=1, disease_ratio=0.5):

    # split train/test patients
    train_idx = random.sample(patient_list, int(len(patients) * train_ratio))
    test_idx = list(set(patients) - set(train_idx))

    # train
    healthy_idx_train = random.sample(train_idx, int(len(train_idx) * (1 - disease_ratio)))
    diseased_idx_train = list(set(train_idx) - set(healthy_idx_train))

    healthy_npy_train = get_npy_idx(healthy_idx_train)
    diseased_npy_train = get_npy_idx(diseased_idx_train)

    # test
    healthy_idx_test = random.sample(test_idx, int(len(test_idx) * (1 - disease_ratio)))
    diseased_idx_test = list(set(test_idx) - set(healthy_idx_test))

    healthy_npy_test = get_npy_idx(healthy_idx_test)
    diseased_npy_test = get_npy_idx(diseased_idx_test)


    X_train = np.concatenate((healthy_features[healthy_npy_train], diseased_features[diseased_npy_train]))
    y_train = np.concatenate((np.zeros(len(healthy_npy_train)), np.ones(len(diseased_npy_train))))

    X_test = np.concatenate((healthy_features[healthy_npy_test], diseased_features[diseased_npy_test]))
    y_test = np.concatenate((np.zeros(len(healthy_npy_test)), np.ones(len(diseased_npy_test))))

    return X_train, y_train, X_test, y_test


ascore = False
ckpt_root = "latents/"

coefs = []

transform = transforms.Compose([
       transforms.ToTensor()
    ])


# size = 10.0
mask = 30
healthy_run = 9
# mask size used during latent optimization
for size in [10.0]:
    for intensity in [0.5]:
        for mask_val in ["zero"]:
            for test_mask in [mask]: # [25, 50, 100, 200, 512]:
                for fe in ["vgg"]:

                    for mse in [1.0]:

                        for run in [9]:
                            healthy_query = FairDataset(None, transform=transform, reg=False, resolution=512, split="test", run=run, intensity=0, size=size, nodule_mask=mask) # mask size used for testing
                            diseased_query = FairDataset(None, transform=transform, reg=False, resolution=512, split="test", run=run, intensity=intensity, size=size, nodule_mask=mask)    

                            # if args.gan == "wgan-gp":
                            #     healthy_features =  get_lr_features(healthy_query, anomaly_score=ascore)
                            #     diseased_features = get_lr_features(diseased_query, anomaly_score=ascore)



                            # elif args.gan == "stylegan2":
               
                            if mask_val == "":
                                mask_fn = "0"
                            else:
                                mask_fn = mask
                            try:
                                healthy_fn = ckpt_root + "test_diseased_{2}_{5}_{0}_{3}_{7}_w_{1}_{4}_{6}_losses".format(run, mse, "0.0", fe, mask_fn, size, mask_val, args.ckpt.split("/")[-1])
                                healthy_losses = torch.load(healthy_fn)
                            except:
                                healthy_fn = ckpt_root + "test_diseased_{2}_{5}_{0}_{3}_{7}_w_{1}_{4}_{6}_losses".format(healthy_run, mse, "0.0", fe, mask_fn, size, mask_val, args.ckpt.split("/")[-1])
                                healthy_losses = torch.load(healthy_fn)

                                print("using run {} for healthy".format(healthy_run))

                            healthy_features =  get_lr_features(healthy_losses, healthy_query, anomaly_score=ascore)

                            # healthy_features = healthy_losses["p_losses"] + 1e-5 * healthy_losses["n_losses"] + mse * healthy_losses["mse_losses"]
                            
                            try:
                                diseased_fn = ckpt_root + "test_diseased_{2}_{5}_{0}_{3}_{7}_w_{1}_{4}_{6}_losses".format(run, mse, intensity, fe, mask_fn, size, mask_val, args.ckpt.split("/")[-1])
                                diseased_losses = torch.load(diseased_fn)
                            except:
                                print(diseased_fn, "missing")
                                continue

                            diseased_features = get_lr_features(diseased_losses, diseased_query, anomaly_score=ascore)

                            # healthy_query = FairDataset(None, transform=transform, reg=False, resolution=512, split="test", run=run, intensity=0, size=size)
                            # healthy_features =  get_lr_features(healthy_prefix + "_latents.npy", healthy_prefix + "_noises.npz", healthy_query, anomaly_score=ascore)

                            # diseased_query = FairDataset(None, transform=transform, reg=False, resolution=512, split="test", run=run, intensity=intensity, size=size)    
                            # diseased_features = get_lr_features(diseased_prefix + "_latents.npy", diseased_prefix + "_noises.npz", diseased_query, anomaly_score=ascore)

                            
                            


                            # BOOTSTRAP FOR LOGISTIC REGRESSION OVER STYLE LATENTS
                            # healthy_dist = np.sum((w_healthy - w_mean) ** 2, axis=-1)
                            # diseased_dist = np.sum((w_diseased - w_mean) ** 2, axis=-1)
                            # diseased_dist = np.sum(w_diseased ** 2, axis=-1)

                            scores = []
                            anomaly_scores = []
                            aucs = []
                            lr_aucs = []
                            tprs = []
                            fprs = []
                            for i in range(100):

                                test_metadata = metadata[metadata["train"] == 0]

                                patients = list(metadata[metadata["train"] == 0]["patient_n"].unique())

                                # X_train, y_train, X_test, y_test = split(patients, train_ratio=0.75)
                                X_test, y_test, _, _ = split(patients, healthy_features, diseased_features, train_ratio=1, disease_ratio=0.5)


                                # clf = LogisticRegression(random_state=0, max_iter=1000, solver="liblinear").fit(X_train, y_train)

                                # # print(clf.predict_proba(X_test))

                                # lr_y_scores = clf.predict_proba(X_test)[:, 1]
                                # lr_fpr, lr_tpr, _ = roc_curve(y_test, lr_y_scores)
                                # lr_roc_auc = auc(lr_fpr, lr_tpr)
                                

                                y_scores = X_test[:, -1] # pixel losses only
                                fpr, tpr, thresh = roc_curve(y_test, y_scores)
                                roc_auc = auc(fpr, tpr)

                                # precision, recall, thresholds = precision_recall_curve(y_test, y_score)
                                # prc_auc = auc(precision, recall)


                                aucs.append(roc_auc)

                                # scores.append(clf.score(X_test, y_test))
                                # print(clf.predict_proba(X_test))
                                # lr_aucs.append(lr_roc_auc)
                                # coefs.append(clf.coef_)
                                # fprs.append(fpr)
                                # tprs.append(tpr)

                            # lw = 2
                            # plt.plot(np.mean(fpr, axis=-1), np.mean(tpr, axis=-1), color='darkorange',
                            #          lw=lw, label='ROC curve (area = %0.2f)' % roc_auc)
                            # plt.plot([0, 1], [0, 1], color='navy', lw=lw, linestyle='--')
                            # plt.xlim([0.0, 1.0])
                            # plt.ylim([0.0, 1.05])
                            # plt.xlabel('False Positive Rate')
                            # plt.ylabel('True Positive Rate')
                            # plt.legend(loc="lower right")
                            # # print(clf.get_params())
                            # # print(X_train, y_train)-
                            # plt.savefig("auprc {}.png".format(intensity))

                            # fig, axs = plt.subplots(3)
                            # fig.suptitle("test_diseased_{2}_{0}_{3}_reg_chest_031275.pt_w_{1}_losses_auc_{4}_{5}".format(run, mse, intensity, fe, np.mean(aucs), np.std(aucs)))
                            # axs[0].hist(healthy_losses["p_losses"].reshape(-1), alpha=0.5, label='healthy p losses')
                            # axs[0].hist(diseased_losses["p_losses"].reshape(-1), alpha=0.5, label='diseased p losses')
                            # axs[1].hist(healthy_losses["mse_losses"].reshape(-1), alpha=0.5, label='healthy mse losses')
                            # axs[1].hist(diseased_losses["mse_losses"].reshape(-1), alpha=0.5, label='diseased mse losses')
                            # axs[2].hist(healthy_losses["n_losses"].reshape(-1), alpha=0.5, label='healthy n losses')
                            # axs[2].hist(diseased_losses["n_losses"].reshape(-1), alpha=0.5, label='diseased n losses')
                            

                            # axs[0].legend(loc='upper right')
                            # axs[1].legend(loc='upper right')
                            # axs[2].legend(loc='upper right')

                            # fig.savefig("figures/test_diseased_{2}_{0}_{3}_reg_chest_031275.pt_w_{1}_losses.png".format(run, mse, intensity, fe))

                            fig, axs = plt.subplots(1)
                            fig.suptitle("test_diseased_{2}_{0}_{3}_reg_chest_031275.pt_w_{1}_{6}_{7}_{8}_losses_auc_{4}_{5}".format(run, mse, intensity, fe, np.mean(aucs), np.std(aucs), mask, test_mask, mask_val))
                            axs.hist(healthy_features[:, -1].reshape(-1), alpha=0.5, label='healthy p losses')
                            axs.hist(diseased_features[:, -1].reshape(-1), alpha=0.5, label='diseased p losses')
                        

                            axs.legend(loc='upper right')

                            fig.savefig("figures/test_diseased_{2}_{5}_{0}_{3}_reg_chest_031275.pt_w_{1}_{4}_{6}.png".format(run, mse, intensity, fe, mask, size, mask_val))
                            fig.clf()
                            

                            print("fe", fe, "intensity", intensity, "size", size, "mse", mse, "run", run, "mask", mask, "mask val", mask_val, "auc", np.mean(aucs), np.std(aucs))
                            # print("lr auc:", np.mean(lr_aucs), np.std(lr_aucs))
                            # print(np.mean(scores), np.std(scores))
                            # print("coef means:", np.mean(coefs, axis=0), np.std(coefs, axis=0))
