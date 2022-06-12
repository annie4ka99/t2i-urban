import generation.OASIS.models.models as models
import pickle
import numpy as np
from PIL import Image
import torch
from torchvision import transforms as TR


oasis_models = []
semantic_nc = 31
configs = [("oasis_real_lap", 'best'), ("oasis_real_no_vgg", '160000'), ("oasis_ade_no_vgg", 'latest')]
for name, ckpt in configs:
    opt = pickle.load(open(f"generation/OASIS/checkpoints/{name}/opt.pkl", 'rb'))
    opt.phase = "test"
    opt.crop_size = 512
    opt.aspect_ratio = 1.0
    opt.label_nc = 30
    opt.contain_dontcare_label = True
    opt.semantic_nc = semantic_nc
    opt.ckpt_iter = ckpt
    opt.checkpoints_dir = 'generation/OASIS/checkpoints'
    opt.gpu_ids = '0'
    opt.name = name
    if name == "oasis_real_lap":
        opt.upsample_mode = 'bilinear'
    else:
        opt.upsample_mode = 'nearest'

    # --- create models ---#
    model = models.OASIS_model(opt)
    model = models.put_on_multi_gpus(model, opt)
    model.eval()
    oasis_models.append(model)


def preprocess_input(labels, semantic_nc):
    label = torch.stack([TR.functional.to_tensor(label) for label in labels])
    label = label * 255
    label = label.long().cuda()
    bs, _, h, w = label.size()
    nc = semantic_nc
    input_label = torch.cuda.FloatTensor(bs, nc, h, w).zero_()
    input_semantics = input_label.scatter_(1, label, 1.0)
    return input_semantics


def tens_to_im(tens):
    out = (tens + 1) / 2
    out.clamp(0, 1)
    return np.transpose(out.detach().cpu().numpy(), (1, 2, 0))


def process_results(generated):
    results = []
    for i in range(len(generated)):
        im = tens_to_im(generated[i]) * 255
        results.append(Image.fromarray(im.astype(np.uint8)))
    return results


def generate_images(labels):
    labels = preprocess_input(labels, semantic_nc)
    results = []

    for model in oasis_models:
        generated = model(None, labels, "generate", None)
        results += process_results(generated)

    return results
