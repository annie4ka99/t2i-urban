import models.models as models
import dataloaders.dataloaders as dataloaders
import utils.utils as utils
import config
import time

#--- read options ---#
opt = config.read_arguments(train=False)

#--- create dataloader ---#
dataloader_train, dataloader_val = dataloaders.get_dataloaders(opt)

#--- create utils ---#
image_saver = utils.results_saver(opt)

# --- create models ---#
model = models.OASIS_model(opt)
model = models.put_on_multi_gpus(model, opt)
model.eval()

#--- iterate over validation set ---#
start_time = time.time()
data_size = len(dataloader_val.dataset) + (len(dataloader_train.dataset) if opt.test_all else 0)
# for i, data_i in enumerate(dataloader_val):
#     image, label = models.preprocess_input(opt, data_i)
#     generated = model(None, label, "generate", None)
#     image_saver(label, generated, data_i["name"], image if opt.save_orig else None, opt.colorize_labels)

if opt.test_all:
    for i, data_i in enumerate(dataloader_train):
        image, label = models.preprocess_input(opt, data_i)
        generated = model(None, label, "generate", None)
        image_saver(label, generated, data_i["name"], image if opt.save_orig else None, opt.colorize_labels)
avg = (time.time() - start_time) / data_size
print(print('avg time for 1 image:%.3f' % avg))


