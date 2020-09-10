import os
import argparse
from modules.advanced_gans.models import *
from torch.autograd import Variable
from modules.models import cPix2PixDiscriminator
import time
import itertools
import pickle, gc
from modules.helpers import (ToTensor,
                             torch,
                             show_intermediate_results,
                             Resize,
                             create_dataloaders,
                             impute_reals_into_fake,
                             save_checkpoint,
                             load_checkpoint,
                             generate_training_strategy,
                             calculate_metrics,
                             printTable)
import logging
import numpy as np
import copy, sys

try:
    logger = logging.getLogger(__file__.split('/')[-1])
except:
    logger = logging.getLogger(__name__)

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

parser = argparse.ArgumentParser()
parser.add_argument('--epoch', type=int, default=0, help='epoch to start training from')
parser.add_argument('--n_epochs', type=int, default=2, help='number of epochs of training')
parser.add_argument('--dataset', type=str, default="BRATS2018", help='name of the dataset')
parser.add_argument('--grade', type=str, default="LGG", help='grade of tumor to train on')
parser.add_argument('--path_prefix', type=str, default="", help='path prefix to choose')
parser.add_argument('--batch_size', type=int, default=4, help='size of the batches')
parser.add_argument('--lr', type=float, default=0.0002, help='adam: learning rate')
parser.add_argument('--b1', type=float, default=0.5, help='adam: decay of first order momentum of gradient')
parser.add_argument('--b2', type=float, default=0.999, help='adam: decay of first order momentum of gradient')
parser.add_argument('--decay_epoch', type=int, default=100, help='epoch from which to start lr decay')
parser.add_argument('--n_cpu', type=int, default=0, help='number of cpu threads to use during batch generation')
parser.add_argument('--img_height', type=int, default=256, help='size of image height')
parser.add_argument('--img_width', type=int, default=256, help='size of image width')
parser.add_argument('--channels', type=int, default=4, help='number of image channels')
parser.add_argument('--out_channels', type=int, default=1, help='number of output channels')
parser.add_argument('--sample_interval', type=int, default=500, help='interval between sampling of images from generators')
parser.add_argument('--train_patient_idx', type=int, default=5, help='number of patients to train with')
parser.add_argument('--checkpoint_interval', type=int, default=-1, help='interval between model checkpoints')
parser.add_argument('--discrim_type', type=int, default=1, help='discriminator type to use, 0 for normal, 1 for PatchGAN')
parser.add_argument('--test_pats', type=int, default=2, help='number of test patients')
parser.add_argument('--model_name', type=str, default='pycharm_test', help='name of mode')
parser.add_argument('--log_level', type=str, default='info', help='logging level to choose')
parser.add_argument('--c_learning', type=int, default=2, help='whether  or not use curriculum learning framework')
parser.add_argument('--type', type=str, default='T1', help='what sequence to synthesize')
parser.add_argument('--use_tanh', action='store_true', help='use tanh normalization throughout')

opt = parser.parse_args()
print(opt)

if 'info' in opt.log_level:
    logging.basicConfig(level=logging.INFO)
elif 'debug' in opt.log_level:
    logging.basicConfig(level=logging.DEBUG)

# =============================================================================
# Create Training and Validation data loaders
# =============================================================================
if opt.path_prefix == "":
    # parent_path = '/scratch/asa224/asa224/Datasets/BRATS2018/HDF5_Datasets/'
    parent_path = '/local-scratch/anmol/data/{}/HDF5_Datasets/'.format(opt.dataset)
else:
    # notice there's one less asa224 here
    parent_path = os.path.join(opt.path_prefix, 'scratch/asa224/Datasets/{}/HDF5_Datasets/'.format(opt.dataset))

if opt.dataset == 'BRATS2018':
    if opt.grade == 'HGG':
        logger.info('Running on HGG Dataset')
        parent_name = 'preprocessed'
        dataset_name = 'training_data_hgg'
        dataset_type = 'cropped'
        ALL_PATS = 210
        TRAINING_PATS = 190
        VALIDATION_PATS = 10
        TESTING_PATS = 10
        resize_slices = 148
    elif opt.grade == 'LGG':
        logger.info('Running on LGG Dataset')
        parent_name = 'preprocessed'
        dataset_name = 'training_data_lgg'
        dataset_type = 'cropped'
        ALL_PATS = 75
        TRAINING_PATS = 70
        resize_slices = 148
elif opt.dataset == 'BRATS2015':
    logger.info("BRATS2015")
    if opt.grade == 'HGG':
        logger.info('Running on HGG Dataset')
        parent_name = 'preprocessed'
        dataset_name = 'training_data_hgg'
        dataset_type = 'cropped'
        ALL_PATS = 220
        TRAINING_PATS = 200
        VALIDATION_PATS = 10
        TESTING_PATS = 10
        resize_slices = 148
    elif opt.grade == 'LGG':
        logger.info('Running on LGG Dataset')
        parent_name = 'preprocessed'
        dataset_name = 'training_data_lgg'
        dataset_type = 'cropped'
        ALL_PATS = 54
        TRAINING_PATS = 45
        resize_slices = 148
else:
    logger.critical("Invalid dataset name: {}".format(opt.dataset))
    sys.exit(-1)

logger.debug('\tparent_path: \t\t{}'.format(parent_path))
logger.debug('\tparent_name: \t\t{}'.format(parent_name))
logger.debug('\tdataset_name: \t\t{}'.format(dataset_name))
logger.debug('\tdataset_type: \t\t{}'.format(dataset_type))

if resize_slices % opt.batch_size != 0:
    logger.critical("Batch size is not compatible, please change it to be a multiple of {}".format(resize_slices))
    sys.exit(-1)

#DEBUG ONLY
# opt.use_tanh = True

if opt.use_tanh:
    which_normalization = 'tanh'
else:
    which_normalization = None

train_range = list(range(0, opt.train_patient_idx))

n_dataloader, dataloader_for_viz = create_dataloaders(parent_path=parent_path,
                               parent_name=parent_name,
                               dataset_name=dataset_name,
                                dataset_type=dataset_type,
                               load_pat_names=True,
                               load_seg=False,
                               transform_fn=[Resize(size=(opt.img_height, opt.img_width)), ToTensor()],
                               apply_normalization=True,
                               which_normalization=which_normalization,
                               train_range=train_range,
                               resize_slices=resize_slices,
                               get_viz_dataloader=True,
                               num_workers=opt.n_cpu,
                               load_indices=None,
                               dataset=opt.dataset,
                               shuffle=False)

test_patient = []
for k in range(0, opt.test_pats):
    test_patient.append(dataloader_for_viz.getitem_via_index(opt.train_patient_idx + k)) # tehre should be no +1

# if train_pat = 200
# The testing loop will evaluate at train_idx = 199 since the condition is train_idx + 1 == opt.train_patient_idx
# testing patient should start from 200 until 209.

# =============================================================================

# =============================================================================
# Initialize Networks
# =============================================================================
#
# os.makedirs('images/%s' % opt.dataset_name, exist_ok=True)
# os.makedirs('saved_models/%s' % opt.dataset_name, exist_ok=True)

cuda = True if torch.cuda.is_available() else False

# =============================================================================
# Loss functions
# =============================================================================
criterion_GAN = torch.nn.BCELoss() if opt.discrim_type == 0 else torch.nn.MSELoss()
criterion_pixelwise = torch.nn.L1Loss()
mse_fake_vs_real = torch.nn.MSELoss()
# =============================================================================

# Loss weight of L1 pixel-wise loss between translated image and real image
lambda_pixel = 100

# Calculate output of image discriminator (PatchGAN)
patch = (opt.out_channels, opt.img_height//2**4, opt.img_width//2**4)

# Initialize generator and discriminator
if which_normalization == 'tanh':
    generator = GeneratorUNet(in_channels=opt.channels, out_channels=opt.out_channels, with_relu=False, with_tanh=True)
else:
    generator = GeneratorUNet(in_channels=opt.channels, out_channels=opt.out_channels, with_relu=True, with_tanh=False)
discriminator = Discriminator(in_channels=opt.out_channels, out_channels=opt.out_channels, dataset='BRATS2018')

# =============================================================================

# =============================================================================
# Where to save results
# =============================================================================

if opt.path_prefix == "":
    root = '/local-scratch/anmol/results_new/project_880/'
else: # NOT USED
    root = os.path.join(opt.path_prefix, 'rrg_proj_dir/Results/project_880_new/mm_synthesis_gan_results/')
    logger.warning("root: {}".format(root))
    logger.warning('Possible bad value for opt.path_prefix')

model = opt.model_name
if not os.path.isdir(root):
    os.mkdir(root)
if not os.path.isdir(os.path.join(root, model)):
    os.mkdir(os.path.join(root, model))
if not os.path.isdir(os.path.join(root, model, "{}".format(opt.dataset), 'scenario_results')):
    os.makedirs(os.path.join(root, model, "{}".format(opt.dataset), 'scenario_results'))
# =============================================================================

# Optimizers
optimizer_G = torch.optim.Adam(generator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))
optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=opt.lr, betas=(opt.b1, opt.b2))

# Send everything to GPU
if cuda:
    generator = nn.DataParallel(generator.cuda())
    discriminator = nn.DataParallel(discriminator.cuda())
    criterion_GAN.cuda()
    criterion_pixelwise.cuda()
    mse_fake_vs_real.cuda()

# =============================================================================
# Init networks and optimizers
# =============================================================================
if opt.epoch != 0:
    # Load pretrained models
    logger.info('Loading previous checkpoint!')
    generator, optimizer_G = load_checkpoint(generator, optimizer_G, os.path.join(root, opt.model_name,
                                                                                  "{}_param_{}_{}.pkl".format(
                                                                                      'generator', opt.model_name,
                                                                                      1)), pickle_module=pickle)
    discriminator, optimizer_D = load_checkpoint(discriminator, optimizer_D, os.path.join(root, opt.model_name,
                                                                                          "{}_param_{}_{}.pkl".format(
                                                                                              'discriminator',
                                                                                              opt.model_name,
                                                                                              1)), pickle_module=pickle)

else:
    # Initialize weights
    generator.apply(weights_init_normal)
    discriminator.apply(weights_init_normal)

# =============================================================================

# Tensor type
Tensor = torch.cuda.FloatTensor if cuda else torch.FloatTensor


# =============================================================================
#  Training
# =============================================================================

# Book keeping
train_hist = {}
train_hist['D_losses'] = []
train_hist['G_losses'] = []
train_hist['per_epoch_ptimes'] = []
train_hist['total_ptime'] = []
train_hist['test_loss'] = {
    'mse': [],
    'psnr': [],
    'ssim': []
}
# Get the device we're working on.
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Create all scenrios: Total will 15, but remove 0000 and 1111
scenarios = list(map(list, itertools.product([0, 1], repeat=4)))

# Generate new label placeholders for this particular batch
# This is for the G (Changed below)
# label_map = torch.ones((opt.batch_size, 4, opt.img_height, opt.img_width), requires_grad=False).cuda().type(
#     torch.cuda.FloatTensor)

# This is for D (Changed below)
if opt.type == 'T1':
    scenarios = [x for x in scenarios if x[0] == 0]
else:
    scenarios = [x for x in scenarios if x[1] == 0]
label_list = torch.from_numpy(np.ones((opt.batch_size,
                                       patch[0],
                                       patch[1],
                                       patch[2]))).cuda().type(torch.cuda.FloatTensor)

# remove the empty scenario and all available scenario
scenarios.remove([0,0,0,0])
# scenarios.remove([1,1,1,1]) # this is no longer invalid in this case.

# sort the scenarios according to decreasing difficulty. Easy scenarios last, and difficult ones first.
scenarios.sort(key=lambda x: x.count(1))

logger.info("Starting Training")
start_time = time.time()

for epoch in range(opt.epoch, opt.n_epochs, 1):
    D_losses = []
    D_real_losses = []
    D_fake_losses = []
    G_train_l1_losses = []
    G_train_losses = []
    G_losses = []
    synth_losses = []

    # patient: Whole patient dictionary containing image, seg, name etc.
    # x_patient: Just the images of a single patient
    # x_r: Batch of images taken from x_patient according to the batch size specified.
    # x_z: Batch from x_r where some sequences are imputed with noise for input to G
    epoch_start_time = time.time()
    for idx_pat, patient in enumerate(n_dataloader):
        logger.info("Current idx_pat: {}".format(idx_pat))
        # if idx_pat > opt.train_patient_idx:
        #     logger.info("Now testing on patient {}".format(opt.train_patient_idx + 1))
        #     main_path = os.path.join(root, model, 'scenario_results')
        #
        #     fixed_p = os.path.join(root, model, 'scenario_results', 'viz' + "_" + str(epoch + 1))
        #
        #     logger.info("Saving result as {}".format(fixed_p))
        #     status = show_intermediate_results(generator, test_patient, save_path=main_path,
        #                                        all_scenarios=copy.deepcopy(scenarios), epoch=epoch,
        #                                        curr_scenario_range=None,
        #                                        batch_size_to_test=opt.batch_size)
        #     break

        # Put the whole patient in GPU to aid quicker training
        x_patient = patient['image']
        batch_indices = list(range(0, resize_slices, opt.batch_size))

        # this shuffles the 2D axial slice batches for efficient training
        # tag1
        random.shuffle(batch_indices)

        # create batches out of this patient
        for _num, batch_idx in enumerate(batch_indices):
            logger.debug("Patient #{}\nBatch #{}".format(idx_pat, _num))

            logger.debug("\tSplicing batch from x_real")
            x_r = x_patient[batch_idx:batch_idx + opt.batch_size, ...]
            # x_r = x_r.cuda().type(Tensor) # Tensor will be either a cuda dtype or cpu dtype

            # Curriculum Learning: Train with easier cases in the first epochs, then start training on harder ones

            if opt.c_learning == 1:
                # Curriculum Learning: Train with easier cases in the first epochs, then start training on harder ones
                if epoch <= 10:
                    rand_val = torch.Tensor([6])  # last one is the easier, train with that.
                if epoch > 10 and epoch <= 20:
                    rand_val = torch.randint(low=3, high=len(scenarios), size=(1,))
                if epoch > 20:
                    rand_val = torch.randint(low=0, high=len(scenarios), size=(1,))
            elif opt.c_learning == 0:
                rand_val = torch.randint(low=0, high=14, size=(1,))
            elif opt.c_learning == 2:
                # always train with the last scenario, means everything present.
                rand_val = torch.Tensor([6])  # last one is the easier, train with that.


            label_scenario = scenarios[int(rand_val.numpy()[0])]
            logger.debug('\tTraining this batch with Scenario: {}'.format(label_scenario))

            # create a new x_imputed and x_real with this label scenario
            x_z = x_r.clone().cuda()

            if opt.discrim_type == 1:
                label_list_r = torch.from_numpy(
                    np.ones((opt.batch_size, patch[0], patch[1], patch[2]))).cuda().type(
                    torch.cuda.FloatTensor)
            else:
                # This is for D when training on real images (Unchanged)
                label_list_r = torch.from_numpy(np.ones((opt.batch_size, 4))).cuda().type(Tensor)

            # impute noise to the input of G
            for idx, k in enumerate(label_scenario):
                if k == 0:
                    x_z[:, idx, ...] = torch.ones((opt.img_height, opt.img_width), device=device) * -1.0
                    # label_map[:, idx, ...] = 0
                #
                #     # this works with both discriminator types.
                #     if opt.dataset != 'ISLES2015' and opt.dataset != 'BRATS2015':
                #         label_list[:, idx] = 0
                #
                # elif k == 1:
                #     # label_map[:, idx, ...] = 1
                #
                #     # this works with both discriminator types.
                #     if opt.dataset != 'ISLES2015' and opt.dataset != 'BRATS2015':
                #         label_list[:, idx] = 1

            # The output is ALWAYS synthesized, so its zero always. ISLES2015 outputs just FLAIR
            label_list[:, 0] = 0

            # TRAIN GENERATOR G
            logger.debug('\tTraining Generator')
            generator.zero_grad()
            optimizer_G.zero_grad()


            # ADDING A RELU AT THE END OF GENERATOR FOR ISLES2015, to make sure values are positive
            fake_x = generator(x_z)

            if opt.type == "T1":
                SEQ_IDX = 0
            elif opt.type == "T2":
                SEQ_IDX = 1

            # if opt.dataset != 'ISLES2015' and opt.dataset != "BRATS2015":
            #     fake_x = impute_reals_into_fake(x_z, fake_x, label_scenario)
            #
            # if opt.dataset != 'ISLES2015' and opt.dataset != "BRATS2015":
            #     pred_fake = discriminator(fake_x, x_r)
            # else:
                # I may have to unsqueeze
            pred_fake = discriminator(fake_x, x_r[:, SEQ_IDX, ...].unsqueeze(1))

            # G_train_loss = BCE_loss(D_result, label_list)
            # The discriminator should think that the pred_fake is real, so we minimize the loss between pred_fake
            # and label_list_r, ie. make the pred_fake look real, and reducing the error that the discriminator makes
            # when predicting it.

            if pred_fake.size() != label_list_r.size():
                logger.warning('Error!')
                import sys

                sys.exit(-1)

            # fooling the discriminator. We REDUCE this loss value so that it does WELL in predicting pred_fake, and label_list_r
            # not backpropagating into discriminator
            # TODO
            loss_GAN = criterion_GAN(pred_fake.detach(), label_list_r)

            # pixel-wise loss
            loss_pixel = 0
            synth_loss = 0

            # if opt.dataset != 'ISLES2015' and opt.dataset != "BRATS2015":
            #     for idx_curr_label, i in enumerate(label_scenario):
            #         if i == 0:
            #             loss_pixel += criterion_pixelwise(fake_x[:, idx_curr_label, ...], x_r[:, idx_curr_label, ...])
            #
            #             synth_loss += mse_fake_vs_real(fake_x[:, idx_curr_label, ...], x_r[:, idx_curr_label, ...])
            # else:
                # fake_x is already (B, 1, 256, 256)
            loss_pixel += criterion_pixelwise(fake_x, x_r[:, SEQ_IDX, ...].unsqueeze(1).type(Tensor))

            synth_loss += mse_fake_vs_real(fake_x, x_r[:, SEQ_IDX, ...].unsqueeze(1).type(Tensor))
            # logger.debug("Min: {}".format(x_r[:, SEQ_IDX, ...].min()))
            # logger.debug("Max: {}".format(x_r[:, SEQ_IDX, ...].max()))
            # logger.debug("Mean: {}".format(x_r[:, SEQ_IDX, ...].mean()))

            # variable that sets the relative importance to loss_GAN and loss_pixel
            lam = 0.9
            G_train_total_loss = (1 - lam) * loss_GAN + lam * loss_pixel

            G_train_total_loss.backward()
            optimizer_G.step()

            # save the losses
            G_train_l1_losses.append(loss_pixel.data[0])
            G_train_losses.append(loss_GAN.data[0])
            G_losses.append(G_train_total_loss.data[0])
            synth_losses.append(synth_loss.data[0])

            # TRAIN DISCRIMINATOR D
            # this takes in the real x as X-INPUT and real x as Y-INPUT
            logger.debug('\tTraining Discriminator')
            discriminator.zero_grad()
            optimizer_D.zero_grad()

            # real loss
            # if opt.dataset == 'ISLES2015' or opt.dataset == "BRATS2015":
            pred_real = discriminator(
                    x_r[:, SEQ_IDX, ...].unsqueeze(1),
                    x_r[:, SEQ_IDX, ...].unsqueeze(1))
            # else:
            #     pred_real = discriminator(x_r, x_r)

            loss_real = criterion_GAN(pred_real, label_list_r)

            # fake loss
            # fake_x = generator(x_z, label_map)s
            fake_x = generator(x_z)

            # if opt.dataset != 'ISLES2015' and opt.dataset != "BRATS2015":
            #     fake_x = impute_reals_into_fake(x_z, fake_x.detach(), label_scenario)
            #
            # if opt.dataset != 'ISLES2015' and opt.dataset != "BRATS2015":
            #     pred_fake = discriminator(fake_x.detach(), x_r)
            # else:
            sh = fake_x.shape
            pred_fake = discriminator(fake_x.detach(),
                                      x_r[:, SEQ_IDX, ...].unsqueeze(1))

            loss_fake = criterion_GAN(pred_fake, label_list)

            D_train_loss = 0.5 * (loss_real + loss_fake)

            # for printing purposes
            D_real_losses.append(loss_real.data[0])
            D_fake_losses.append(loss_fake.data[0])
            D_losses.append(D_train_loss.data[0])

            D_train_loss.backward()
            optimizer_D.step()

            logger.info(" E [{}/{}] P #{} ".format(epoch, opt.n_epochs,
                                                              idx_pat) + 'B [%d/%d] - loss_d: [real: %.5f, fake: %.5f, comb: %.5f], loss_g: [gan: %.5f, l1: %.5f, comb: %.5f], synth_loss_mse(ut): %.5f' % (
                            (_num + 1), resize_slices // opt.batch_size, torch.mean(torch.FloatTensor(D_real_losses)),
                            torch.mean(torch.FloatTensor(D_fake_losses)),
                            torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_train_losses)),
                            torch.mean(torch.FloatTensor(G_train_l1_losses)), torch.mean(torch.FloatTensor(G_losses)),
                        torch.mean(torch.FloatTensor(synth_losses))))
            # Check if we have trained with exactly opt.train_patient_idx patients (if opt.train_patient_idx is 10, then idx_pat will be 9, so this condition will evaluate to true
        if idx_pat + 1 == opt.train_patient_idx:
            logger.info('Testing on test set for this fold')
            main_path = os.path.join(root, model, "{}".format(opt.dataset), 'scenario_results')

            logger.info("Saving results at {}".format(main_path))

            generator.eval()

            logger.info("Calculating metric on test set")
            result_dict_test = calculate_metrics(generator, test_patient, save_path=main_path,
                                                 all_scenarios=copy.deepcopy(scenarios), epoch=epoch,
                                                 curr_scenario_range=[6, 7],
                                                 batch_size_to_test=1, dataset="BRATS2015", # keep this to B15 to maintain compatibility
                                                 seq_type=opt.type)

            logger.info("\t\tTesting Performance Numbers")
            printTable(result_dict_test)
            gc.collect()

            logger.info("Writing detailed visualizations for each scenario")
            status = show_intermediate_results(generator, test_patient, save_path=main_path,
                                               all_scenarios=copy.deepcopy(scenarios), epoch=epoch,
                                               curr_scenario_range=[6, 7],
                                               batch_size_to_test=opt.batch_size, seq_type=opt.type,
                                               dataset="BRATS2015") # keep this to B15 to maintain

            generator.train()
            gc.collect()
            break

    epoch_end_time = time.time()
    per_epoch_ptime = epoch_end_time - epoch_start_time

    print(
        '[%d/%d] - ptime: %.2f, loss_d: [real: %.5f, fake: %.5f, comb: %.5f], loss_g: [gan: %.5f, l1: %.5f, comb: %.5f], '
        'synth_loss_mse(ut): %.5f' % (
        (epoch + 1), opt.n_epochs, per_epoch_ptime, torch.mean(torch.FloatTensor(D_real_losses)),
        torch.mean(torch.FloatTensor(D_fake_losses)),
        torch.mean(torch.FloatTensor(D_losses)), torch.mean(torch.FloatTensor(G_train_losses)),
        torch.mean(torch.FloatTensor(G_train_l1_losses)), torch.mean(torch.FloatTensor(G_losses)),
        torch.mean(torch.FloatTensor(synth_losses))))

    # Checkpoint the models

    gen_state_checkpoint = {
            'epoch': epoch + 1,
            'arch': opt.model_name,
            'state_dict': generator.state_dict(),
            'optimizer' : optimizer_G.state_dict(),
        }

    des_state_checkpoint = {
        'epoch': epoch + 1,
        'arch': opt.model_name,
        'state_dict': discriminator.state_dict(),
        'optimizer': optimizer_D.state_dict(),
    }

    save_checkpoint(gen_state_checkpoint, os.path.join(root, model, 'generator_param_{}_{}.pkl'.format(model, epoch + 1)),
                    pickle_module=pickle)

    save_checkpoint(des_state_checkpoint,
                    os.path.join(root, model, 'discriminator_param_{}_{}.pkl'.format(model, epoch + 1)),
                    pickle_module=pickle)

    with open(os.path.join(root, model, "{}".format(opt.dataset),
                           'result_dict_test_epoch_{}.pkl'.format(epoch)), 'wb') as f:
        pickle.dump(result_dict_test, f)

    logger.info('[Testing] num_pats: {}, mse: {:.5f}, psnr: {:.5f}, ssim: {:.5f}'.format(
        opt.test_pats,
        result_dict_test['mean']['mse'],
        result_dict_test['mean']['psnr'],
        result_dict_test['mean']['ssim']
    ))

    train_hist['D_losses'].append(torch.mean(torch.FloatTensor(D_losses)))
    train_hist['G_losses'].append(torch.mean(torch.FloatTensor(G_losses)))

    train_hist['test_loss']['mse'].append(result_dict_test['mean']['mse'])
    train_hist['test_loss']['psnr'].append(result_dict_test['mean']['psnr'])
    train_hist['test_loss']['ssim'].append(result_dict_test['mean']['ssim'])

    train_hist['per_epoch_ptimes'].append(per_epoch_ptime)

    end_time = time.time()
    total_ptime = end_time - start_time
    train_hist['total_ptime'].append(total_ptime)

    print("Avg one epoch ptime: %.2f, total %d epochs ptime: %.2f" % (
    torch.mean(torch.FloatTensor(train_hist['per_epoch_ptimes'])), opt.n_epochs, total_ptime))

with open(os.path.join(root, model, 'train_hist.pkl'), 'wb') as f:
    pickle.dump(train_hist, f)
