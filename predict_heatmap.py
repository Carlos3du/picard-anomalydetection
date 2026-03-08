##############################
# code for predicting anomaly detection heatmaps of images
# using pluralistic image completion.
##############################

### IMPORTS
import os
from argparse import ArgumentParser

from utils import *
from modules import *
from heatmapping import *
from eval import *

# torch
import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn.functional as F

# other libs
from datetime import datetime
import random
from random import sample
import matplotlib
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from time import time

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='configs/config.yaml',
                    help="configuration")
parser.add_argument('--seed', type=int, default=1337,
                    help="manual random seed")
parser.add_argument('--checkpoint_dir', type=str,
                    help="path to saved inpainter model checkpoint directory")
parser.add_argument('--checkpoint_iter', type=int,
                    help="iteration number of saved model checkpoint")

def main():
    args = parser.parse_args()
    config = get_config(args.config)

    ############################################################
    ### (1) GPU setup
    ############################################################
    cuda = config['cuda']
    device_check = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print('running on {}'.format(device_check))

    # set which devices CUDA sees
    device_ids = config['gpu_ids'] # indices of devices for models, data and otherwise
    if cuda:
        os.environ['CUDA_VISIBLE_DEVICES'] = ','.join(str(i) for i in device_ids)

    # all devices are then indexed from this set
    model_device = 0

    # set random seed
    seed = args.seed
    random.seed(seed)
    torch.manual_seed(seed)
    if cuda:
        torch.cuda.manual_seed_all(seed)

    ############################################################
    ### (2) model setup
    ############################################################
    # choose model and dataset we'll be working with
    model_type = 'dropout'
    dataset_name = config['dataset_name']

    PIL_img_format = 'L' if (config["test"]["patch_shape"][-1] == 1) else 'RGB'
    completion_img_size = config["test"]["patch_shape"][0]
    # ^ side size of image to be completed (may be just part of larger heatmap image)

    # load utils
    normalize_img = load_img_normalizer(model_type)

    # hyperparameter and checkpoint setup 
    hyperparams = {
        'dropout' : {
            'p_dropout' : config["test"]["droprate"]
        },
    }

    checkpoints = {
        'dropout' : {
            'gen' : args.checkpoint_dir,
            'dis' : args.checkpoint_dir,
            'iter' : args.checkpoint_iter
        },
    }

    # load inpainter and completion feature extractor
    inpainter = load_multi_inpainter(
        model_type, 
        checkpoints[model_type], 
        hyperparams[model_type], 
        device_ids,
        dropoutmodel_config=args.config
    )

    feature_extractor = load_inpainting_feature_extractor(
        model_type, 
        checkpoints[model_type], 
        hyperparams[model_type], 
        device_ids,
        dropoutmodel_config=args.config
    )

    # heatmapping settings
    # visualization and analysis settings
    save_heatmap_data = config["test"]["save_heatmap_data"]
    save_heatmap_plots = config["test"]["save_heatmap_plots"] 

    save_progressive_heatmap = config["test"]["save_progressive_heatmap"]
    log_compute_times = config["test"]["log_compute_times"]

    # heatmapping parameters
    mask_size = config["test"]["mask_shape"][0]
    window_size = config["test"]["patch_shape"][0]
    window_stride = config["test"]["patch_stride"]
    heatmap_M_inpaint = config["test"]["heatmap_M_inpaint"]
    heatmap_metrics = config["test"]["heatmap_metrics"]
    parallel_batchsize = config["test"]["parallel_batchsize"]

    # misc scoring settings
    only_check_nonblack_pixels = config["test"]["only_check_nonblack_pixels"]

    # logger
    log_dir = 'test_logs'
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    logger = Logger('test', log_dir, heatmap_metrics)
    dt_now = datetime.now()

    ############################################################
    ### (3) heatmap generation
    ############################################################

    test_data_fnames = []
    for root, dirs, files in os.walk(config["test_data_path"]):
        for f in files:
            # Garante que só pega arquivos de imagem e ignora outros tipos
            if f.lower().endswith(('.png', '.dcm', '.jpg', '.jpeg')):
                test_data_fnames.append(os.path.join(root, f))

    with torch.no_grad():
        for test_data in test_data_fnames:
            # 1) load image
            img = pil_loader(test_data, img_format=PIL_img_format)
            img = transforms.ToTensor()(img)
            # don't normalize at the image level: will normalize at the patch level
            img = img.unsqueeze(dim=0)
            
            # plot bbox on img
            img = img.cpu()
            print(test_data)
            #show_images(img, custom_figsize=(10, 14))
            img = img.cuda()
            
            ignore_mask = None
            if only_check_nonblack_pixels:
                print('ONLY CHECKING NONBLACK PIXELS')
                # mask of size image; True where there are pixels that 
                # we don't want to include in evaluation
                ignore_mask = (normalize_img(img) == -1.).cpu()

            # 2) generate heatmaps
            tin = time()
            heatmaps = generate_anomaly_heatmap_slidingwindow_PARALLEL(
                img, 
                inpainter,
                feature_extractor,
                metrics=heatmap_metrics,                                                   
                mask_size=mask_size,
                window_size=window_size,
                window_stride=window_stride,            
                M_inpaint=heatmap_M_inpaint,
                heatmap_batch_size=parallel_batchsize,
                heatmap_type='nonaveraged',
                img_normalizer = normalize_img,
                save_progressive_heatmap = save_progressive_heatmap
            )
            tout = time()
            
            print('time to create heatmap = {} sec'.format(tout - tin))
            
            # plot and save heatmap data and images
            # plot and save heatmap data and images
            for heatmap_metric in heatmap_metrics:
                
                import re
                
                # 1. Captura quem é o paciente atual e o estudo logo no início
                match_paciente = re.search(r'(DBT-P\d+)', test_data)
                match_estudo = re.search(r'(DBT-S\d+)', test_data)
                
                paciente = match_paciente.group(1) if match_paciente else "SemPaciente"
                estudo = match_estudo.group(1) if match_estudo else "SemEstudo"
                
                # 2. Cria as pastas base e ADICIONA a pasta do paciente no final!
                savedir = os.path.join('heatmaps', dataset_name, model_type, dt_now.strftime("%m-%d-%Y_%H-%M-%S"))
                
                savedir_maps = os.path.join(savedir, 'data', paciente)
                savedir_plots = os.path.join(savedir, 'plots', paciente)
                
                for path in [savedir_maps, savedir_plots]:
                    os.makedirs(path, exist_ok=True) # exist_ok=True evita erros se a pasta já existir
                        
                # 3. Define o nome do arquivo de forma inteligente (como combinamos antes)
                nome_base = os.path.basename(test_data).replace('.png', '').replace('.dcm', '')
                
                filename = f"{paciente}_{estudo}_{nome_base}_{heatmap_metric}_{heatmap_M_inpaint}_{hyperparams['dropout']['p_dropout']}_{mask_size}_{window_size}_{window_stride}.pt"
                
                # 4. Salva o arquivo de dados (.pt) dentro da pasta do paciente
                filename_map = os.path.join(savedir_maps, filename)
                if save_heatmap_data:
                    torch.save(heatmaps[heatmap_metric], filename_map)

         

                # plot heatmap
                fig, ax = plt.subplots(figsize=(10, 14))
                im = ax.imshow((heatmaps[heatmap_metric]).cpu(), cmap=plt.cm.hot, interpolation='none') 
                cbar = fig.colorbar(im, extend='max')
                title = 'anomaly metric: {}\nM={}, p={}, mask size = {}\nwindow size = {}, window stride = {}'.format(
                                                    heatmap_metric,
                                                    heatmap_M_inpaint, 
                                                    hyperparams['dropout']['p_dropout'],
                                                    mask_size,
                                                    window_size,
                                                    window_stride,
                )
                plt.title(title, fontsize=20)
                
                # save heatmap plot 'data_new/test/cancer/val_DBT-P01700_DBT-S01353_lmlo_Cancer_0.png'
                filename_img = filename.replace('.pt', '.png')
                filename_img = os.path.join(savedir_plots, filename_img)
                if save_heatmap_plots:
                    plt.savefig(fname=filename_img, bbox_inches = 'tight')
                plt.close('all')
            
            # log heatmaps on image
            log_hyperparams = [window_stride, window_size, mask_size, 
                            heatmap_M_inpaint, parallel_batchsize, 'nonaveraged', 
                            hyperparams['dropout']['p_dropout']]
            if log_compute_times:
                logger.write_msg('heatmap compute time on {} GPUs = {}\n'.format(len(device_ids), tout-tin))

if __name__ == '__main__':
    main()