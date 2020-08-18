import os
from human_body_prior.tools.omni_tools import makepath, log2file
from human_body_prior.data.prepare_data import prepare_vposer_datasets
from datetime import datetime

expr_code = datetime.now().strftime("%d %m %Y %H:%M:%S")

amass_dir = r'/content/drive/My Drive/LAZAR/AMASS'

vposer_datadir = makepath('prepared/%s' % (expr_code))

logger = log2file(os.path.join(vposer_datadir, '%s.log' % (expr_code)))
logger('[%s] Preparing data for training VPoser.'%expr_code)

amass_splits = {
    'vald': ['HumanEva', 'MPIHDM05', 'SFU', 'MPImosh'],
    'test': ['Transitions_mocap', 'SSMsynced'],
    'train': ['CMU', 'MPILimits', 'TotalCapture', 'EyesJapanDataset', 'KIT', 'BMLmovi', 'BMLrub', 'EKUT', 'TCDhandMocap', 'ACCAD']
}
amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['vald'])))

prepare_vposer_datasets(amass_splits, amass_dir, vposer_datadir, logger=logger)