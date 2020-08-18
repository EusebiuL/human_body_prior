import os
from human_body_prior.tools.omni_tools import makepath, log2file
from human_body_prior.data.prepare_data import prepare_vposer_datasets

expr_code = 'uniqueid1'

amass_dir = '/Users/eusebiu/Desktop/AMAS'

vposer_datadir = makepath('prepared/%s' % (expr_code))

logger = log2file(os.path.join(vposer_datadir, '%s.log' % (expr_code)))
logger('[%s] Preparing data for training VPoser.'%expr_code)

amass_splits = {
    'vald': ['HumanEva', 'MPI_HDM05', 'SFU', 'MPI_mosh'],
    'test': ['Transitions_mocap', 'SSM_synced'],
    'train': ['CMU', 'MPI_Limits', 'TotalCapture', 'Eyes_Japan_Dataset', 'KIT', 'BMLmovi', 'BMLrub', 'EKUT', 'TCD_handMocap', 'ACCAD']
}
amass_splits['train'] = list(set(amass_splits['train']).difference(set(amass_splits['test'] + amass_splits['vald'])))

prepare_vposer_datasets(amass_splits, amass_dir, vposer_datadir, logger=logger)