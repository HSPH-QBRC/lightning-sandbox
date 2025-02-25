import sys
import os
from pathlib import Path

import hydra
from omegaconf.errors import ConfigAttributeError
from omegaconf import OmegaConf
import pandas as pd
import torch

from custom_lightning_modules import load_pl_module
from data_modules import load_dataset
from models import load_model

torch.set_float32_matmul_precision('medium')

HPARAMS_FILE = 'hparams.yaml'


class FeatureExtractor(torch.nn.Module):
    '''
    We effectively wrap the existing model (a nn.Module)
    with this so we can grab the values from a particular 
    layer/module of the model
    '''

    def __init__(self, original_model, layer_name):
        super().__init__()
        self.original_model = original_model
        self.layer_name = layer_name

    def forward(self, x):
        for name, module in self.original_model.model.net._modules.items():
            x = module(x)  # forward pass through each layer
            if name == self.layer_name:
                return x
        print('Warning: Never passed through intended layer.')


def extract_features(feature_extractor, data_module):

    with torch.no_grad():
        output_feature_mtx = []
        img_ids = []
        dataloader = data_module.predict_dataloader()
        for batch_ndx, batch in enumerate(dataloader):
            img_batch = batch[0]
            img_ids.extend(batch[1])
            fl = feature_extractor(img_batch) # (batch_size, n, 1, 1)
            fl = fl.squeeze() # (batch_size, n)
            output_feature_mtx.append(fl)

    output_feature_mtx = torch.concat(output_feature_mtx)
    df = pd.DataFrame(output_feature_mtx)
    df.columns = [f'f{x}' for x in range(df.shape[1])]
    df = pd.concat([
        pd.Series(img_ids, name='img_id'),
        df], axis=1)
    return df


def load_model_weights(model, checkpoint_data, prefix='model'):
    '''
    Loads the weights from training into the provided model. 
    '''
    # The model weights have a 'prefix' which we need to strip
    weights = {k[len(prefix)+1:]: v for k, v in checkpoint_data["state_dict"].items() if k.startswith(f"{prefix}.")}
    model.load_state_dict(weights)
    model.eval()


@hydra.main(version_base='1.3', config_path="../../conf", config_name="config")
def main(cfg):
    
    # load the config from the trained model
    try:
        train_output_dir = Path(cfg.train_output_dir)
    except ConfigAttributeError:
        print('Need to specify the config YAML from training with the'
              ' "++train_output_dir" argument')
        sys.exit(1)

    # in the train_output_dir, we expect a file named after
    # the HPARAMS variable
    train_config_file = train_output_dir / HPARAMS_FILE
    if not train_config_file.exists():
        print(f'We expected a yaml file at {train_config_file}.')
        sys.exit(1)

    train_config = OmegaConf.load(train_config_file)['cfg']

    # merge the current args/params with those from training:
    final_cfg = OmegaConf.merge(train_config, cfg)

    # while hydra will automatically save the original cfg object
    # in the output directory, we explicitly save the fully resolved
    # config here, so we don't have to look at the training config
    # AND the config provided by the command line args.
    final_cfg_path = Path(os.getcwd()) / '.hydra/final_config.yaml'
    OmegaConf.save(final_cfg, final_cfg_path)

    try:
        checkpoint_file = final_cfg.checkpoint_file
    except ConfigAttributeError:
        # if not specified, assume a default checkpoint filename
        checkpoint_file = 'last.ckpt'

    ckpt_path = train_output_dir / 'checkpoints' / checkpoint_file
    if not ckpt_path.exists():
        print('Could not find a checkpoint file to load the model.'
              f' Tried: {ckpt_path}. You can specify "++checkpoint_file"'
              f' as the name in {ckpt_path.parent}')
        sys.exit(1)
        
    # load the checkpoint so we can grab the weights
    checkpoint_data = torch.load(ckpt_path)

    # Load the Pytorch lightning module specified in the config
    selected_model = load_model(final_cfg.model)
    load_model_weights(selected_model, checkpoint_data)
    pl_module = load_pl_module(final_cfg, selected_model)

    datamodule = load_dataset(final_cfg.dataset)
    datamodule.prepare_data()
    datamodule.setup(stage='predict')

    # Wrap the model
    try:
        layer_name = final_cfg.layer_name
    except ConfigAttributeError:
        print('Need to know the layer to extract from using the'
              ' "++layer_name" argument')
        sys.exit(1)

    feature_extractor = FeatureExtractor(pl_module, layer_name)

    feature_df = extract_features(feature_extractor, datamodule)

    try:
        output_name = final_cfg.output_name
    except ConfigAttributeError:
        output_name = 'results.tsv'
    feature_df.to_csv(output_name, sep='\t', index=False)


if __name__ == "__main__":
    main()