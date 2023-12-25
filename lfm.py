import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

from lightfm import LightFM


def encode_virus_features(data, virus_index, feature_field, feature_index=None, drop_invalid=True):
    """
    Constructs a sparse binary feature matrix (one-hot encoding of features) based on the input dataframe and a mapping
    from columns of the matrix back into the original features of viruses.

    Parameters
    ----------
    data : pd.DataFrame
        Input dataframe.
    virus_index : pd.Index
        Index (pandas.index) which provides mapping from the matrix raoows to the initial virus_id.
    feature_field : str
        The name of the feature (column label).
    feature_index : pd.Index, optional
        The corresponding feature index, by default None (meaning, it will be generated).
    drop_invalid : bool, optional
        Whether to drop invalid features not present in the provided feature index or raise a ValueError.
        True by default (discard inconsistent features).

    Returns
    -------
    Tuple[csr_matrix, pd.Index]
        A sparse binary feature matrix and the corresponding feature index
        that maps columns of the matrix to the corresponding feature values.
    """
    virus_data = data.loc[data['virus_id'].isin(virus_index)]
    if feature_index is None:
        virus_features, virus_feature_idx = pd.factorize(
            virus_data[feature_field], sort=True
        )
        virus_feature_idx.name = feature_field
    else:
        virus_feature_idx = feature_index
        virus_features = feature_index.get_indexer_for(virus_data[feature_field])
        valid_features_mask = (virus_features >= 0)
        if not valid_features_mask.all():
            if drop_invalid:
                virus_features = virus_features[valid_features_mask]
                virus_data = virus_data.loc[valid_features_mask]
            else:
                raise ValueError(f'Some {feature_field} features are not present in the provided index!')

    virus_feature_matrix = csr_matrix(
        (
            np.ones(len(virus_features)),
            (
                virus_index.get_indexer_for(virus_data['virus_id']),
                virus_features
            )
        ),
        shape = (len(virus_index), len(virus_feature_idx))
    )
    return virus_feature_matrix, virus_feature_idx


def build_lfm_model(config, data, data_description, early_stop_config=None, iterator=None):
    """
    Builds a LightFM model using the given configuration, data and data description.

    Parameters
    ----------
    config : dict
        A dictionary containing the configuration for the model. It must contain the following keys:
        'no_components', 'max_sampled', 'loss', 'learning_schedule', 'user_alpha' and 'item_alpha'.
    data : sparse matrix of interactions in COO format of shape (n_users, n_items)
        The training data.
    data_description : dict
        A dictionary containing information about the data. It must contain the following keys:
        'interactions', 'user_features' and 'item_features'.
    early_stop_config : dict, optional (default=None)
        A dictionary containing early stopping configuration. If not provided, default values will be used.

    Returns
    -------
    model : LightFM object The trained LightFM model.
    """
    # the model
    model = LightFM(
        no_components = config['no_components'],
        loss = config['loss'],
        learning_schedule = config['learning_schedule'],
        learning_rate = config.get('learning_rate', 0.05), # default as in the docs
        user_alpha = config['user_alpha'],
        item_alpha = config['item_alpha'],
        max_sampled = config['max_sampled'],
        random_state = config.get('random_state', None)
    )
    # early stoppping configuration
    es_config = check_early_stop_config(early_stop_config)

    # training
    if iterator is None:
        iterator = lambda x: x
    for epoch in iterator(range(config['max_epochs'])):
        try:
            train_lfm_epoch(epoch, model, data, data_description, es_config)
        except StopIteration:
            break
    return model


def check_early_stop_config(early_stop_config):
    """
    Validates the early stop configuration and returns a config dictionary.

    Parameters
    ----------
    early_stop_config : dict, optional
        Dictionary containing the early stop configuration.

    Returns
    -------
    es_dict : dict
        Dictionary containing the early stop configuration, or a dictionary
        with 'stop_early' set to False if no valid configuration is provided.
    """
    if early_stop_config is None:
        early_stop_config = {}
    try:
        es_dict = {
            'early_stopper': early_stop_config['evaluation_callback'],
            'callback_interval': early_stop_config['callback_interval'],
            'holdout': early_stop_config['holdout'],
            'data_description': early_stop_config['data_description'],
            'stop_early': True
        }
    except KeyError: # config is invalid, doesn't contain required keys
        es_dict = {'stop_early': False} # disable early stopping
    return es_dict


def train_lfm_epoch(
    epoch, model, train, data_description, es_config,
):
    """
    Train a LightFM model for a single epoch. Optionally validate the model
    and raise StopIteration if the early stopping condition is met.

    Parameters
    ----------
    epoch : int
        The current epoch number.
    model : LightFM object
        The LightFM model to be trained.
    train : scipy.sparse matrix
        The training data matrix in COO format.
    data_description : dict
        A dictionary containing the user and item feature matrices.
    es_config : dict
        A dictionary containing the early stopping configuration parameters.

    Returns
    -------
    None

    Raises
    ------
    StopIteration: If the early stopping condition is met.
    """
    model.fit_partial(
        train,
        user_features = data_description['user_features'],
        item_features = data_description['item_features'],
        epochs = 1
    )
    if not es_config['stop_early']:
        return

    metrics_check_interval = es_config['callback_interval']
    if (epoch+1) % metrics_check_interval == 0:
        # evaluate model and raise StopIteration if early stopping condition is met
        early_stopper_call = es_config['early_stopper']
        early_stopper_call(
            epoch,
            model,
            es_config['holdout'],
            es_config['data_description']
        )