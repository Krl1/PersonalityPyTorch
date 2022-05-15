RANDOM_SEED = 42


class WandbConfig:
    project_name = 'personality_connected'
    run_name = 'cnn4_1e-5'
    save_dir = '.'
    entity = 'krl1'


class CreateDataConfig:
    connect = True
    test_size_ratio = 0.09
    classification = False
    Y_threshold = 0.5


class LocationConfig:
    raw_data = 'dataset/'
    new_data = './new_data/'
    checkpoints_dir = 'model/checkpoints'
    best_model = 'model/best.pt'
    
    
class TrainingConfig:
    batch_size = 32
    epochs = 100
    gpus = 1
    deterministic = True
    accumulate_grad_batches = 1
    patience = 5
    lr = 1e-5
    batch_norm = True
    negative_slope = 0.0
    sigmoid = False
    max_pool_ceil_mode = False
