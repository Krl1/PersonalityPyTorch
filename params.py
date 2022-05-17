RANDOM_SEED = 42


class WandbConfig:
    project_name = 'personality_crop_connected'
    run_name = '1e-5_True_0.0_0.0'
    save_dir = '.'
    entity = 'krl1'


class CreateDataConfig:
    connect = True
    test_size_ratio = 0.09
    classification = True
    Y_threshold = 0.5


class LocationConfig:
    labels = 'dataset/'
    raw_data = 'dataset/full_images/'
    crop_data = 'dataset/crop_images/'
    new_data = './new_data/'
    checkpoints_dir = 'model/checkpoints'
    best_model = 'model/best.pt'
    
    
class TrainingConfig:
    batch_size = 128
    epochs = 50
    gpus = 1
    deterministic = True
    accumulate_grad_batches = 1
    patience = 5
    lr = 1e-5
    batch_norm = False
    negative_slope = 0.0
    dropout = 0.0
