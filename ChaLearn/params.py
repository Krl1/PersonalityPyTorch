RANDOM_SEED = 42


class WandbConfig:
    project_name = 'personality_crop_connected'
    run_name = 'final_run'
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
    crop_data = 'dataset/my_crop_images/'
    new_data = './my_data/'
    old_data = './old_data/'
    checkpoints_dir = 'model/checkpoints'
    best_model = 'model/best.pt'
    shuffle_data = './shuffle_data/'
    
    
class TrainingConfig:
    batch_size = 2
    epochs = 50
    gpus = 1
    deterministic = True
    accumulate_grad_batches = 1
    patience = 10
    lr = 1e-4
    batch_norm = False
    negative_slope = 0.0
    dropout = 0.3
