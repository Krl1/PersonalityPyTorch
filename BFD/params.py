RANDOM_SEED = 42


class WandbConfig:
    project_name = 'personality_crop_connected'
    run_name = 'final_run'
    save_dir = '.'
    entity = 'krl1'


class CreateDataConfig:
    connect = True
    test_size_ratio = 0.2
    classification = True
    Y_threshold = 0.5


class LocationConfig:
    labels = '../data/BFD/'
    images = '../data/BFD/images/'
    crop_images = '../data/BFD/crop_images/'
    data = '../data/BFD/'
    checkpoints_dir = '../model/BFD/checkpoints'
    best_model = '../model/BFD/best.pt'
    
    
class TrainingConfig:
    batch_size = 32
    epochs = 50
    gpus = 1
    deterministic = True
    accumulate_grad_batches = 1
    patience = 10
    lr = 1e-4
    batch_norm = False
    negative_slope = 0.0
    dropout = 0.3
