RANDOM_SEED = 42


class WandbConfig:
    project_name = 'personality'
    run_name = 'cnn4_pytorch_normalize'
    save_dir = '.'
    entity = 'krl1'


class LocationConfig:
    checkpoints_dir = 'model/checkpoints'
    best_model = 'model/best.pt'
    data = 'small_data'
    train_data = 'small_data'
    test_data = 'small_data'
    
    
class TrainingConfig:
    batch_size = 2
    epochs = 10000
    gpus = 1
    deterministic = True
    accumulate_grad_batches = 1
    patience = 10
    lr = 1e-6
