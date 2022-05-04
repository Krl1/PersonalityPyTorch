RANDOM_SEED = 42


class WandbConfig:
    project_name = 'personality'
    run_name = 'cnn4_pytorch_connected'
    save_dir = '.'
    entity = 'krl1'


class LocationConfig:
    checkpoints_dir = 'model/checkpoints'
    best_model = 'model/best.pt'
    data = 'data_connected'
    train_data = 'data_connected/train'
    test_data = 'data_connected/test'
    
    
class TrainingConfig:
    batch_size = 128
    epochs = 100
    gpus = 1
    deterministic = True
    accumulate_grad_batches = 2
    patience = 15
    lr = 1e-6
