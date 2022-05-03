RANDOM_SEED = 42


class WandbConfig:
    project_name = 'personality'
    run_name = 'cnn4_pytorch_test'
    save_dir = '.'
    entity = 'krl1'


class LocationConfig:
    checkpoints_dir = 'model/checkpoints'
    best_model = 'model/best.pt'
    data = 'data'
    train_data = 'data/train'
    test_data = 'data/test'
    
    
class TrainingConfig:
    batch_size = 128
    epochs = 100
    gpus = 1
    deterministic = True
    accumulate_grad_batches = 2
    patience = 3
    lr = 10e-6