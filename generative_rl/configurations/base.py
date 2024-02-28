from easydict import EasyDict

config = EasyDict(
    train = dict(
        env = dict(
            env_id = 'HalfCheetah-v2',
            seed = 0,
        ),
        dataset = dict(
            dataset_path = 'd4rl/halfcheetah-medium-v0',
            batch_size = 64,
            num_workers = 4,
        ),
        model = dict(
            policy = dict(
                hidden_sizes = [256, 256],
                activation = 'relu',
                output_activation = 'relu',
                output_scale = 5.0,
            ),
            QNetwork = dict(
                hidden_sizes = [256, 256],
                activation = 'relu',
                output_activation = 'relu',
            ),
            Q_t_function = dict(
                hidden_sizes = [256, 256],
                activation = 'relu',
                output_activation = 'relu',
            ),
        ),
        parameter = dict(
            policy_lr = 1e-3,
            q_lr = 1e-3,
            q_t_lr = 1e-3,
            num_train_steps_per_epoch = 1000,
            num_train_epochs = 100,
            num_eval_steps_per_epoch = 1000,
            num_eval_epochs = 10,
        ),
    ),
    deploy = dict(
        num_deploy_steps = 1000,
    ),
)

