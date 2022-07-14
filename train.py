from asyncio.log import logger
import torch
from torch import nn

import colossalai
from colossalai.trainer import Trainer, hooks
from colossalai.utils import MultiTimer
from colossalai.nn.metric import Accuracy
from colossalai.logging import disable_existing_loggers

import wandb

#from lamda_pytorch.config.config import CFG
from lamda_pytorch.config.config import CFG

from lamda_pytorch.stream_dataloader import stream_dataloaders

from lamda_pytorch.lamda_pytorch import lamda_model
from lamda_pytorch.utils.utils import LaMDA_Loss, AutoregressiveWrapper

def LaMDA_Trainer(cfg: CFG):
    assert torch.cuda.is_available()
    disable_existing_loggers()

    parser = colossalai.get_default_parser()

    parser.add_argument(
        '--use_trainer',
        action='store_true',
        help='whether to use trainer'
    )

    args = parser.parse_args()

    if args.use_zero:
        pass
    else:
        colossalai.launch_from_torch(
            config='.lamda_pytorch/config/colossal_config.py', 
            seed = cfg.seed
        )

    # LaMDA model
    model = lamda_model()
    model = AutoregressiveWrapper(model)

    # setup dataloaders
    if cfg.use_huggingface == True:
        train_dataloader, test_dataloader = stream_dataloaders(cfg)

    # loss function
    loss_fn = LaMDA_Loss()

    # optimizer function

    optimizer = torch.optim.Adam(
        model.parameters(), 
        lr = cfg.lr
    )

    # initialze model, optimizer, criterion, and data loaders

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model,
        optimizer,
        loss_fn,
        train_dataloader,
        test_dataloader
    )

    if cfg.use_wandb == True:

        # initialize Weights and Biases Logging
        wandb.init(project = cfg.project_name)

        engine.train()
        for step, batch in enumerate(train_dataloader):

            inputs, labels = batch['inputs'].cuda(), batch['labels'].cuda()
            
            engine.zero_grad()
            outputs = engine(inputs)

            train_loss = engine.loss_fn(outputs, labels)
            wandb.log({"train_loss": train_loss})

            engine.backward(train_loss)
            engine.step()
            wandb.log({"step": step})
            
            engine.eval()
            for step, batch in enumerate(test_dataloader):
                inputs, labels = batch['inputs'].cuda(), batch['labels'].cuda()

                with torch.no_grad():
                    outputs = engine(inputs)
                    test_loss = engine.loss_fn(outputs, labels)
                    wandb.log({"test_loss": test_loss})
                
                engine.backward(test_loss)
                engine.step()

        wandb.alert(
            title = 'Training Complete',
            text = "Training complete."
        )

    else:

        # Time session with ColossalAI
        timer = MultiTimer()

        # trainer
        trainer = Trainer(
            engine = engine,
            timer =  timer,
            logger = logger
        )

        hook_list = [
            hooks.LossHook(),
            hooks.AccuracyHook(accuracy_func = Accuracy()),
            hooks.LogMetricByEpochHook(logger)
        ]

        trainer.fit(
            train_dataloader = train_dataloader,
            epochs = args.epochs,
            test_dataloader = test_dataloader,
            hooks = hook_list,
            display_progress = args.display_progress,
            test_interval = args.test_interval
        )


if __name__ == "__main__":

    cfg = CFG()

    LaMDA_Trainer(cfg)