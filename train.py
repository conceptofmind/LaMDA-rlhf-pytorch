import torch
from torch import nn

import colossalai

from config.config import CFG

from dataloader.stream_dataloader import stream_dataloaders

from lamda_pytorch.lamda_pytorch import lamda_model

def LaMDA_Trainer(cfg: CFG):
    parser = colossalai.get_default_parser()

    parser.add_argument(
        '--use_trainer',
        action='store_true',
        help='whether to use trainer'
    )

    args = parser.parse_args()

    colossalai.launch_from_torch(config='./config/colossal_config.py')

    # LaMDA model
    model = lamda_model()

    # setup dataloaders
    if cfg.use_huggingface:
        train_dataloader, test_dataloader = stream_dataloaders(cfg)

    # loss function
    loss_fn = nn.CrossEntropyLoss()

    # optimizer function

    optimizer = torch.optim.Adam(model.parameters(), lr = cfg.lr)

    # initialze model, optimizer, criterion, and data loaders

    engine, train_dataloader, test_dataloader, _ = colossalai.initialize(
        model,
        optimizer,
        loss_fn,
        train_dataloader,
        test_dataloader
    )

    engine.train()
    for step, batch in enumerate(train_dataloader):
        inputs, labels = batch['inputs'].cuda(), batch['labels'].cuda()
        
        engine.zero_grad()
        outputs = engine(inputs)

        train_loss = engine.loss_fn(outputs, labels)

        engine.backward(train_loss)
        engine.step()

        engine.eval()
        for step, batch in enumerate(test_dataloader):
            inputs, labels = batch['inputs'].cuda(), batch['labels'].cuda()

            with torch.no_grad():
                outputs = engine(inputs)
                test_loss = engine.loss_fn(outputs, labels)
            
            engine.backward(test_loss)
            engine.step()

if __name__ == "__main__":

    cfg = CFG()

    LaMDA_Trainer(cfg)