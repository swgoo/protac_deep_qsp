from pathlib import Path
from typing import Callable
from attr import dataclass
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from torch import Tensor

from torch.optim.lr_scheduler import StepLR

from lightning.fabric import Fabric, seed_everything
import yaml

@dataclass
class Config:
    feature: list[str]
    target: dict[str,tuple[float, float]]
    argument: dict[str,tuple[float, float]]
    train_dataset_path: str
    test_dataset_path: str
    test_prediction_path: str
    train_prediction_path: str
    batch_size: int = 600
    num_workers: int = 1
    num_epochs: int = 100
    lr: float = 0.001
    dry_run: bool = False
    log_interval: int = 10
    model_save_path: str = 'data/model.pth'
    seed: int = 42
    feature_normalization : bool = True


    
def dmax(alpha : Tensor, e0: Tensor, kcat: Tensor, kdegp: Tensor, kde: Tensor, kdp: Tensor, *args, **kwargs) -> Tensor:
    return alpha * e0 * kcat * kdegp / (
    alpha * e0 * kcat * kdegp + 
    alpha * e0 + 
    kde + kdp + 
    2 * torch.sqrt(kde * kdp))

def dc50(alpha : Tensor, e0: Tensor, kcat: Tensor, kdegp: Tensor, kde: Tensor, kdp: Tensor, *args, **kwargs) -> Tensor:
    return 1/2 * (
        alpha * e0 * kcat * kdegp + 
        alpha * e0 + 
        kde + kdp + 
        4 * torch.sqrt(kde * kdp) - 
        torch.sqrt(
        (alpha * e0 * kcat * kdegp + 
            alpha * e0 + 
            kde + kdp + 
            4 * torch.sqrt(kde * kdp))**2 - 
        4 * kde * kdp
        )
    )

function_dict : dict[str, Callable] = {
    'dmax': dmax,
    'dc50': dc50
}

def clamp_with_sigmoid(x: Tensor, min: float, max: float) -> Tensor:
    return (max - min) * x.sigmoid() + min

class Model(nn.Module):
    def __init__(
            self, 
            num_features: int, 
            arguments_coefficient_boundary: dict[str, tuple[float, float],],
            funuctions: list[Callable] = [dmax, dc50]
            ):
        super(Model, self).__init__()
        self.arguments_coefficient_boundary = arguments_coefficient_boundary
        self.dnn = nn.Sequential(
            nn.Linear(num_features, 2*num_features),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2*num_features, 2*num_features),
            nn.Dropout(0.1),
            nn.ReLU(),
            nn.Linear(2*num_features, len(arguments_coefficient_boundary)),
        )
        self.funuctions = funuctions
    
    def forward(self, batch):
        arguments_coefficient_logit = self.dnn(batch['feature'])
        batch.pop('feature')
        arguments_original = batch
        arguments = {}
        for i, (name, (min, max)) in enumerate(self.arguments_coefficient_boundary.items()):
            arguments[name] = clamp_with_sigmoid(arguments_coefficient_logit[:,i], min, max) * arguments_original[name]
        targets = torch.stack([f(**arguments) for f in self.funuctions], dim=1)
        return targets


class Dataset(torch.utils.data.Dataset):
    def __init__(self, df: pd.DataFrame, config: Config):
        self.arguments_cols = list(config.argument.keys())
        self.argument = torch.tensor(df[self.arguments_cols].values, dtype=torch.float32)
        self.feature = torch.tensor(df[config.feature].values, dtype=torch.float32)
        self.target_cols = list(config.target.keys())
        self.targets = torch.tensor(df[self.target_cols].values, dtype=torch.float32)
        if config.feature_normalization:
            self.feature = (self.feature - self.feature.mean(dim=0)) / self.feature.std(dim=0)
    
    def __len__(self):
        return len(self.feature)
    
    def __getitem__(self, idx):
        input = {name: value for name, value in zip(self.arguments_cols, self.argument[idx]) }
        input['feature'] = self.feature[idx]
        return idx, (input, self.targets[idx])
    
def logit(x: Tensor, min: float, max: float) -> Tensor:
    x = torch.clamp(x, min+1e-5, max-1e-5)
    return torch.log((x - min) / (max - x))


    
def run(config : Config | Path):
    fabric = Fabric()
    if isinstance(config, Path):
        config = Config(**yaml.safe_load(open(config, 'r')))
    seed_everything(config.seed)

    with fabric.rank_zero_first(local=False):
        train_dataset = Dataset(pd.read_csv(config.train_dataset_path), config)
        test_dataset = Dataset(pd.read_csv(config.test_dataset_path), config)

    
    train_loader = DataLoader(train_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config.batch_size, num_workers=config.num_workers, shuffle=False)

    train_loader, test_loader = fabric.setup_dataloaders(train_loader, test_loader)

    target_boundary : dict[str, tuple[float, float]]= config.target


    function_list = [function_dict[name] for name in target_boundary.keys()]
    
    model = Model(num_features=len(config.feature), arguments_coefficient_boundary=config.argument, funuctions=function_list)
    optimizer = optim.Adam(model.parameters(), lr=config.lr)
    scheduler = StepLR(optimizer, step_size=1, gamma=0.995)


    model, optimizer = fabric.setup(model, optimizer)

    target_col_names = list(target_boundary.keys())
    predict_col_names = [f'{name}_predict' for name in target_col_names]

    
    for epoch in range(config.num_epochs):
        model.train()
        output_total = []
        target_total = []
        for batch_idx, batch in train_loader:
            optimizer.zero_grad()
            input, target = batch
            output = model(input)
            output_logit = torch.stack([logit(x, min, max) for x, (min, max) in zip(output.t(), target_boundary.values())], dim=1)
            target_logit = torch.stack([logit(x, min, max) for x, (min, max) in zip(target.t(), target_boundary.values())], dim=1)
            with torch.no_grad():
                output_total.append(torch.stack([clamp_with_sigmoid(x, min, max) for x, (min, max) in zip(output_logit.t(), target_boundary.values())], dim=1))
                target_total.append(torch.stack([clamp_with_sigmoid(x, min, max) for x, (min, max) in zip(target_logit.t(), target_boundary.values())], dim=1))

            loss = F.mse_loss(output_logit, target_logit)

            loss.backward()
            optimizer.step()

            if (epoch == 0) or ((epoch+1)% config.log_interval == 0):
                print(f"Train Epoch: {epoch} Loss: {loss.item():.6f}")
            
            if config.dry_run:
                break
        
        output_total = torch.cat(output_total)
        target_total = torch.cat(target_total)
        prediction_and_target = torch.cat([output_total, target_total], dim=1)
        # save the prediction and target
        if config.train_prediction_path is not None:
            pd.DataFrame(prediction_and_target.detach().cpu().numpy(), columns=predict_col_names+target_col_names).to_csv(config.train_prediction_path, index=False)
        
        scheduler.step()
        
        model.eval()
        test_loss = 0
        with torch.no_grad():
            output_total = []
            target_total = []
            for batch_idx, batch in test_loader:
                input, target = batch
                output = model(input)
                output_total.append(output)
                target_total.append(target)
                output_logit = torch.stack([logit(x, min, max) for x, (min, max) in zip(output, target_boundary.values())], dim=1)
                target_logit = torch.stack([logit(x, min, max) for x, (min, max) in zip(target, target_boundary.values())], dim=1)
                test_loss += F.mse_loss(output_logit, target_logit)
                if config.dry_run:
                    break
        output_total = torch.cat(output_total)
        target_total = torch.cat(target_total)
        prediction_and_target = torch.cat([output_total, target_total], dim=1)
        # save the prediction and target
        if config.test_prediction_path is not None:
            pd.DataFrame(prediction_and_target.cpu().numpy(), columns=predict_col_names+target_col_names).to_csv(config.test_prediction_path, index=False)

        test_loss = fabric.all_gather(test_loss).sum() / len(test_loader.dataset)

        print(f"\nTest set: Average loss: {test_loss:.4f}\n")

        if config.dry_run:
            break
    
        if config.model_save_path is not None:
            fabric.save(config.model_save_path, model.state_dict())
    
if __name__ == "__main__":
    run(Path('data/raw/config.yaml'))
