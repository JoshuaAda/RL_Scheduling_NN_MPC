
import torch
from torch.utils.data import Dataset


##### Dataset class for the MPC training
class CustomMPCDataset(Dataset):
    def __init__(self, states, inputs,bounds=None, transform=None):
        self.states= torch.from_numpy(states).type(torch.float)
        self.input=torch.from_numpy(inputs).type(torch.float)
        self.transform = transform
        if bounds is not None:
            self.input=(self.input+bounds[0])/(bounds[1]-bounds[0])
        if self.transform is not None:
            input=self.transform(self.input)
            self.input=input.float()



    def __len__(self):
        return len(self.input)

    def __getitem__(self, idx):
        states=self.states[idx]
        input = self.input[idx]

        return states, input

