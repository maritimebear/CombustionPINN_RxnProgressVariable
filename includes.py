import torch


class Dataset():
    def __init__(self, dataframe, input_label="x", output_label="reaction_progress"):
        self.inputs, self.outputs = [dataframe[key].to_numpy().reshape(-1, 1) for key in (input_label, output_label)]
        # Reshape to (batch_dim * input_dim)
        
        assert len(self.inputs) == len(self.outputs)

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return (self.inputs[idx], self.outputs[idx])


def train(dataloader, network, loss_fn, optimiser):
    network.train()
    loss_list = list()
    for batch in dataloader:
        x, y = batch
        y_hat = network(x)
        loss = loss_fn(y_hat, y)
        loss.backward()
        optimiser.step()
        optimiser.zero_grad()
        loss_list.append(loss.detach().item())

    return sum(loss_list) / len(loss_list)  # Mean loss over epoch

