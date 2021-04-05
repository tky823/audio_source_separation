import torch.nn as nn

class MLP(nn.Module):
    def __init__(self, n_bins, hidden_channels=1024, num_layers=5):
        super().__init__()

        net = []
        for n in range(num_layers):
            if n == 0:
                net.append(nn.Linear(n_bins, hidden_channels))
            elif n == num_layers - 1:
                net.append(nn.Linear(hidden_channels, n_bins))
            else:
                net.append(nn.Linear(hidden_channels, hidden_channels))
            net.append(nn.ReLU())
        self.net = nn.Sequential(*net)


    def forward(self, input):
        """
        Args:
            input (batch_size, n_bins)
        Returns:
            output (batch_size, n_bins)
        """
        output = self.net(input)

        return output

def _test():
    torch.manual_seed(111)

    batch_size = 4
    n_bins = 2049

    input = torch.randn(batch_size, n_bins)
    model = MLP(n_bins, num_layers=5)
    output = model(input)

    print(input.size(), output.size())

if __name__ == '__main__':
    import torch

    _test()
