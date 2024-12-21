import torch
from torch.utils.data import Dataset

class MIDITokensDataset(Dataset):
    def __init__(self, tokens_list, max_length, pad_token, EOS_token):
        # Convert all tokens to tensors during initialization and pad the list
        self.tokens_list = (
            (
                (
                    torch.tensor(
                        tokens[:-1] + [pad_token] * (self.max_length - len(tokens))
                    ),
                    torch.tensor(
                        tokens[1:]
                        + self.EOS_token
                        + [pad_token] * (self.max_length - len(tokens))
                    ),
                )
                if len(tokens) < max_length
                else torch.tensor((tokens[:-1], tokens[1:] + self.EOS_token))
            )
            for tokens in tokens_list
        )

        self.EOS_token = EOS_token

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        # The labels should be the input shifted right + the EOS token
        input, labels = self.tokens_list[idx]
        return input, labels
