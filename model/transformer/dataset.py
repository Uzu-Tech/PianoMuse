from torch.utils.data import Dataset


class MIDITokensDataset(Dataset):
    def __init__(self, tokens_list, max_length):
        # The pad token should be the next token after the EOS token
        pad_token = tokens_list[0][-1] + 1
        # Pad the list
        self.tokens_list = [
            (
                # Remove the EOS token in inputs and add to labels
                tokens[:-1] + [pad_token] * (max_length - len(tokens) - 1),
                # labels
                tokens[1:] + [pad_token] * (max_length - len(tokens) - 1),
            )
            for tokens in tokens_list
        ]

    def __len__(self):
        return len(self.tokens_list)

    def __getitem__(self, idx):
        input, labels = self.tokens_list[idx]
        return input, labels
