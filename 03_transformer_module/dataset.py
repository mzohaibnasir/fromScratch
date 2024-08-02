import torch
import torch.nn as nn
from torch.utils.data import Dataset


class BilingualDataset(Dataset):
    def __init__(self, ds, tokenizer_src, tokenizer_tgt, src_lang, tgt_lang,
                 seq_len):
        super().__init__()

        self.ds = ds
        self.tokenizer_src = tokenizer_src
        self.tokenizer_tgt = tokenizer_src
        self.src_lang = src_lang
        self.tgt_lang = tgt_lang

        # special tokens initilization
        self.sos_token = torch.tensor([tokenizer_src.token_to_id(["[SOS]"])],
                                      dtype=torch.int64)
        self.eos_token = torch.tensor([tokenizer_src.token_to_id(["[EOS]"])],
                                      dtype=torch.int64)
        self.pad_token = torch.tensor([tokenizer_src.token_to_id(["[PAD]"])],
                                      dtype=torch.int64)

    def __len__(self):
        return len(self.ds)

    def __getitem__(self, index):
        src_tgt_pair = self.ds[index]

        src_text = src_tgt_pair["translation"][self.src_lang]
        tgt_text = src_tgt_pair["translation"][self.tgt_lang]

        # encoding
        enc_input_tokens = self.tokenizer_src.encode(
            src_text).ids  # (seq_len) -> (seq_len)
        dec_input_tokens = self.tokenizer_tgt.encode(tgt_text).ids

        # calculating # of padding tokens to be added
        enc_num_padding_tokens = self.seq_len - len(enc_input_tokens) - 2
        dec_num_padding_tokens = self.seq_len - len(dec_input_tokens) - 1

        # padding tokens should never be-ve
        if enc_num_padding_tokens < 0 or dec_num_padding_tokens < 0:
            raise ValueError("sentence is too long... exceeds seq_len limit")

        # lets build three tensors for encoder input and decoder input and also for lebel. So, one tensor would be send to encoder,
        # one to decoder input and one that we expect as decoder's output and that output will be called label/target
        encoder_input = torch.cat([
            self.sos_token,
            torch.tensor(enc_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * enc_num_padding_tokens,
                         dtype=torch.int64),
        ])

        decoder_input = torch.cat([
            self.sos_token,
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            torch.tensor([self.pad_token.item()] * dec_input_tokens,
                         dtype=torch.int64),
        ])

        label = torch.cat([
            torch.tensor(dec_input_tokens, dtype=torch.int64),
            self.eos_token,
            torch.tensor([self.pad_token.item()] * dec_input_tokens,
                         dtype=torch.int64),
        ])

        print(f"\t\t\t\tPadding added")
        print(f"\t\t\t\tsequenceTextLimit: {self.seq_len}")
        print(f"\t\t\t\tsrctextShape: {encoder_input.shape}")
        print(f"\t\t\t\ttgttextShape: {decoder_input.shape}")
        print(f"\t\t\t\tlabeltextShap: {label.shape}")

        return {
            "encoder_input":
            encoder_input,  # (seq_len)
            "decoder_input":
            decoder_input,  # (seq_len)
            "encoder_mask":
            (encoder_input != self.pad_token
             ).unsqueeze(0).unsqueeze(0).int(),  # (1, batch, seq,_len)
            "decoder_mask":
            (decoder_input != self.pad_token).unsqueeze(0).int()
            & causal_mask(
                decoder_input.size(0)),  # (1, seq_len) & (1, seq_len, seq_len)
            # causal_mask will build matrix of seq_len * seq_len     # (1,batch, seq_len) & (1,seq_len, seq_len)
            "label":
            label,
            "src_text":
            src_text,
            "tgt_text":
            tgt_text,
        }


def causal_mask(size):
    """
    causal_mask will build matrix of seq_len * seq_len     # (1,batch, seq_len) & (1,seq_len, seq_len)
    we want each word in decider to only watch non padding words that come before it.
    Below matrix represent K\*Q in softmax attention, we want to hide all the values above the diagonal.
    so we want all values above diagonal to be masked out."""

    mask = torch.triu(torch.ones(1, size, size), diagonal=1).type(torch.int)
    print(mask[:50, :50])
    return mask == 0


print(causal_mask(10))
