import torch
import torch.nn as nn
import warnings
from datasets import load_dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.trainers import WordLevelTrainer
from tokenizers.pre_tokenizers import Whitespace
from pathlib import Path
from dataset import BilingualDataset, causal_mask
from torch.utils.data import Dataset, DataLoader, random_split
from model import build_transformer
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from config import get_weights_file_path, get_config
from tqdm import tqdm

# validation


def greedy_decode(model, source, source_mask, tokenizer_src, tokenizer_tgt,
                  max_len, device):
    sos_idx = tokenizer_tgt.token_to_id("[SOS]")
    eos_idx = tokenizer_tgt.token_to_id("[EOS]")

    # precompute the encoder output
    encoder_output = model.encode(source, source_mask)

    # initialize decoder_input with sos token
    decoder_input = (torch.empty(1,
                                 1).fill_(sos_idx).type_as(source).to(device)
                     )  # (batch, tokens for decoder input)
    # print(f"Greedy: decoder_input: {decoder_input.shape}")

    # keep asking the decoder to output next token untill we get <eos> or max_len
    while True:
        if decoder_input.size(1) == max_len:  # exceeds max_len
            break

        # build the mask for target(decoder_input)
        # causal_because we dont want our decoder input  to watch future words
        decoder_mask = (causal_mask(
            decoder_input.size(1)).type_as(source_mask).to(device))

        # any other mask is not needed beacuse we dont have padding tokens
        # calculate output of decoder
        out = model.decode(encoder_output, source_mask, decoder_input,
                           decoder_mask)

        # get the next token probabilities using projection_layer, but we only want projection of last token
        prob = model.project(out[:, -1])

        # select the token with the max probability( this is greeady search)
        _, next_word = torch.max(prob, dim=1)

        # now append it back to decoder_input, so that it becomes input of next iterations
        decoder_input = torch.cat(
            [
                decoder_input,
                torch.empty(1, 1).type_as(source).fill_(
                    next_word.item()).to(device),
            ],
            dim=1,
        )

        if next_word == eos_idx:
            break
        # squeezing batch_dim
    return decoder_input.squeeze(0)


def run_validation(
    model,
    validation_ds,
    tokenizer_src,
    tokenizer_tgt,
    max_len,
    device,
    print_msg,
    global_step,
    writer,
    num_examples=2,
):
    model.eval()  # tells python that we'll be evaluating model
    count = 0
    # we'll be inferencing `num_examples` sentences  and analyze ouput of model
    source_texts = []
    expected = []
    predicted = []

    # size of control window (just use a default value)
    console_width = 80
    with torch.no_grad():  # disabling gradient calcuation
        for batch in validation_ds:
            # validation_ds has batch_size=1
            count += 1  # we only want to inference two sentences, so we'll keep a counT OF how many we have

            encoder_input = batch["encoder_input"].to(device)  # (b, seq_len)
            encoder_mask = batch["encoder_mask"].to(
                device)  # (b, 1, 1, seq_len)
            # verify size of batch is 1
            assert encoder_input.size(
                0) == 1, "batch_size must be 1 for validation"
            # for inference, we are supposed to calculate encoder output just once, and reuse it for every token that model will output from decoder.
            # because encoder output is calculated first,  parallelly, beacuse it represents the fixed input sequence which remains constant while the decoder generates the
            # output sequence token by token ie. not parallel. encoder is supposed to create a rich representation of input sequence so it can be fed at one. but decoder generates one token at a time
            # so we'll be doing greedy decoding
            model_out = greedy_decode(
                model,
                encoder_input,
                encoder_mask,
                tokenizer_src,
                tokenizer_tgt,
                max_len,
                device,
            )

            source_text = batch["src_text"][0]
            target_text = batch["tgt_text"][0]

            # convert back to word
            model_out_text = tokenizer_tgt.decode(
                model_out.detach().cpu().numpy())

            # append
            source_texts.append(source_text)
            expected.append(target_text)
            predicted.append(model_out_text)

            # print on console
            # we are using print_msg because it is not suggested to print directly on console when tqdm progress bar is runnin.. so we'll be using tqdm's print function
            # so that output doesn't interfere with progress_bar printing
            print_msg("-" * console_width)  # print some bars
            print_msg(f"SOURCE:{source_text}")

            print_msg(f"TARGET:{target_text}")
            print_msg(f"PREDICTED:{model_out_text}")

            if count == num_examples:
                break

        # send it to tensorboard
        # if writer:
        # using TorchMetrics to calculate CharErrorRate, BLEU, WordErrorRate
        #    break

    # pass


def get_all_sentences(ds, lang):
    # pasrsing each item which is a pair in dataset # (english, italian)
    # print(ds)
    for item in ds:
        sentence = item["translation"][lang]
        # print(f"Sentence ({lang}): {sentence}")
        yield item["translation"][lang]


def get_or_build_tokenizer(config, ds, lang):
    """lang: language to build tokenizer for"""
    # config['tokenizer_file'] = '../tokenizers/tokenizer_{0}'
    tokenizer_path = Path(
        config["tokenizer_file"].format(lang))  # mean we can change
    if not Path.exists(tokenizer_path):
        tokenizer = Tokenizer(WordLevel(unk_token="[UNK]"))
        # split by wordspace
        tokenizer.pre_tokenizer = Whitespace()
        trainer = WordLevelTrainer(
            special_tokens=["[UNK]", "[PAD]", "[SOS]", "[EOS]"],
            min_frequency=2)

        print("tokenizer training started...")
        tokenizer.train_from_iterator(get_all_sentences(ds, lang),
                                      trainer=trainer)
        tokenizer.save(str(tokenizer_path))
    else:
        tokenizer = Tokenizer.from_file(str(tokenizer_path))

    print(f"{lang}_tokenizer initiated!")
    return tokenizer


def get_ds(config):
    ds_raw = load_dataset("opus_books",
                          f"{config['lang_src']}-{config['lang_tgt']}",
                          split="train")
    # print(f"\n\n\n\n\n\n\n\nds_raw: {ds_raw}")

    # build tokenizer
    print("Building TOkenizer...")
    tokenizer_src = get_or_build_tokenizer(config, ds_raw, config["lang_src"])
    tokenizer_tgt = get_or_build_tokenizer(config, ds_raw, config["lang_tgt"])

    # print(f"\n\n\ntokenizer_src:{tokenizer_src}")
    # print(f"\n\n\ntokenizer_tgt:{tokenizer_tgt}")

    # 90% for training - 10% for testing
    train_ds_size = int(0.9 * len(ds_raw))
    val_ds_size = len(ds_raw) - train_ds_size
    train_ds_raw, val_ds_raw = random_split(ds_raw,
                                            [train_ds_size, val_ds_size])

    train_ds = BilingualDataset(
        train_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )
    val_ds = BilingualDataset(
        val_ds_raw,
        tokenizer_src,
        tokenizer_tgt,
        config["lang_src"],
        config["lang_tgt"],
        config["seq_len"],
    )

    # to choose max-len, we'll make sure its greater than all other sentences' length
    max_len_src, max_len_tgt = 0, 0
    for item in ds_raw:
        # print("\n\n item:", item)

        # load each sentence , convert it to ids using tokenizer and i check length.
        src_sentence = item["translation"][config["lang_src"]]
        tgt_sentence = item["translation"][config["lang_tgt"]]
        # print(f"Source Sentence : {src_sentence}")
        # print(f"Target Sentence : {tgt_sentence}")

        src_ids = tokenizer_src.encode(src_sentence).ids
        tgt_ids = tokenizer_tgt.encode(tgt_sentence).ids

        # print(f"Source Sentence : {src_ids}")
        # print(f"Target Sentence : {tgt_ids}")

        # print(f"Source Sentence Length: {len(src_ids)}")
        # print(f"Target Sentence Length: {len(tgt_ids)}")
        # print("\n\nsrc_ids:", src_ids )
        # print("\n\tgt_ids:", tgt_ids )
        max_len_src = max(max_len_src, len(src_ids))
        max_len_tgt = max(max_len_tgt, len(tgt_ids))

        #############################3

    print(f"Max len src: {max_len_src}")
    print(f"Max len tgt: {max_len_tgt}")

    # data loader
    train_dataloader = DataLoader(train_ds,
                                  batch_size=config["batch_size"],
                                  shuffle=True)
    val_dataloader = DataLoader(
        val_ds, batch_size=1, shuffle=True
    )  # batch_size=1 because we want to process each sentence one by one
    return train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt


def get_model(config, vocab_src_len, vocab_tgt_len):
    model = build_transformer(
        vocab_src_len,
        vocab_tgt_len,
        config["seq_len"],
        config["seq_len"],
        config["d_model"],
    )
    return model


def train_model(config):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # device = torch.device("cpu")
    print(f"Using device: {device}")

    Path(config["model_folder"]).mkdir(parents=True, exist_ok=True)
    # print(f"config: {config}")

    train_dataloader, val_dataloader, tokenizer_src, tokenizer_tgt = get_ds(
        config)

    model = get_model(config, tokenizer_src.get_vocab_size(),
                      tokenizer_tgt.get_vocab_size()).to(device)

    # enable tensorboard
    writer = SummaryWriter(config["experiment_name"])

    # optomizer
    optimizer = torch.optim.Adam(model.parameters(), lr=config["lr"], eps=1e-9)

    # Resume training incase model trainig crashes
    initial_epoch = 0
    global_step = 0
    if config["preload"]:
        model_filename = get_weights_file_path(config, config["preload"])
        print(f"Preloading  model: {model_filename}")
        # loading file
        state = torch.load(model_filename)
        initial_epoch = state["epoch"] + 1
        optimizer.load_state_dict(state["optimizer_state_dict"])
        global_step = state["global_step"]

    # loss fn
    loss_fn = nn.CrossEntropyLoss(
        ignore_index=tokenizer_src.token_to_id("[PAD]"), label_smoothing=0.1)
    # ignore_index: To ignore padding tokens so that they don't have any impact on calculating loss
    # Label smoothing is a technique used to smooth the target labels by assigning a small probability to the incorrect classes and reducing the confidence on the correct class.
    # This helps prevent the model from becoming too confident and overfitting to the training data.
    # label_smoothing=0.1 means that for each true label, 10% of the probability mass is redistributed to all other classes.

    for epoch in range(initial_epoch, config["num_epochs"]):
        batch_iterator = tqdm(train_dataloader,
                              desc=f"Processing epoch: {epoch:02d}")
        for batch in batch_iterator:
            # print(f"batch: {batch.keys()}\n\n\n")

            model.train()
            # model.train() tells your model that you are training the model. This helps inform layers such as Dropout and BatchNorm, which are designed to behave differently during
            # training and evaluation. For instance, in training mode, BatchNorm updates a moving average on each new batch; whereas, for evaluation mode, these updates are frozen.
            encoder_input = batch["encoder_input"].to(
                device)  # (batch, seq_len)
            decoder_input = batch["decoder_input"].to(
                device)  # (batch, seq_len)
            encoder_mask = batch["encoder_mask"].to(
                device)  # (batch, 1, 1, seq_len)
            decoder_mask = batch["decoder_mask"].to(device)

            print(
                f"\t\t\t\t\tencoder_input (batch, seq_len): {encoder_input.shape}"
            )
            print(
                f"\t\t\t\t\tdecoder_input (batch, seq_len): {decoder_input.shape}"
            )
            # print(
            #     f"\t\t\t\t\tencoder_mask (1, batch, seq,_len): {len(encoder_mask)}"
            # )
            print(
                f"\t\t\t\t\tencoder_mask (batch, 1, 1, seq,_len): {encoder_mask.shape}"
            )
            # print(f"\t\t\t\t\tencoder_mask[1]: {encoder_mask[1].shape}")

            print(
                f"\t\t\t\t\tdecoder_mask (batch, 1, seq_len, seq_len): {decoder_mask.shape}"
            )

            # run through transformer modules
            encoder_output = model.encode(encoder_input, encoder_mask)
            decoder_output = model.decode(encoder_output, encoder_mask,
                                          decoder_input, decoder_mask)
            proj_output = model.project(decoder_output)
            print(
                f"\t\t\t\t\tprojected layer (batch, seq_len, vocab_size): {proj_output.shape}"
            )
            # (batch, seq_len, vocab_size) --> (batch*seq_len, vocab_size)
            proj_output = proj_output.view(-1, tokenizer_tgt.get_vocab_size())
            print(
                f"\t\t\t\t\tprojected layer reshaped (batch*seq_len, vocab_size): {proj_output.shape}"
            )

            label = batch["label"].to(device)  # (batch, seq)
            print(f"\t\t\t\t\tlabel (batch, seq,_len): {label.shape}")
            label = label.view(-1)
            print(f"\t\t\t\t\tlabel (batch*seq_len): {label.shape}")

            loss = loss_fn(proj_output, label)
            batch_iterator.set_postfix({"loss": f"{loss.item():6.3f}"})
            """
            The intuition behind using the reshaped projected output tensor `(batch*seq_len, vocab_size)` and the reshaped target sequence tensor `(batch*seq_len)` as inputs
            to the cross-entropy loss function is to compare the predicted probability distribution over the vocabulary for each position in the sequence with the true label 
            for that position. 
            The projected output tensor `(batch*seq_len, vocab_size)` is a tensor that contains the predicted probability distribution over the vocabulary for each position 
            in the sequence. Each row of the tensor corresponds to a position in the sequence, and each column corresponds to a word in the vocabulary. The value in each 
            cell is the predicted probability of that word appearing at that position in the sequence.
            The target sequence tensor `(batch*seq_len)` is a tensor that contains the true label for each position in the sequence. Each element of the tensor is an integer
            that corresponds to the index of the true word in the vocabulary.The cross-entropy loss function calculates the loss between the predicted probability distribution 
            and the true label for each position in the sequence. It does this by taking the negative log-likelihood of the true label given the predicted probability distribution.
            The loss is then averaged over all positions in the sequence to get the final loss value.
            By reshaping the projected output tensor and the target sequence tensor to have the same shape `(batch*seq_len)`, we can compare the predicted probability distribution 
            and the true label for each position in the sequence. This allows us to update the model parameters in a way that minimizes the loss and improves the accuracy of the model.
            """

            # loss loss
            writer.add_scalar("train_loss", loss.item(), global_step)
            writer.flush()

            # backpropagate
            loss.backward()

            # update weights
            optimizer.step()
            optimizer.zero_grad()
            global_step += 1

        # its better to run validation on every few steps.. but here i'll be running on iteration
        run_validation(
            model,
            val_dataloader,
            tokenizer_src,
            tokenizer_tgt,
            config["seq_len"],
            device,
            lambda msg: batch_iterator.write(msg),
            global_step,
            writer,
        )

        # save model
        model_filename = get_weights_file_path(config, f"{epoch:02d}")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "global_step": global_step,
            },
            model_filename,
        )


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    config = get_config()
    train_model(config)
