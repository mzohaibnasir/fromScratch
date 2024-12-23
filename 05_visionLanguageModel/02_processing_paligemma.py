from typing import Dict, List, Tuple, Optional, Union, Iterable
import numpy as np
from PIL import Image
import torch

# https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5]


class PaliGemmaProcessor:
    """
    given the input text(user prompt) and input image,
    preprocess it(resize,rescale)
    and will create this text tokens with placeholders for image tokens
    """

    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer, num_image_tokens: int, image_size: int):
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        # TOKNIZER: https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md
        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}
        tokenizer.add_special_tokens(tokens_to_add)

        EXTRA_TOKENS = [
            f"<loc{i:04d}>" for i in range(1024)
        ]  # these tokens are used for object detection (bounding box) tasks

        EXTRA_TOKENS += [
            f"<seg{i:03}>" for i in range(128)
        ]  # these tokens are used for object segmentation tasks

        tokenizer.add_tokens(EXTRA_TOKENS)
        self.image_token_id = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        # we will add the BOS and EOS tokens ourselves

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image], # in input we have image or list of images
        padding: str = "longest",
        truncation: bool = True,
    ) -> dict:
        """
        
        it allows instance of class to be called as a function
        i.e. paligemmaprocessor(...)
        

        # only accepts one image and one text because will complicate the code
        """
        assert len(images) ==1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts"

        # process  images
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1.0/255.0,
            image_mean = IMAGENET_STANDARD_MEAN,
            image_std = IMAGENET_STANDARD_STD,
        ) # it woill load rescale normalize the image and convert it to tensor that can be processed by vision model    


        # convert the returned list of numpy arrays to a single numpy array with shape [batch_size, channel, height, width[]
        pixel_values = np.stack(pixel_values, axis=0)

        # convert the numpy array to a torch tensor
        pixel_values = torch.tensor(pixel_values)


        # input to the model: this method will createa tokenes of text and place holders for image tokens

        # prepend a 'self.image_seq_len' number of image tokens to the prompt
        input_strings = [
            add_image_token_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
            
        ]


        # we tokenize it using placehodler tokens fo rimage
        # returns input ids and attention mask
        inputs = self.tokenizer(
            input_strings,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )

        return_data = {"pixel_values": pixel_values,**inputs]}

        return return_data