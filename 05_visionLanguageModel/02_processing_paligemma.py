from typing import Dict, List, Tuple, Optional, Union, Iterable
import numpy as np
from PIL import Image
import torch

# https://github.com/google-research/big_vision/blob/main/big_vision/configs/proj/paligemma/README.md

IMAGENET_STANDARD_MEAN = [0.5, 0.5, 0.5]
IMAGENET_STANDARD_STD = [0.5, 0.5, 0.5] # # we gererallyuse mean and std of imagenet dataset, one for each channel




def add_image_token_to_prompt(
          prefix_prompt, # user prompt
          bos_token,
          image_seq_len,
          image_token,
          ):
    """
    # Quoting from the blog (https://huggingface.co/blog/paligemma#detailed-inference-process):
    #   The input text is tokenized normally. 
    #   A <bos> token is added at the beginning, and an additional newline token (\n) is appended. newline token is added to the end of the prompt telling the model that the prompt has ended.
    #   This newline token is an essential part of the input prompt the model was trained with, so adding it explicitly ensures it's always there.
    #   The tokenized text is also prefixed with a fixed number of <image> tokens.
    # NOTE: from the paper it looks like the `\n` should be tokenized separately, but in the HF implementation this is not done.
    #       ref to HF implementation: https://github.com/huggingface/transformers/blob/7f79a97399bb52aad8460e1da2f36577d5dccfed/src/transformers/models/paligemma/processing_paligemma.py#L55-L73
    """



    # we will add image tokens based on how many image tokens this model needs. incase of paiigemma-224, it is 256 image tokens
    return f"{image_token*image_seq_len}{bos_token}{prefix_prompt}\n"

    # "tell me where is photographer\n"  =>photographer\n might becaome a single token so we dont wwant \n to be merged theredire add \n seraprartely/manually


def resize(image: Image,
           size: Tuple[int, int], 
           resample: Image.Resampling = None,
           reducing_gap: Optional[int] = None,
           ) -> np.ndarray:
       height, width = size
       resized_image = image.resize(
            (width, height), 
            resample=resample,
            reducing_gap=reducing_gap
            )
       return resized_image


def rescale(image: np.ndarray, 
            scale: float,
            dtype : np.dtype = np.float32
            ) -> np.ndarray:
      rescaled_image = image.astype(dtype) * scale
      rescaled_image = rescaled_image.astype(dtype)
      return rescaled_image


def normalize(image: np.ndarray,
                mean: Union[float, Iterable[float]],
                std: Union[float, Iterable[float]],
                ) -> np.ndarray:
        mean = np.array(mean)
        std = np.array(std)
        image = (image - mean) / std
        return image

def process_images(
        images: List[Image.Image],
        size: Dict[str, int]= None,
        resample: Image.Resampling = None,
        rescale_factor: float = None,
        image_mean: Optional[Union[float, List[float]]] = None, # we gererally use mean and std of imagenet dataset
        image_std: Optional[Union[float, List[float]]] = None,
)-> List[np.ndarray]:
    height, width = size[0], size[1]
    images = [
        resize(image=image, size=(height, width), resample=resample) for image in images
    ]

    # convert images to numpy arrays
    images = [np.array(image) for image in images]

    # rescale the pixel values to be in the range [0, 1]
    images = [rescale(image, scale=rescale_factor) for image in images]

    # normalize the images to have mean 0 and std 1
    images = [normalize(image, mean=image_mean, std=image_std) for image in images]

    # move the channel dimension to the front
    # the model expects the images to be in the format [channel, height, width]
    images = [image.transpose(2, 0, 1) for image in images]
    return images

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
        ) # it woill load rescale normalize the image and convert it to tensor that can be processed by vision model. It returns list of tensors   


        # convert the returned list of numpy arrays to a single numpy array with shape [batch_size, channel, height, width[]
        # here we are getting stack of images
        pixel_values = np.stack(pixel_values, axis=0) # adding batch size

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
            
        ]#  tokeniing is not happeininghere.. and image placeholders are being added here # returns strings includin image polaceholder tokens


        # we tokenize it NOW.
        # returns input ids(not embeddings) and attention mask
        # as we are not using padding, so attention_mask will be just 1
        inputs = self.tokenizer(
            input_strings,
            padding=padding,
            truncation=truncation,
            return_tensors="pt",
        )

        return_data = {"pixel_values": pixel_values,**inputs]} # returning pixel values and tokenized input ids

        return return_data
    
