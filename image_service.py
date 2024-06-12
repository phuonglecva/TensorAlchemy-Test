import asyncio
import copy
import time
from typing import Optional
import typing
import pydantic
import bittensor as bt

from neurons.miners.StableMiner.utils import clean_nsfw_from_prompt, do_logs, nsfw_image_filter

class ImageGeneration(bt.Synapse):
    """
    A simple dummy protocol representation which uses bt.Synapse as its base.
    This protocol helps in handling dummy request and response communication between
    the miner and the validator.

    Attributes:
    - dummy_input: An integer value representing the input request sent by the validator.
    - dummy_output: An optional integer value which, when filled, represents the response from the miner.
    """

    # Required request input, filled by sending dendrite caller.
    prompt: str = pydantic.Field("Bird in the sky", allow_mutation=False)
    negative_prompt: str = pydantic.Field(None, allow_mutation=False)
    prompt_image: Optional[bt.Tensor]
    images: typing.List[bt.Tensor] = []
    num_images_per_prompt: int = pydantic.Field(1, allow_mutation=False)
    height: int = pydantic.Field(1024, allow_mutation=False)
    width: int = pydantic.Field(1024, allow_mutation=False)
    generation_type: str = pydantic.Field("text_to_image", allow_mutation=False)
    guidance_scale: float = pydantic.Field(7.5, allow_mutation=False)
    seed: int = pydantic.Field(1024, allow_mutation=False)
    steps: int = pydantic.Field(50, allow_mutation=False)


def convert_image_generation_to_dict(image_generation: ImageGeneration) -> dict:
    # Convert the ImageGeneration object to a dictionary can pass to payload API call
    return {
        "prompt": image_generation.prompt,
        "negative_prompt": image_generation.negative_prompt,
        "prompt_image": image_generation.prompt_image.tolist() if image_generation.prompt_image is not None else [],
        "images": [image.tolist() for image in image_generation.images] if image_generation.images is not None else [],
        "num_images_per_prompt": image_generation.num_images_per_prompt,
        "height": image_generation.height,
        "width": image_generation.width,
        "generation_type": image_generation.generation_type,
        "guidance_scale": image_generation.guidance_scale,
        "seed": image_generation.seed,
        "steps": image_generation.steps,
    }
# generate example image_generation object
# image_dict = {
#     "prompt": "Bird in the sky",
#     "negative_prompt": None,
#     "prompt_image": [],
#     "images": [],
#     "num_images_per_prompt": 1,
#     "height": 1024,
#     "width": 1024,
#     "generation_type": "text_to_image",
#     "guidance_scale": 7.5,
#     "seed": 1024,
#     "steps": 50,
# }
def convert_dict_to_image_generation(dict_obj: dict) -> ImageGeneration:
    # Convert the dictionary to an ImageGeneration object
    images = dict_obj["images"]
    images = [bt.Tensor.tensor(image) for image in images] if images != [] else []
    prompt_image = dict_obj["prompt_image"]
    prompt_image = bt.Tensor.tensor(prompt_image) if prompt_image is not None else None
    return ImageGeneration(
        prompt=dict_obj["prompt"],
        negative_prompt=dict_obj["negative_prompt"],
        prompt_image=prompt_image,
        images=images,
        num_images_per_prompt=dict_obj["num_images_per_prompt"],
        height=dict_obj["height"],
        width=dict_obj["width"],
        generation_type=dict_obj["generation_type"],
        guidance_scale=dict_obj["guidance_scale"],
        seed=dict_obj["seed"],
        steps=dict_obj["steps"],
    )
from neurons.utils import output_log, sh
import torch
from diffusers import (
    AutoPipelineForImage2Image,
    AutoPipelineForText2Image,
    DPMSolverMultistepScheduler,
)
from neurons.safety import StableDiffusionSafetyChecker
from transformers import CLIPImageProcessor

class ImageGeneration:
    def __init__(self):
        self.load_models()
        
    def load_models(self):
        ### Load the text-to-image model
        self.t2i_model = AutoPipelineForText2Image.from_pretrained(
            self.config.miner.model,
            torch_dtype=torch.float16,
            use_safetensors=True,
            variant="fp16",
        ).to(self.config.miner.device)
        self.t2i_model.set_progress_bar_config(disable=True)
        self.t2i_model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.t2i_model.scheduler.config
        )

        ### Load the image to image model using the same pipeline (efficient)
        self.i2i_model = AutoPipelineForImage2Image.from_pipe(self.t2i_model).to(
            self.config.miner.device,
        )
        self.i2i_model.set_progress_bar_config(disable=True)
        self.i2i_model.scheduler = DPMSolverMultistepScheduler.from_config(
            self.i2i_model.scheduler.config
        )

        self.safety_checker = StableDiffusionSafetyChecker.from_pretrained(
            "CompVis/stable-diffusion-safety-checker"
        ).to(self.config.miner.device)
        self.processor = CLIPImageProcessor()

        ### Set up mapping for the different synapse types
        self.mapping = {
            "text_to_image": {"args": self.t2i_args, "model": self.t2i_model},
            "image_to_image": {"args": self.i2i_args, "model": self.i2i_model},
        }
        
        
    async def generate(self, synapse: dict):
        synapse = convert_dict_to_image_generation(synapse)
        ### Misc
        timeout = synapse.timeout
        self.stats.total_requests += 1
        start_time = time.perf_counter()

        ### Set up args
        local_args = copy.deepcopy(self.mapping[synapse.generation_type]["args"])
        local_args["prompt"] = [clean_nsfw_from_prompt(synapse.prompt)]
        local_args["width"] = synapse.width
        local_args["height"] = synapse.height
        local_args["num_images_per_prompt"] = synapse.num_images_per_prompt
        try:
            local_args["guidance_scale"] = synapse.guidance_scale

            if synapse.negative_prompt:
                local_args["negative_prompt"] = [synapse.negative_prompt]
        except:
            bt.logging.info("Values for guidance_scale or negative_prompt were not provided.")

        try:
            local_args["num_inference_steps"] = synapse.steps
        except:
            bt.logging.info("Values for steps were not provided.")

        ### Get the model
        model = self.mapping[synapse.generation_type]["model"]

        if synapse.generation_type == "image_to_image":
            local_args["image"] = T.transforms.ToPILImage()(
                bt.Tensor.deserialize(synapse.prompt_image)
            )

        ### Output logs
        do_logs(self, synapse, local_args)

        ### Generate images & serialize
        for attempt in range(3):
            try:
                seed = synapse.seed if synapse.seed != -1 else self.config.miner.seed
                local_args["generator"] = [
                    torch.Generator(device=self.config.miner.device).manual_seed(seed)
                ]
                images = model(**local_args).images
                
                synapse.images = [
                    bt.Tensor.serialize(self.transform(image)) for image in images
                ]
                output_log(
                    f"{sh('Generating')} -> Succesful image generation after {attempt+1} attempt(s).",
                    color_key="c",
                )
                break
            except Exception as e:
                bt.logging.error(
                    f"Error in attempt number {attempt+1} to generate an image: {e}... sleeping for 5 seconds..."
                )
                await asyncio.sleep(5)
                if attempt == 2:
                    images = []
                    synapse.images = []
                    bt.logging.error(
                        f"Failed to generate any images after {attempt+1} attempts."
                    )

        ### Count timeouts
        if time.perf_counter() - start_time > timeout:
            self.stats.timeouts += 1

        ### Log NSFW images
        if any(nsfw_image_filter(self, images)):
            bt.logging.debug(f"An image was flagged as NSFW: discarding image.")
            self.stats.nsfw_count += 1
            synapse.images = []

        ### Log to wandb
        try:
            if self.wandb:
                ### Store the images and prompts for uploading to wandb
                self.wandb._add_images(synapse)

                #### Log to Wandb
                self.wandb._log()

        except Exception as e:
            bt.logging.error(f"Error trying to log events to wandb.")

        #### Log time to generate image
        # generation_time = time.perf_counter() - start_time
        # self.stats.generation_time += generation_time
        # output_log(
        #     f"{sh('Time')} -> {generation_time:.2f}s | Average: {self.stats.generation_time / self.stats.total_requests:.2f}s",
        #     color_key="y",
        # )
        return synapse
from fastapi  import FastAPI

app = FastAPI() 
image_generation = ImageGeneration
@app.post("/image_generation")
def image_generation(image_generation: dict):
    # Convert the request to ImageGeneration object
    image_generation_obj = convert_dict_to_image_generation(image_generation)
    image_generation_obj = image_generation.generate(image_generation_obj)
    image_dict = convert_image_generation_to_dict(image_generation_obj)
    return {
        "data": image_dict
    }

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)