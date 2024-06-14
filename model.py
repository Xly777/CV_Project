import torch
import numpy as np
from PIL import Image
import cv2
import matplotlib.pyplot as plt
from segment_anything import sam_model_registry, SamPredictor
from diffusers import StableDiffusionInpaintPipeline, DPMSolverMultistepScheduler
from loguru import logger
model_type="vit_h"
sam_checkpoint="./models/sam_vit_h_4b8939.pth"
sam = None
try:
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint).to("cuda:0")
    predictor = SamPredictor(sam)
    logger.info(f"loading sam sucessfully.....")

    pipe = StableDiffusionInpaintPipeline.from_pretrained(
        "models/stable-diffusion",
        torch_dtype=torch.float16,
    )
    pipe.scheduler = DPMSolverMultistepScheduler.from_config(pipe.scheduler.config)
    pipe.to("cuda:1")
    logger.info(f"loading stable diffusion sucessfully")

except Exception as e:
    print(f"Error loading model: {e}")


def segment_image(image, points):
    """
    使用 SAM 模型对图像进行分割。

    参数:
    image_path (str): 图像文件的路径。
    points (list of tuples): 用于分割的一系列点坐标 [(x1, y1), (x2, y2), ...]。
    model_type (str): 使用的模型类型（默认: "vit_h"）。
    sam_checkpoint (str): SAM 模型的检查点文件路径。

    返回:
    np.ndarray: 分割后的图像和掩码。
    """


    try:
        predictor.set_image(image)

        input_points = np.array(points)
        input_labels = np.ones(len(points))

        masks, scores, logits = predictor.predict(
            point_coords=input_points,
            point_labels=input_labels,
            multimask_output=True
        )

        best_mask = masks[np.argmax(scores)]

        segmented_image = image.copy()
        # 0为没有被sam出来的部分， 1为sam出来的
        segmented_image[best_mask == 0] = 0

        return segmented_image, best_mask
    except Exception as e:
        print(f"Error during prediction: {e}")
        return None, None

def generate_image(image, mask_image):
    prompt = "high resolution"
    image_pil = Image.fromarray(image)
    inverted_mask_image = (mask_image.astype(np.uint8))

    image = pipe(prompt="", image=image_pil, mask_image=inverted_mask_image).images[0]
    
    return image

if __name__ == "__main__":
    image_path = "./image.png"
    image = None
    try:
        image = cv2.imread(image_path)
        if image is None:
            raise FileNotFoundError(f"Image file {image_path} not found.")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    except Exception as e:
        print(f"Error loading image: {e}")

    points = [(320, 228)]

    segmented_image,best_mask = segment_image(image, points)

    plt.imshow(segmented_image)
    plt.title("Segmented Image")
    plt.savefig('1.png')

    plt.imshow(best_mask)
    plt.title("Segmented Mask Image")
    plt.savefig('1_mask.png')
    image1 = generate_image(image, best_mask)
    image1.save("./yellow_cat_on_park_bench1.png")
    image.save("./origin.png")
    # plt.show()
