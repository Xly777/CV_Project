import gradio as gr
from PIL import Image
from model import segment_image,generate_image

# 存储点击位置的列表
click_coordinates = []

def get_coordinates(img, evt: gr.SelectData):
    global click_coordinates
    click_coordinates.append((evt.index[0], evt.index[1]))
    segmented_image, best_mask=segment_image(img, click_coordinates)
    # 将mask图像的透明度设置为50%
    mask_image = best_mask.convert("RGBA")
    alpha = mask_image.split()[3]
    alpha = alpha.point(lambda p: p * 0.5)
    mask_image.putalpha(alpha)
    # 将mask图像覆盖到原图上
    combined_image = Image.alpha_composite(img.convert("RGBA"), mask_image)
    return img, click_coordinates, combined_image,best_mask

def clear_coordinates():
    global click_coordinates
    click_coordinates = []
    return ""
best_mask=None
with gr.Blocks() as demo:
    with gr.Row():
        img_input = gr.Image(type="pil", label="上传图片")
        img_mask = gr.Image(type="pil", label="处理后图片")
        img_result=gr.Image(type="pil", label="结果图片")
    coordinates_display = gr.Textbox(label="点击位置坐标", interactive=False)
    clear_button = gr.Button("清除点击")
    result_button=gr.Button("生成图片")
    img_input.select(get_coordinates, [img_input], [img_input, coordinates_display, img_mask,best_mask])
    clear_button.click(clear_coordinates, [], coordinates_display)
    result_button.click(generate_image, [img_input,best_mask], img_result)
demo.launch()
