import gradio as gr
from PIL import Image,ImageDraw
from model import segment_image,generate_image
import numpy as np
# 存储点击位置的列表
click_coordinates = []
def generate_gradio_image(image_input,best_mask):
    return generate_image(np.array(image_input),best_mask)
def plot_coordinates_on_image(img, coords, point_color=(0, 139, 139), point_radius=15):
    draw = ImageDraw.Draw(img)
    # 绘制坐标点
    for x, y in coords:
        draw.ellipse((x - point_radius, y - point_radius, x + point_radius, y + point_radius), fill=point_color)
    return img
def get_coordinates(img, evt: gr.SelectData):
    global click_coordinates
    click_coordinates.append((evt.index[0], evt.index[1]))
    segmented_image, best_mask=segment_image(np.array(img), click_coordinates)
    # 将mask图像的透明度设置为50%
    _best_mask=Image.fromarray(best_mask)
    mask_image = _best_mask.convert("RGBA")
    alpha = mask_image.split()[3]
    alpha = alpha.point(lambda p: p * 0.5)
    mask_image.putalpha(alpha)
    # 将mask图像覆盖到原图上
    combined_image = Image.alpha_composite(img.convert("RGBA"), mask_image)
    combined_image=plot_coordinates_on_image(combined_image, click_coordinates)
    return img, click_coordinates, combined_image,best_mask

def clear_coordinates():
    global click_coordinates
    click_coordinates = []
    return ""
def cancel():
    global click_coordinates
    click_coordinates = click_coordinates[:-1]
    return click_coordinates
with gr.Blocks() as demo:
    best_mask=gr.State(None)
    with gr.Row():
        img_input = gr.Image(type="pil", label="上传图片")
        img_mask = gr.Image(type="pil", label="处理后图片")
        img_result=gr.Image(type="pil", label="结果图片")
    coordinates_display = gr.Textbox(label="点击位置坐标", interactive=False)
    cancel_button=gr.Button("撤销上一次点击")
    clear_button = gr.Button("清除所有点击")
    result_button=gr.Button("生成图片")
    img_input.select(get_coordinates, [img_input], [img_input, coordinates_display, img_mask,best_mask])
    cancel_button.click(cancel, [], coordinates_display)
    clear_button.click(clear_coordinates, [], coordinates_display)
    result_button.click(generate_gradio_image, [img_input,best_mask], img_result)

demo.launch(server_name="0.0.0.0", server_port=7860)
