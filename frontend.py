import gradio as gr
from PIL import Image

def process_image(image, coordinates):
    # 这个函数将由你来实现
    # 例如：result_image = your_function(image, coordinates)
    result_image = image  # 暂时返回原图，之后你可以替换成你的函数逻辑
    return result_image

# 存储点击位置的列表
click_coordinates = []

def get_coordinates(img, evt: gr.SelectData):
    global click_coordinates
    click_coordinates.append((evt.index[0], evt.index[1]))
    return img, click_coordinates, process_image(img, click_coordinates)

def clear_coordinates():
    global click_coordinates
    click_coordinates = []
    return ""

with gr.Blocks() as demo:
    with gr.Row():
        img_input = gr.Image(type="pil", label="上传图片")
        img_output = gr.Image(type="pil", label="处理后图片")
        
    coordinates_display = gr.Textbox(label="点击位置坐标", interactive=False)
    clear_button = gr.Button("清除点击")

    img_input.select(get_coordinates, [img_input], [img_input, coordinates_display, img_output])
    clear_button.click(clear_coordinates, [], coordinates_display)

demo.launch()
