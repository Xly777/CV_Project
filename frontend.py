import gradio as gr
from PIL import Image

def process_image(image, coordinates):
    ##TODO
    result_image = image  
    return result_image

def get_coordinates(img, evt: gr.SelectData):
    return img, (evt.index[0], evt.index[1])

def main(image, coordinates):
    return process_image(image, coordinates)

with gr.Blocks() as demo:
    with gr.Row():
        img_input = gr.Image(type="pil", label="上传图片")
        img_output = gr.Image(type="pil", label="处理后图片")
        
    coordinates_display = gr.Textbox(label="点击位置坐标", interactive=False)
    
    img_input.select(get_coordinates, [img_input], [img_input, coordinates_display]).then(
        main, [img_input, coordinates_display], img_output)

demo.launch()
