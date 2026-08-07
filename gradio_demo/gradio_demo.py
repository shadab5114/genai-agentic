import gradio as gr
#from huggingface_hub import HfFolder

def add_numbers(Num1, Num2):
    return Num1 + Num2

#Define the interface

demo = gr.Interface(
    fn=add_numbers,
    inputs=[gr.Number(label="Number 1"), gr.Number(label="Number 2")],
    outputs=gr.Number(label="Result")
)

# Launch the interface
demo.launch(server_name="127.0.0.1", server_port= 7860)
