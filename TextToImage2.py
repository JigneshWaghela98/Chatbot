import tkinter as tk
from PIL import ImageTk
from diffusers import StableDiffusionPipeline
from torch import autocast

# Create app user interface
app = tk.Tk()
app.geometry("532x632")
app.title("Text to Image app")
app.configure(bg='black')

# Create input box on the user interface
prompt = tk.Entry(app, width=60)
prompt.place(x=10, y=10)

# Create a placeholder to show the generated image
img_placeholder = tk.Label(app, height=512, width=512)
img_placeholder.place(x=10, y=110)

# Download stable diffusion model from hugging face
modelid = "CompVis/stable-diffusion-v1-4"
device = "cpu"
stable_diffusion_model = StableDiffusionPipeline.from_pretrained(modelid)

# Generate image from text
def generate_image():
    """ This function generates an image from text with stable diffusion"""
    with autocast(device):
        image = stable_diffusion_model(prompt.get(), guidance_scale=8.5)["sample"][0]

    # Display the generated image on the user interface
    img = ImageTk.PhotoImage(image)
    img_placeholder.configure(image=img)

# Create a button to trigger image generation
trigger = tk.Button(app, text="Generate", command=generate_image)
trigger.place(x=206, y=60)

app.mainloop()
