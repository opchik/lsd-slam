import os
from src.models import App

images_path = os.path.join("data", "images")
real_data = os.path.join("data", "times.txt")


def setup_config(app: App):
	app.images_path = images_path
	app.real_data = real_data



