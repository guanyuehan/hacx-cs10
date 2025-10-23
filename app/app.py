from flask import Flask, url_for, request, render_template, jsonify
import os
from PIL import Image

app = Flask(__name__)

def predict_image(img):
	# the thing goes here
	return 0





@app.route('/',methods = ['GET','POST'])
def index():
	pred = None
	if request.method == 'POST':
		if 'image' not in request.files or request.files['image'] == "":
			pred = "no file"
		else:
			try:
				image = request.files['image']
				img = Image.open(image.stream).convert('RGB')
				pred = predict_image(img)
			except Exception as e:
				pred = f"Error: {str(e)}"
			
	return render_template('index.html',pred = pred)



if __name__ == '__main__':
	app.run(debug=False)