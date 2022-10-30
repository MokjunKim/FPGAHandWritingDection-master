from flask import Flask, request, render_template
from PIL import Image
from io import BytesIO
import base64
from recog import inference

app = Flask(__name__)

@app.route('/')
def onHome():
	return render_template('home.html')

@app.route('/result', methods=['POST'])
def onResult():
	reqData = request.form.get('img')
	if reqData != '':
		img = reqData[22:] + '=' * (4 - len(reqData) % 4)
		img = base64.b64decode(img)
		img = BytesIO(img)
		img = Image.open(img)
		img.save('image.png')
		(fomula, result) = inference()

	return render_template('result.html', formula=fomula, result=result)

if __name__ == '__main__':
	app.run(host='0.0.0.0', port=3004)
