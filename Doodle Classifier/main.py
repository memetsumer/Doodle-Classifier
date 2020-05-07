from flask import Flask, Response, json, render_template, request
from flask_assets import Bundle, Environment
from doodle_classifier import DoodleClassifier
import numpy as np

app = Flask(__name__)

js = Bundle('jquery.min.js', 'p5.min.js', 'p5.dom.js', 'sketch.js', output='gen/main.js')

assets = Environment(app)

assets.register('main_js', js)


@app.route('/', methods=["GET", "POST"])
def index():
    if request.method == 'GET':
        return render_template("index.html")
    if request.method == 'POST':
        dc = DoodleClassifier()
        data = request.json
        vectorized_inputs = np.array(data).reshape((-1, 1))
        guess = np.argmax(dc.predict_doodle(vectorized_inputs))
        if guess == 0:
            guess = "It's an ice cream!"
        elif guess == 1:
            guess = "It's a kitty cat!"
        elif guess == 2:
            guess = "It looks like the Eiffel Tower!"
        response = {'answer': guess}
        return Response(json.dumps(response), mimetype='application/json')


if __name__ == '__main__':
    app.run(debug=True)
