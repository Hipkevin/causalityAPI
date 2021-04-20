from flask import Flask, request, render_template
from config import Config
from modelConfig import ModelConfig
from util.model import BiLstmCRF

config = Config()
modelConfig = ModelConfig()
model = BiLstmCRF(modelConfig)


app = Flask(__name__)


@app.route('/')
def home():
    result = ''
    trigger = ''

    return render_template('index.html', text=result, trigger=trigger, config=config)


@app.route('/predict/')
def predict():
    result = ''
    trigger = ''

    return render_template('predict.html', text=result, trigger=trigger, config=config)

@app.route('/predict/result', methods=['GET', 'POST'])
def result():
    data = request.form
    text = str(data['text'])
    trigger = str(data['trigger'])

    result = str(list(text))
    return render_template('predict.html', text=result, trigger=trigger)


if __name__ == '__main__':
    app.run(host=config.ip, port=config.port, debug=True)
