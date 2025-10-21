from flask import logging, Flask, render_template, request, flash , abort , jsonify
from flask_cors import CORS

app = Flask(__name__)
app.secret_key = "Susastho.ai"
CORS(app)

from nlp_api.app import llm_infer
#from tts_api.app import to_audio
#from asr_api.app import handler
#from nlp_api.app_t1 import llm_infer

@app.route('/')
def index():
    return "NLP Backend API"


if __name__ == "__main__":
    app.run(debug=False)
