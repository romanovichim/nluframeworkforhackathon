from flask import Flask, render_template, request, send_from_directory
import os
from carrydialog import girl_answer

UPLOAD_FOLDER = os.path.basename('music')

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


@app.route("/")
def main():
    return render_template("main.html")

@app.route("/children")
def children():
    return render_template("children.html")

@app.route("/carry")
def carry():
    img = './static/carry.jpg'
    return render_template("carry.html", img=img)

@app.route("/get")
def get_bot_response():
    userText = request.args.get('msg')
    #return str(english_bot.get_response(userText))
    return str(userText)

@app.route("/getcarry")
def get_carry_response():
    userText = request.args.get('msg')
    #return str(english_bot.get_response(userText))
    return girl_answer(userText)

@app.route('/music/<path:filename>')
def download_file(filename):
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

if __name__ == "__main__":
    app.run()
