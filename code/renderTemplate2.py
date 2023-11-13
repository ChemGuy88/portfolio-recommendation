"""
This is a test of Flask's `render_template` method, using the `template_folder` argument.
"""


from flask import Flask, render_template

app = Flask(__name__, template_folder="flaskTemplates")


@app.route("/")
def hello_world():
    return render_template("hello.html")
