"""
This is a test of Flask's `render_template` method
"""

from flask import Flask, render_template

app = Flask(__name__)


@app.route("/")
def hello_world():
    return render_template("hello1.html")
