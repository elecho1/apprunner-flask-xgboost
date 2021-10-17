from flask import Flask, jsonify

app = Flask(__name__)

@app.route("/")
def index():
    return "Hello World!"

@app.route("/api/v1/hello")
def hello():
    return jsonify({
        "message": "Hello World!"
    })

if __name__ == "__main__":
    app.run()
