from flask import Flask

app = Flask('ping')

@app.route('/')
def ping():
    return "pong"

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=9666)