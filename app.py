from flask import Flask

app = Flask(__name__)

@app.route('/')
def hello():
    return "小学期快乐！"

if __name__ == '__main__':
    app.run()