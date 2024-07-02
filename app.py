from flask import Flask,session,request,redirect,render_template

app = Flask(__name__)

@app.route('/')
def hello_world():  # put application's code here
    return session.clear()

if __name__ == '__main__':
    app.run()