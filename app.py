from flask import Flask,session,request,redirect,render_template

app = Flask(__name__)

@app.route('/')
def hello_world():  # put application's code here
    return session.clear()

@app.route('/<path:path>')
def catch_all(path):
    return render_template('404.html')

if __name__ == '__main__':
    app.run()