from flask import Flask,session,request,redirect,render_template
import re
from apscheduler.schedulers.background import BackgroundScheduler
import subprocess
import os
from pytz import utc

app = Flask(__name__)
app.secret_key = 'this is secret_key you know ?'

from views.page import page
from views.user import user
app.register_blueprint(page.pb)
app.register_blueprint(user.ub)

@app.route('/')
def hello_world():  # put application's code here
    return session.clear()

@app.before_request
def before_reuqest():
    pat = re.compile(r'^/static')
    if re.search(pat,request.path):return
    elif request.path == '/user/login' or request.path == '/user/register':return
    elif session.get('username'):return
    return redirect('/user/login')

@app.route('/<path:path>')
def catch_all(path):
    return render_template('404.html')

def run_script():
    current_dir = os.path.dirname(os.path.abspath(__file__))
    spider_script = os.path.join(current_dir, 'spider', 'main.py')
    # cutComments_script = os.path.join(current_dir, 'utils', 'cutComments.py')
    # cipingTotal_script = os.path.join(current_dir, 'utils', 'cipingTotal.py')

    scripts = [
        ("Spider Script", spider_script),
        # ("Cut Comments Script", cutComments_script),
        # ("Ciping Total Script", cipingTotal_script)
    ]

    for script_name, script_path in scripts:
        try:
            print(f"Running {script_name}...")
            subprocess.run(['python', script_path], check=True)
            print(f"{script_name} finished successfully.")
        except subprocess.CalledProcessError as e:
            print(f"An error occurred while running {script_name}: {e}")


if __name__ == '__main__':
    scheduler = BackgroundScheduler(timezone=utc)
    scheduler.add_job(run_script, 'interval', hours=5)
    scheduler.start()

    try:
        app.run()
    finally:
        scheduler.shutdown()