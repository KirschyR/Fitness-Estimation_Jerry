from flask import Flask, send_from_directory, render_template, jsonify
import os
import pathlib

APP = Flask(__name__, template_folder='templates', static_folder='static')

OUTDIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), 'outputs')

@APP.route('/')
def index():
    return render_template('index.html')

@APP.route('/list')
def list_files():
    tree = {}
    if os.path.isdir(OUTDIR):
        for week in sorted(os.listdir(OUTDIR)):
            week_path = os.path.join(OUTDIR, week)
            if not os.path.isdir(week_path):
                continue
            tree[week] = {}
            for state in sorted(os.listdir(week_path)):
                state_path = os.path.join(week_path, state)
                if not os.path.isdir(state_path):
                    continue
                files = [f for f in sorted(os.listdir(state_path)) if os.path.isfile(os.path.join(state_path, f))]
                tree[week][state] = files
    return jsonify(tree)

@APP.route('/files/<path:filename>')
def serve_file(filename):
    return send_from_directory(OUTDIR, filename)

if __name__ == '__main__':
    port = int(os.getenv('PORT', '5000'))
    APP.run(host='127.0.0.1', port=port, debug=True)
