from flask import Flask, render_template, request, url_for, jsonify

app = Flask(__name__, static_folder='assets')


@app.route('/')
def home():
        
    return render_template('index.html')
     



if __name__ == '__main__':
     app.run(debug=True)
    