from flask import Flask,render_template,request
import numpy as np

app=Flask(__name__)

@app.route("/")
def home():
    return render_template('index.html',prediction=None)

@app.route("/submit",methods=['POST','GET'])
def helper():
    if request.method=="POST":
        try:
            topic = request.form['topic']
            length = request.form['length']
            tone = request.form['tone']
            tone_map = {
                "0": "casual",
                "1": "formal",
                "2": "professional"
            }
            tone_label = tone_map.get(tone, "casual")
            print(f"Prompt: {topic}")
            print(f"Tone: {tone_label}")
            print(f"Length: {length}")
        except Exception as e:
            print(f"Error processing request: {str(e)}")
            return render_template('index.html', result=f"Error: {str(e)}")
    return render_template("blog.html")
        
if __name__=='__main__':
    app.run(debug=True)