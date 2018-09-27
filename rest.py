from fileinput import filename

from flask import Flask, request

class recive_video_file:
    def __init__(self):
        pass

# Initialize the Flask application
app = Flask(__name__)

path = "/opt/lampp/htdocs/INSIDE/upload"


# start_time = time.time()

@app.route('/upload', methods=['POST'])
def upload_file():
    videofilename = request.get_json()
    print videofilename

    return "Worked"

if __name__ == '__main__':
     app.run()
# Run the app :)
# if __name__ == '__main__':
#     fragmentation.framerate.calframerate(path)
