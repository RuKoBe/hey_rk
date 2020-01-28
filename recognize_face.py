import face_recognition
import cv2
import numpy as np
import eel
import glob
import ujson
from pynput.keyboard import Key, Controller

print(
    "                                                 ___\r\n" +
    "                                              .'/   \\\r\n" +
    "   .              __.....__                  / /     \\\r\n" +
    " .'|          .-''         '. .-.          .-| |     | \r\n" +
    "<  |         /     .-''''-.  `.\ \        / /| |     | \r\n" +
    " | |        /     /________\   \\\\ \      / / |/`.   .' \r\n" +
    " | | .'''-. |                  | \ \    / /   `.|   |  \r\n" +
    " | |/.'''. \\\\    .-------------'  \ \  / /     ||___|  \r\n" +
    " |  /    | | \    '-.____...---.   \ `  /      |/___/  \r\n" +
    " | |     | |  `.             .'     \  /       .'.--.  \r\n" +
    " | |     | |    `''-...... -'       / /       | |    | \r\n" +
    " | '.    | '.                   |`-' /        \_\    / \r\n" +
    " '---'   '---'                   '..'          `''--'  \r\n"
)
print("By Ruben Kober")
print("Starting UI...")
print("Loading methods...")
eel.init("ui")
eel.setProgress(25, "Loading methods...")


def write(data):
    f = open("config.json", "w")
    f.write(ujson.dumps(data))
    f.close()


print("Looking for config...")
startpage = "ui.html"
config = {}
with open("config.json", "r") as configfile:
    config = ujson.load(configfile)
if config["firstrun"] == True:
    print("HEY! firstrun detected starting in setup mode!")
    startpage = "setup.html"
eel.setProgress(50, "Initializing Variables...")


@eel.expose
def getConfig():
    return str(ujson.dumps(config))


@eel.expose
def setup(wm, gr, logo, css):
    config["welcomemessage"] = wm
    config["greeting"] = gr
    config["logo"] = logo
    config["css"] = css
    config["firstrun"] = False
    write(config)


@eel.expose
def run():
    print("Initializing Variables...")
    face_locations = []
    face_encodings = []
    face_names = []
    process_this_frame = True
    wereFace = False
    name = ""
    eel.setProgress(75, "Loading Persons...")
    print("Loading Persons...")
    known_face_encodings = []
    for face in glob.glob("known_people/*.jpg"):
        known_face_encodings.append(face_recognition.face_encodings(face_recognition.load_image_file(face))[0])
    known_face_names = open("known_people/names.txt").readlines()
    video_capture = cv2.VideoCapture(0)
    eel.setProgress(100, "Finished")
    print("HEY! is up and running!")
    while True:
        ret, frame = video_capture.read()
        small_frame = cv2.resize(frame, (0, 0), fx=1, fy=1)
        rgb_small_frame = small_frame[:, :, ::-1]
        if process_this_frame:
            face_locations = face_recognition.face_locations(rgb_small_frame)
            face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

            face_names = []
            for face_encoding in face_encodings:
                matches = face_recognition.compare_faces(known_face_encodings, face_encoding)
                name = "unknown"
                face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
                best_match_index = np.argmin(face_distances)
                if matches[best_match_index]:
                    name = known_face_names[best_match_index]

                face_names.append(name)

        process_this_frame = not process_this_frame
        if name in known_face_names:
            currentlyDetectedPerson = name
            eel.setUser(name)
            eel.welcome(name)
            eel.sleep(3)
            wereFace = True
            name = ""
        else:
            eel.sleep(.1)
            if wereFace:
                eel.wait()
                wereFace = False
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    video_capture.release()


@eel.expose
def full():
    Controller().press(Key.f11)


eel.start("startup.html?r=" + startpage)
