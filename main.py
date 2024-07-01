import cv2
import base64
import pickle
import sqlite3
from flet import *
import threading
import face_recognition
import os

trained_model_file = 'trained_model.pkl'
face_recognition_tolerance = 0.6

def create_database():
    conn = sqlite3.connect('app_database.db')
    cursor = conn.cursor()
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS admin (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            username TEXT NOT NULL,
            password TEXT NOT NULL
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS users (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            name TEXT,
            address TEXT,
            phone TEXT,
            email TEXT
        )
    ''')
    cursor.execute('''
        CREATE TABLE IF NOT EXISTS records (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            user_id INTEGER,
            check_in_time TEXT,
            check_out_time TEXT,
            image BLOB,
            FOREIGN KEY (user_id) REFERENCES users (id)
        )
    ''')
    conn.commit()
    conn.close()

# Function to load images from folder
def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

# Function to prepare training data and generate encodings
def prepare_training_data(data_folder):
    labels = []
    faces_encodings = []

    for person_name in os.listdir(data_folder):
        person_folder = os.path.join(data_folder, person_name)
        if os.path.isdir(person_folder):
            images = load_images_from_folder(person_folder)
            for image in images:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                face_locations = face_recognition.face_locations(rgb_image)
                if face_locations:
                    encodings = face_recognition.face_encodings(rgb_image, face_locations)
                    if encodings:
                        faces_encodings.append(encodings[0])
                        labels.append(person_name)
    return faces_encodings, labels

# Function to train and save the model
def train_model():
    data_folder = 'images'  # Folder where user images are stored
    faces_encodings, labels = prepare_training_data(data_folder)
    with open(trained_model_file, 'wb') as file:
        pickle.dump((faces_encodings, labels), file)
    print("Model trained and saved.")

# Ensure the model is trained if not already
if not os.path.isfile(trained_model_file):
    train_model()

def main(page: Page):
    create_database()

    BG = '#041955'
    FWG = '#97b4ff'
    FG = '#3450a1'
    PINK = '#eb06ff'

    circle = Stack(
        controls=[
            Container(
                width=100,
                height=100,
                border_radius=border_radius.all(50),
                bgcolor='white12'
            ),
            Container(
                gradient=SweepGradient(
                    center=alignment.center,
                    start_angle=0.0,
                    end_angle=3,
                    stops=[0.5, 0.5],
                    colors=['#00000000', PINK],
                ),
                width=100,
                height=100,
                border_radius=border_radius.all(50),
                content=Row(
                    alignment='center',
                    controls=[
                        Container(
                            padding=padding.all(5),
                            bgcolor=BG,
                            width=90,
                            height=90,
                            border_radius=border_radius.all(50),
                            content=Container(
                                bgcolor=FG,
                                height=80,
                                width=80,
                                border_radius=border_radius.all(40),
                                content=CircleAvatar(
                                    opacity=0.8,
                                    foreground_image_src="/assets/images/1.png"
                                ),
                            ),
                        ),
                    ],
                ),
            ),
        ],
    )

    def shrink(e):
        try:
            page_2.controls[0].width = 120
            page_2.controls[0].scale = transform.Scale(0.8, alignment=alignment.center_right)
            page_2.controls[0].border_radius = border_radius.only(
                top_left=35,
                top_right=0,
                bottom_left=35,
                bottom_right=0
            )
            page_2.update()
        except Exception as err:
            print(f"Error in shrink: {err}")

    def restore(e):
        try:
            page_2.controls[0].width = 400
            page_2.controls[0].border_radius = border_radius.all(35)
            page_2.controls[0].scale = transform.Scale(1, alignment=alignment.center_right)
            page_2.update()
        except Exception as err:
            print(f"Error in restore: {err}")

    def update_frame(image_control, frame):
        try:
            img = cv2.imencode('.jpg', frame)[1].tobytes()
            image_control.src_base64 = base64.b64encode(img).decode('utf-8')
            page.update()
        except Exception as err:
            print(f"Error in update_frame: {err}")

    def capture_frames(image_control):
        try:
            cap = cv2.VideoCapture(2)
            if not cap.isOpened():
                print("Error: Could not open camera.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    print("Error: Could not read frame.")
                    break
                frame = cv2.flip(frame, 1)
                frame = recognize_faces(frame)
                update_frame(image_control, frame)

            cap.release()
        except Exception as err:
            print(f"Error in capture_frames: {err}")

    def recognize_faces(frame):
        try:
            if not os.path.isfile(trained_model_file):
                return frame

            with open(trained_model_file, 'rb') as file:
                faces_encodings, labels = pickle.load(file)

            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
                matches = face_recognition.compare_faces(faces_encodings, face_encoding, tolerance=face_recognition_tolerance)
                name = "Unknown"
                color = (0, 0, 255)

                if True in matches:
                    first_match_index = matches.index(True)
                    name = labels[first_match_index]
                    color = (0, 255, 0)

                cv2.rectangle(frame, (left, top), (right, bottom), color, 2)
                cv2.putText(frame, name, (left + 6, bottom - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            return frame
        except Exception as err:
            print(f"Error in recognize_faces: {err}")
            return frame

    def register_user(e):
        name_input = TextField(label="Name")
        address_input = TextField(label="Address")
        phone_input = TextField(label="Phone")
        email_input = TextField(label="Email")

        def capture_images_from_camera(e, user_dir, name):
            cap = cv2.VideoCapture(0)
            for i in range(3):
                ret, frame = cap.read()
                if ret:
                    image_path = os.path.join(user_dir, f"{name}_{i+1}.jpg")
                    cv2.imwrite(image_path, frame)
                    print(f"Image saved at {image_path}")
                else:
                    print(f"Error capturing image {i+1}")
            cap.release()
            capture_choice_dialog.open = False
            page.update()

        def upload_images_from_folder(e, user_dir, name):
            for i, uploaded_file in enumerate(e.files):
                file_extension = os.path.splitext(uploaded_file.name)[1]
                image_path = os.path.join(user_dir, f"{name}_{i+1}{file_extension}")
                with open(image_path, 'wb') as f:
                    f.write(uploaded_file.get_buffer())
                print(f"Image saved at {image_path}")
            capture_choice_dialog.open = False
            page.update()

        def save_user(e):
            name = name_input.value
            address = address_input.value
            phone = phone_input.value
            email = email_input.value

            user_dir = os.path.join('images', name)
            os.makedirs(user_dir, exist_ok=True)

            capture_choice_dialog.title = Text("Choose Image Source")
            capture_choice_dialog.content = Column(
                controls=[
                    ElevatedButton(text="Capture from Camera", on_click=lambda e: capture_images_from_camera(e, user_dir, name)),
                    FilePicker(on_result=lambda e: upload_images_from_folder(e, user_dir, name))
                ]
            )
            capture_choice_dialog.open = True
            page.update()

            conn = sqlite3.connect('app_database.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, address, phone, email) VALUES (?, ?, ?, ?)",
                           (name, address, phone, email))
            conn.commit()
            conn.close()

            print("User data saved successfully")

            # Train the model after saving the user
            train_model()

        save_button = ElevatedButton(text="Save", on_click=save_user)

        register_dialog = AlertDialog(
            modal=True,
            title=Text("Register New User"),
            content=Column(controls=[name_input, address_input, phone_input, email_input, save_button])
        )

        capture_choice_dialog = AlertDialog(
            modal=True,
            content=Text(""),
        )

        page.overlay.append(register_dialog)
        page.overlay.append(capture_choice_dialog)
        register_dialog.open = True
        page.update()

    webcam_image = Image(expand=True)
    threading.Thread(target=capture_frames, args=(webcam_image,), daemon=True).start()

    first_page_contents = Container(
        expand=True,
        content=Column(
            expand=True,
            controls=[
                Row(
                    alignment=MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        Container(
                            on_click=lambda e: shrink(e),
                            content=Icon(icons.MENU)
                        ),
                        Row(
                            controls=[
                                Icon(icons.SEARCH),
                                Icon(icons.NOTIFICATIONS_OUTLINED)
                            ],
                        ),
                    ],
                ),
                Container(height=20),
                Text("All Cameras"),
                Container(
                    expand=True,
                    alignment=alignment.center,
                    content=webcam_image,
                ),
            ],
        ),
    )

    page_1 = Container(
        expand=True,
        bgcolor=BG,
        border_radius=border_radius.all(35),
        padding=padding.only(left=5, top=6, right=20),
        content=Column(
            expand=True,
            controls=[
                Row(
                    alignment=MainAxisAlignment.END,
                    controls=[
                        Container(
                            border_radius=border_radius.all(25),
                            padding=padding.only(top=13, left=13),
                            height=50,
                            width=50,
                            bgcolor=FG,
                            on_click=lambda e: restore(e),
                            content=Text('<')
                        ),
                    ],
                ),
                Container(height=20),
                circle,
                Text('Wahab\nNaseer', size=32, weight=FontWeight.BOLD),
                Container(height=25),
                Row(
                    controls=[
                        Icon(icons.SEARCH, color='white60'),
                        Text('Search', size=15, weight=FontWeight.W_300, color='white', font_family='poppins')
                    ],
                ),
                Container(height=5),
                Container(
                    on_click=register_user,
                    content=Row(
                        controls=[
                            Icon(icons.PERSON_ADD_ALT, color='white60'),
                            Text('Register', size=15, weight=FontWeight.W_300, color='white', font_family='poppins'),
                        ],
                    ),
                ),
                Container(height=5),
                Row(
                    controls=[
                        Icon(icons.VIDEO_CAMERA_FRONT, color='white60'),
                        Text('Add camera', size=15, weight=FontWeight.W_300, color='white', font_family='poppins')
                    ],
                ),
            ],
        ),
    )

    page_2 = Row(
        alignment=MainAxisAlignment.END,
        expand=True,
        controls=[
            Container(
                expand=True,
                bgcolor=FG,
                border_radius=border_radius.all(35),
                animate=animation.Animation(600, AnimationCurve.DECELERATE),
                animate_scale=animation.Animation(400, curve=AnimationCurve.DECELERATE),
                padding=padding.only(top=5, left=20, right=20, bottom=5),
                content=Column(
                    expand=True,
                    controls=[
                        first_page_contents
                    ]
                )
            )
        ]
    )

    container = Container(
        expand=True,
        bgcolor=BG,
        border_radius=border_radius.all(35),
        content=Stack(
            controls=[
                page_1,
                page_2,
            ]
        )
    )

    pages = {
        '/': View(
            "/",
            [
                container
            ],
        )
    }

    def route_change(route):
        print(f"Route changed to {route}")
        try:
            page.views.clear()
            page.views.append(
                pages[page.route]
            )
            page.update()
        except Exception as err:
            print(f"Error in route_change: {err}")

    page.on_route_change = route_change
    page.go(page.route)

app(target=main, assets_dir='assets')
