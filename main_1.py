import flet as ft
from flet import *
import threading
import cv2
import os
import base64
import sqlite3
import pickle
import face_recognition

trained_model_file = 'trained_model.pkl'
face_recognition_tolerance = 0.5

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

    cursor.execute('SELECT * FROM admin')
    if not cursor.fetchone():
        cursor.execute('INSERT INTO admin (username, password) VALUES (?, ?)', ('admin', 'admin123'))
    
    conn.commit()
    conn.close()

def load_images_from_folder(folder):
    images = []
    for filename in os.listdir(folder):
        img_path = os.path.join(folder, filename)
        if os.path.isfile(img_path):
            img = cv2.imread(img_path)
            if img is not None:
                images.append(img)
    return images

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

def train_model():
    data_folder = 'images'
    faces_encodings, labels = prepare_training_data(data_folder)
    with open(trained_model_file, 'wb') as file:
        pickle.dump((faces_encodings, labels), file)
    print("Model trained and saved.")

if not os.path.isfile(trained_model_file):
    train_model()

def main(page: Page):
    create_database()

    BG = '#041955'
    FG = '#3450a1'

    global is_authenticated
    is_authenticated = False

    def login_page(page: Page):
        username_input = TextField(label="Username", width=300)
        password_input = TextField(label="Password", password=True, can_reveal_password=True, width=300)

        def authenticate(e):
            username = username_input.value
            password = password_input.value

            conn = sqlite3.connect('app_database.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM admin WHERE username=? AND password=?", (username, password))
            admin = cursor.fetchone()
            conn.close()

            if admin:
                global is_authenticated
                is_authenticated = True
                page.go('/main')
            else:
                print("Error: Invalid credentials.")
                page.snack_bar = SnackBar(content=Text("Invalid credentials, please try again."), open=True)
                page.update()

        login_button = ElevatedButton(text="Login", on_click=authenticate)

        login_container = Container(
            expand=True,
            alignment=alignment.center,
            content=Column(
                horizontal_alignment='center',
                controls=[
                    Text("Login", size=30, weight='bold'),
                    username_input,
                    password_input,
                    login_button
                ]
            )
        )

        page.views.clear()
        page.views.append(View('/', [login_container]))
        page.update()

    def shrink(e):
        try:
            page_2.controls[0].width = 120
            page_2.controls[0].scale = ft.transform.Scale(0.8, alignment=alignment.center_right)
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
            page_2.controls[0].scale = ft.transform.Scale(1, alignment=alignment.center_right)
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

    def capture_frames(image_control, camera_index, detect_faces=False):
        try:
            cap = cv2.VideoCapture(camera_index)
            if not cap.isOpened():
                print(f"Error: Could not open camera {camera_index}.")
                return

            while True:
                ret, frame = cap.read()
                if not ret:
                    print(f"Error: Could not read frame from camera {camera_index}.")
                    break
                frame = cv2.flip(frame, 1)
                if detect_faces:
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

    def search_user(e):
        search_name_input = TextField(label="Enter Name")

        def fetch_user_details(e):
            search_name = search_name_input.value
            if not search_name:
                print("Error: Name field is empty.")
                return

            conn = sqlite3.connect('app_database.db')
            cursor = conn.cursor()
            cursor.execute("SELECT * FROM users WHERE name=?", (search_name,))
            user = cursor.fetchone()
            conn.close()

            if user:
                user_id, name, address, phone, email = user

                user_image_path = None
                user_folder = os.path.join('images', name)
                if os.path.exists(user_folder):
                    for filename in os.listdir(user_folder):
                        if filename.endswith(".jpg") or filename.endswith(".png"):
                            user_image_path = os.path.join(user_folder, filename)
                            break

                user_image = Image(src=f"data:image/jpeg;base64,{base64.b64encode(open(user_image_path, 'rb').read()).decode()}" if user_image_path else "")

                user_details_dialog = AlertDialog(
                    modal=True,
                    title=Text("User Details"),
                    content=Column(controls=[
                        Text(f"Name: {name}"),
                        Text(f"Address: {address}"),
                        Text(f"Phone: {phone}"),
                        Text(f"Email: {email}"),
                        Container(content=user_image, height=200, width=300),
                    ]),
                    actions=[
                        ElevatedButton(text="Close", on_click=lambda e: close_user_details_dialog())
                    ]
                )

                def close_user_details_dialog():
                    user_details_dialog.open = False
                    page.update()

                page.overlay.append(user_details_dialog)
                user_details_dialog.open = True
                page.update()
            else:
                print("Error: User not found.")

        search_button = ElevatedButton(text="Search", on_click=fetch_user_details)
        close_button = ElevatedButton(text="Close", on_click=lambda e: close_search_dialog())

        def close_search_dialog():
            search_dialog.open = False
            page.update()

        search_dialog = AlertDialog(
            modal=True,
            title=Text("Search User"),
            content=Column(controls=[
                search_name_input,
                search_button,
            ]),
            actions=[close_button]
        )

        page.overlay.append(search_dialog)
        search_dialog.open = True
        page.update()

    def add_camera(e):
        rtsp_link_input = TextField(label="Enter RTSP Link")

        def save_camera(e):
            rtsp_link = rtsp_link_input.value
            if not rtsp_link:
                print("Error: RTSP Link field is empty.")
                return

            new_webcam_image = Image(expand=True)
            threading.Thread(target=capture_frames, args=(new_webcam_image, 3, True, rtsp_link), daemon=True).start()

            first_page_contents.controls.append(
                Container(
                    expand=True,
                    alignment=alignment.center,
                    content=new_webcam_image,
                )
            )

            add_camera_dialog.open = False
            page.update()

        save_button = ElevatedButton(text="Save", on_click=save_camera)
        close_button = IconButton(icon=ft.icons.CLOSE, on_click=lambda e: close_add_camera_dialog())

        def close_add_camera_dialog():
            add_camera_dialog.open = False
            page.update()

        add_camera_dialog = AlertDialog(
            modal=True,
            title=Text("Add Camera"),
            content=Column(controls=[
                rtsp_link_input,
                save_button,
            ]),
            actions=[close_button]
        )

        page.overlay.append(add_camera_dialog)
        add_camera_dialog.open = True
        page.update()

    def register_user(e):
        name_input = TextField(label="Name")
        address_input = TextField(label="Address")
        phone_input = TextField(label="Phone")
        email_input = TextField(label="Email")
        captured_images = []
        capture_in_progress = [False]

        webcam_image = Image(expand=True)
        threading.Thread(target=capture_frames, args=(webcam_image, 0), daemon=True).start()

        def capture_images(e):
            if capture_in_progress[0]:
                return
            capture_in_progress[0] = True

            cap = cv2.VideoCapture(0)
            if not cap.isOpened():
                print("Error: Could not open registration camera.")
                capture_in_progress[0] = False
                return

            for i in range(3):
                ret, frame = cap.read()
                if ret:
                    captured_images.append(frame)
                    print(f"Captured image {i + 1}")
                else:
                    print(f"Error capturing image {i + 1}")
            cap.release()

            capture_in_progress[0] = False
            page.update()

        def save_user(e):
            name = name_input.value
            address = address_input.value
            phone = phone_input.value
            email = email_input.value

            if not all([name, address, phone, email]):
                print("Error: All fields must be filled.")
                return

            user_dir = os.path.join('images', name)
            os.makedirs(user_dir, exist_ok=True)

            for i, img in enumerate(captured_images):
                image_path = os.path.join(user_dir, f"{name}_{i + 1}.jpg")
                cv2.imwrite(image_path, img)
                print(f"Image saved at {image_path}")

            conn = sqlite3.connect('app_database.db')
            cursor = conn.cursor()
            cursor.execute("INSERT INTO users (name, address, phone, email) VALUES (?, ?, ?, ?)",
                           (name, address, phone, email))
            conn.commit()
            conn.close()

            print("User data saved successfully")
            train_model()
            register_dialog.open = False
            page.update()

        save_button = ElevatedButton(text="Save", on_click=save_user)
        capture_button = ElevatedButton(text="Capture Images", on_click=capture_images)
        close_button = IconButton(icon=ft.icons.CLOSE, on_click=lambda e: close_register_dialog())

        def close_register_dialog():
            register_dialog.open = False
            page.update()

        register_dialog = AlertDialog(
            modal=True,
            title=Row([Text("Register New User"), close_button], alignment=ft.MainAxisAlignment.SPACE_BETWEEN),
            content=Column(controls=[
                name_input,
                address_input,
                phone_input,
                email_input,
                Container(content=webcam_image, height=200, width=300),
                capture_button,
                save_button
            ], scroll=ft.ScrollMode.AUTO)
        )

        page.overlay.append(register_dialog)
        register_dialog.open = True
        page.update()

    def route_change(route):
        global is_authenticated
        print(f"Route changed to {route}")

        try:
            page.views.clear()
            if route == '/' and not is_authenticated:
                login_page(page)
            elif route == '/main' and is_authenticated:
                page.views.append(View('/main', [main_page_content]))
            else:
                login_page(page)
            page.update()
        except Exception as err:
            print(f"Error in route_change: {err}")

    webcam_image_1 = Image(expand=True)
    webcam_image_2 = Image(expand=True)
    
    threading.Thread(target=capture_frames, args=(webcam_image_1, 1, True), daemon=True).start()
    threading.Thread(target=capture_frames, args=(webcam_image_2, 2, True), daemon=True).start()

    first_page_contents = Container(
        expand=True,
        content=Column(
            expand=True,
            controls=[
                Row(
                    alignment=ft.MainAxisAlignment.SPACE_BETWEEN,
                    controls=[
                        Container(
                            on_click=lambda e: shrink(e),
                            content=Icon(ft.icons.MENU)
                        ),
                        Row(
                            controls=[
                                Icon(ft.icons.NOTIFICATIONS_OUTLINED)
                            ],
                        ),
                    ],
                ),
                Container(height=20),
                Text("All Cameras"),
                Row(
                    expand=True,
                    alignment=alignment.center,
                    controls=[
                        Column(
                            expand=True,
                            alignment=alignment.center,
                            controls=[
                                Text("Entry Camera"),
                                Container(
                                    expand=True,
                                    alignment=alignment.center,
                                    content=webcam_image_1,
                                ),
                            ],
                        ),
                        Column(
                            expand=True,
                            alignment=alignment.center,
                            controls=[
                                Text("Exit Camera"),
                                Container(
                                    expand=True,
                                    alignment=alignment.center,
                                    content=webcam_image_2,
                                ),
                            ],
                        ),
                    ],
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
                    alignment=ft.MainAxisAlignment.END,
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
                Text('Wahab\nNaseer', size=32, weight=ft.FontWeight.BOLD),
                Container(height=25),
                Container(
                    on_click=search_user,
                    content=Row(
                        controls=[
                            Icon(ft.icons.SEARCH, color='white60'),
                            Text('Search', size=15, weight=ft.FontWeight.W_300, color='white', font_family='poppins')
                        ],
                    ),
                ),
                Container(height=5),
                Container(
                    on_click=register_user,
                    content=Row(
                        controls=[
                            Icon(ft.icons.PERSON_ADD_ALT, color='white60'),
                            Text('Register', size=15, weight=ft.FontWeight.W_300, color='white', font_family='poppins'),
                        ],
                    ),
                ),
                Container(height=5),
                Container(
                    on_click=add_camera,
                    content=Row(
                        controls=[
                            Icon(ft.icons.VIDEO_CAMERA_FRONT, color='white60'),
                            Text('Add camera', size=15, weight=ft.FontWeight.W_300, color='white', font_family='poppins')
                        ],
                    ),
                ),
            ],
        ),
    )

    page_2 = Row(
        alignment=ft.MainAxisAlignment.END,
        expand=True,
        controls=[
            Container(
                expand=True,
                bgcolor=FG,
                border_radius=border_radius.all(35),
                animate=animation.Animation(600, ft.AnimationCurve.DECELERATE),
                animate_scale=animation.Animation(400, curve=ft.AnimationCurve.DECELERATE),
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
        '/': View("/", [container]),
        '/main': View("/main", [container]),
    }

    page.on_route_change = route_change
    page.go(page.route)

app(target=main, assets_dir='assets')
