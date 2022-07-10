import numpy as np
import cv2
import time
import os
from ctypes import CDLL

face_cascade = cv2.CascadeClassifier(
    'haarcascade_frontalface_default.xml')  # Oggetto del face detector

'''
Funzione che taglia alcuni parti della foto, permettendo all'algoritmo di operare solo con quelle più rilevanti.
Limite: necessario registrare il volto in maniera frontale
:param image: immagine da tagliare
:param faces_coord: cordinate per delimitare il viso
:return faces: coordinate del volto escluso le parti laterali
'''


def cut_faces(image, faces_coord) -> list:

    faces = []
    for (x, y, w, h) in faces_coord:  # Ritaglia parti del viso

        w_rm = int(0.2 * w / 2)
        faces.append(image[y: y + h, x + w_rm:  x + w - w_rm])

    return faces


'''
Funzione che permette l'aggiunta di nuovi utenti al dataset creando opportunamento le directory per ognuno
:param None
:return None
'''


def add_person() -> None:

    # prende in input  il nome della persona che si vuole registrare
    person_name = input('Inserisci il nome della persona: ').lower()
    peopleFolder = 'people_folder'
    folder = peopleFolder + '/' + person_name

    if not os.path.exists(peopleFolder):

        os.mkdir(peopleFolder)

    # Controlla che non sia presente già una cartella con lo stesso nome
    if not os.path.exists(folder):

        input("Verranno scattate 20 foto. Premere ENTER per avviare la procedura.")
        os.mkdir(folder)  # Crea un nuovo folder per salvare le foto
        video = cv2.VideoCapture(0)
        # Carica l'Haarcascade per identificare il volto.
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
        cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)

        # viene avviato il ciclo per scattare le 20 foto
        for counter in range(1, 21):

            _, frame = video.read()
            if counter == 1:

                time.sleep(6)

            else:

                time.sleep(1)

            # Trova le coordinate di tutta le facce nel frame
            faces = detector.detectMultiScale(frame)

            if len(faces):  # Se sono state identificate le coordinate

                # Rimuove le parti inutili del volto
                cut_face = cut_faces(frame, faces)
                face_bw = cv2.cvtColor(cut_face[0], cv2.COLOR_BGR2GRAY)
                # Equalizzazione Istogramma
                face_bw_eq = cv2.equalizeHist(face_bw)
                # Effettua un Resizing dell'immagine in 100x100 pixel
                face_bw_eq = cv2.resize(
                    face_bw_eq, (100, 100), interpolation=cv2.INTER_CUBIC)
                # cv2.imshow('Face Recogniser', face_bw_eq)
                cv2.imwrite(folder + '/' + str(counter) + '.jpg', face_bw_eq)
                print('Images Saved:' + str(counter))
                # Mostra la faccia che è stata salvata
                cv2.imshow('Saved Face', face_bw_eq)

            cv2.imshow('Video Feed', frame)
            cv2.waitKey(50)

    else:

        # Viene visualizzato un messaggi di errore in caso il folder sia già presente
        print("This name already exists.")


'''
Funzione che permette di riconoscere il viso il real-time
:param lock: variabile che identifica la funziona di blocco schermo
:return None
'''


def live(lock) -> None:

    cv2.namedWindow('Predicting for')
    images = []
    labels = []
    labels_dic = {}
    people = []
    for person in os.listdir("people_folder"):
        if not person.startswith("."):
            people.append(person)

    threshold = 100  # Soglia per l'algoritmo di riconoscimento facciale/distanza consentita
    # da un'altra faccia

    for i, person in enumerate(people):

        labels_dic[i] = person
        for image in os.listdir("people_folder/" + person):

            images.append(cv2.imread(
                'people_folder/' + person + '/' + image, 0))
            labels.append(i)

    labels = np.array(labels)

    # rec_eig = cv2.face.EigenFaceRecognizer_create()
    # Crea un oggetto LBHP face recognizer
    rec_lbhp = cv2.face.LBPHFaceRecognizer_create()
    rec_lbhp.train(images, labels)  # Addestra il modello
    cv2.namedWindow('face')
    webcam = cv2.VideoCapture(0)
    while True:

        _, frame = webcam.read()
        # Prende le coordinate del volto nel frame
        faces = face_cascade.detectMultiScale(frame, 1.3, 5)
        if len(faces):

            # Taglia la faccia per inserirla nel nostro modello predittivo
            cut_face = cut_faces(frame, faces)
            face = cv2.cvtColor(cut_face[0], cv2.COLOR_BGR2GRAY)
            face = cv2.equalizeHist(face)  # Equalizzazione Istogramma
            # Ridimensiona l'immagine del volto.
            face = cv2.resize(face, (100, 100), interpolation=cv2.INTER_CUBIC)
            cv2.imshow('face', face)
            collector = cv2.face.StandardCollector_create()
            rec_lbhp.predict_collect(face, collector)
            # Trova la faccia più vicina alla nostra (confrontando le distanze e selezionando la minima)
            conf = collector.getMinDist()
            print('Confidence ', conf)
            pred = collector.getMinLabel()
            if conf < threshold:  # Se un match del volto è trovato

                # Prendi il nome della persona, maiuscolo
                txt = labels_dic[pred].upper()

            else:

                txt = 'Sconosciuto'  # Se non riconosciuto, segnalo come sconosciuto
                if lock:  # se lock è vera allora procedo con il blocco schermo
                    loginPF = CDLL(
                        '/System/Library/PrivateFrameworks/login.framework/Versions/Current/login')
                    result = loginPF.SACLockScreenImmediate()

            # Inserisce il testo nel frame corrente
            cv2.putText(frame, txt, (faces[0][0], faces[0][1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)
            print(faces)
            cv2.rectangle(frame, (faces[0][0], faces[0][1]), (faces[0][0] + faces[0][2], faces[0]
                          [1] + faces[0][3]), (255, 255, 0), 8)  # Crea un rettangolo attorno al volto
            cv2.putText(frame, "ESC to exit", (
                5, frame.shape[0] - 10), cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)

        cv2.imshow("Live", frame)  # Mostra il frame

        if cv2.waitKey(20) & 0xFF == 27:

            destroy()
            break


'''
funzione che termina i processi di face recognition
:param None
:return None
'''


def destroy() -> None:

    cv2.destroyAllWindows()
