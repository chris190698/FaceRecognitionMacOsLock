import numpy as np
import cv2
import time
import os
from ctypes import CDLL

'''
Funzione che permette l'aggiunta di nuovi utenti al dataset creando opportunamento le directory per ognuno
@:param None
@:return None
'''
def add_person() -> None:

    # prende in input  il nome della persona che si vuole registrare
    person_name = input('Inserisci il nome della persona: ').lower()
    folder = 'people_folder' + '/' + person_name

    # Controlla che non sia presente già una cartella con lo stesso nome
    if not os.path.exists(folder):

        input("Verranno scattate 20 foto. Premere ENTER per avviare la procedura.")
        os.mkdir(folder)  # Crea un nuovo folder per salvare le foto
        video = cv2.VideoCapture(0)
        detector = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Carica l'Haarcascade per identificare il volto.
        timer = 0
        cv2.namedWindow('Video Feed', cv2.WINDOW_AUTOSIZE)
        cv2.namedWindow('Saved Face', cv2.WINDOW_NORMAL)

        #viene avviato il ciclo per scattare le 20 foto
        for counter in range(20):

            _, frame = video.read()
            if counter == 1:
                time.sleep(6)
            else:
                time.sleep(1)

            faces = detector.detectMultiScale(frame)  # Trova le coordinate di tutte le facce nel frame

            if len(faces):  # Se abbiamo alcuni volti

                cut_face = cut_faces(frame, faces)  # Rimuove le parti inutili del volto

                face_bw = cv2.cvtColor(cut_face[0], cv2.COLOR_BGR2GRAY)

                face_bw_eq = cv2.equalizeHist(face_bw)  # Equalizzazione Istogramma
                face_bw_eq = cv2.resize(face_bw_eq, (100, 100),
                                        interpolation=cv2.INTER_CUBIC)  # Effettua un Resizing dell'immagine in 100x100 pixel
                # cv2.imshow('Face Recogniser', face_bw_eq)

                cv2.imwrite(folder + '/' + str(counter) + '.jpg',
                            face_bw_eq)
                print('Images Saved:' + str(counter))
                counter += 1
                cv2.imshow('Saved Face', face_bw_eq)  # Mostra la faccia che è stata salvata

            cv2.imshow('Video Feed', frame)
            cv2.waitKey(50)

    else:

        # Viene visualizzato un messaggi di errore in caso il folder sia già presente
        print("This name already exists.")