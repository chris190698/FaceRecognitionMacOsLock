from faceRecFunction import *


face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')  # Oggetto del face detector

roi_gray = []


# Rimuove le parti laterali del volto

# Questo è fatto in modo tale che l'algoritmo debba lavorare solo con le parti rilevanti e più importanti dell'immagine
# Mi raccomando un limite consiste nel registrare il volto frontalmente.
def cut_faces(image, faces_coord):
    faces = []

    for (x, y, w, h) in faces_coord:  # Ritaglia parti del viso
        w_rm = int(0.2 * w / 2)
        faces.append(image[y: y + h, x + w_rm:  x + w - w_rm])

    return faces  # Restituisce le coordinate della faccia





# Effettua la face recognition in real time, premendo ESC chiude la face recognition

def live():
    cv2.namedWindow('Predicting for')
    images = []
    labels = []
    labels_dic = {}
    people = []
    for person in os.listdir("people_folder"):
        if not person.startswith("."):
            people.append(person)

    threshold = 80  # Soglia per l'algoritmo di riconoscimento facciale/distanza consentita
    # da un'altra faccia

    for i, person in enumerate(people):

        labels_dic[i] = person

        for image in os.listdir("people_folder/" + person):
            images.append(cv2.imread('people_folder/' + person + '/' + image, 0))
            labels.append(i)

    labels = np.array(labels)

    # rec_eig = cv2.face.EigenFaceRecognizer_create()
    rec_lbhp = cv2.face.LBPHFaceRecognizer_create()  # Crea un oggetto LBHP face recognizer

    rec_lbhp.train(images, labels)  # Addestra il modellos

    cv2.namedWindow('face')
    webcam = cv2.VideoCapture(0)
    while True:
        _, frame = webcam.read()

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # Prende le cordinate del volto nel frame

        if len(faces):
            cut_face = cut_faces(frame, faces)  # Taglia la faccia per inserirla nel nostro modello predittivo

            face = cv2.cvtColor(cut_face[0], cv2.COLOR_BGR2GRAY)
            face = cv2.equalizeHist(face)  # Equalizzazione Istogramma
            face = cv2.resize(face, (100, 100), interpolation=cv2.INTER_CUBIC)  # Ridimensiona l'immagine del volto.

            cv2.imshow('face', face)

            collector = cv2.face.StandardCollector_create()
            rec_lbhp.predict_collect(face, collector)
            conf = collector.getMinDist()  # Trova la faccia più vicina alla nostra (confrontando le distanze e selezionando la minima)

            print('Confidence ', conf)
            pred = collector.getMinLabel()
            txt = ''

            if conf < threshold:  # Se un match del volto è trovato
                txt = labels_dic[pred].upper()  # Prendi il nome della persona, maiuscolo
            else:
                txt = 'Sconosciuto'  # Se non riconosciuto, segnalo come sconosciuto



            cv2.putText(frame, txt,
                        (faces[0][0], faces[0][1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)  # Inserisce il testo nel frame corrente

            print(faces)
            cv2.rectangle(frame, (faces[0][0], faces[0][1]), (faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]),
                          (255, 255, 0), 8)  # Crea un rettangolo attorno al volto

            cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)

        cv2.imshow("Live", frame)  # Mostra il frame

        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyAllWindows()
            break

def lock():
    cv2.namedWindow('Predicting for')
    images = []
    labels = []
    labels_dic = {}
    people = []
    for person in os.listdir("people_folder"):
        if not person.startswith("."):
            people.append(person)

    threshold = 80  # Soglia per l'algoritmo di riconoscimento facciale/distanza consentita
    # da un'altra faccia

    for i, person in enumerate(people):

        labels_dic[i] = person

        for image in os.listdir("people_folder/" + person):
            images.append(cv2.imread('people_folder/' + person + '/' + image, 0))
            labels.append(i)

    labels = np.array(labels)

    # rec_eig = cv2.face.EigenFaceRecognizer_create()
    rec_lbhp = cv2.face.LBPHFaceRecognizer_create()  # Crea un oggetto LBHP face recognizer

    rec_lbhp.train(images, labels)  # Addestra il modellos

    cv2.namedWindow('face')
    webcam = cv2.VideoCapture(0)
    while True:
        _, frame = webcam.read()

        faces = face_cascade.detectMultiScale(frame, 1.3, 5)  # Prende le cordinate del volto nel frame

        if len(faces):
            cut_face = cut_faces(frame, faces)  # Taglia la faccia per inserirla nel nostro modello predittivo

            face = cv2.cvtColor(cut_face[0], cv2.COLOR_BGR2GRAY)
            face = cv2.equalizeHist(face)  # Equalizzazione Istogramma
            face = cv2.resize(face, (100, 100), interpolation=cv2.INTER_CUBIC)  # Ridimensiona l'immagine del volto.

            cv2.imshow('face', face)

            collector = cv2.face.StandardCollector_create()
            rec_lbhp.predict_collect(face, collector)
            conf = collector.getMinDist()  # Trova la faccia più vicina alla nostra (confrontando le distanze e selezionando la minima)

            print('Confidence ', conf)
            pred = collector.getMinLabel()
            txt = ''

            if conf < threshold:  # Se un match del volto è trovato
                txt = labels_dic[pred].upper()  # Prendi il nome della persona, maiuscolo
            else:
                txt = 'Sconosciuto'  # Se non riconosciuto, segnalo come sconosciuto

                #Le due righe di codice sottostanti, sfruttando il package ctypes permettono di effettuare la lock screen del pc in uso.
                loginPF = CDLL('/System/Library/PrivateFrameworks/login.framework/Versions/Current/login')
                result = loginPF.SACLockScreenImmediate()

            cv2.putText(frame, txt,
                        (faces[0][0], faces[0][1] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 3, (66, 53, 243), 2)  # Inserisce il testo nel frame corrente

            print(faces)
            cv2.rectangle(frame, (faces[0][0], faces[0][1]), (faces[0][0] + faces[0][2], faces[0][1] + faces[0][3]),
                          (255, 255, 0), 8)  # Crea un rettangolo attorno al volto

            cv2.putText(frame, "ESC to exit", (5, frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_PLAIN, 1.3, (66, 53, 243), 2, cv2.LINE_AA)

        cv2.imshow("Live", frame)  # Mostra il frame

        if cv2.waitKey(20) & 0xFF == 27:
            cv2.destroyAllWindows()
            break


while True:
    print("Hello there please select one of the below")
    print('Press 1 for adding a new face')
    print('Press 2 for the live recognition')
    print('Press 3 to test Lock Screen Functionality')
    print('Press 4 to exit')

    choice = int(input())

    if choice > 3 or choice < 1 or choice == '':
        print('Please select a valid choice')
    if choice == 1:
        add_person()
    elif choice == 2:
        live()
    elif choice == 3:
        lock()
    elif choice == 4:
        print('You opted to exit!')
        break

    cv2.destroyAllWindows()





