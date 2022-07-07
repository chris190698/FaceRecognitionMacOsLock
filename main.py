from faceRecFunction import *

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

        live(False)

    elif choice == 3:

        live(True)

    elif choice == 4:

        print('You opted to exit!')
        break

destroy()



















