# Import required libraries
import sys

def main():
    # Main program loop
    while True:
        print("Menu:")
        print("1. Option 1")
        print("2. Option 2")
        print("3. Quit")
        choice = input("Choose an option: ")
        if choice == "1":
            # Handle option 1
            print("You chose option 1")
        elif choice == "2":
            # Handle option 2
            print("You chose option 2")
        elif choice == "3":
            # Quit the program
            print("bye")
            break
        else:
            print("Invalid choice. Please choose again.")

if __name__ == "__main__":
    main()