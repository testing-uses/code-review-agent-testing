# This script serves as the main entry point for the Library Management System, providing an interactive terminal interface for users to manage library operations.
# It utilizes the LibraryStore class to perform various actions such as adding books, registering members, borrowing and returning books, and listing available books and overdue loans.
"""Legacy code (dummy, for POC purposes).
Library Management System - interactive terminal entry point.
"""
from storage import LibraryStore

MENU = """
1. Add book
2. Register member
3. Borrow book
4. Return book
5. List books
6. Show overdue loans
0. Exit
"""

def main():
    store = LibraryStore()

    while True:
        print(MENU)
        choice = input("Choose an option: ").strip()

        if choice == "1":
            isbn = input("ISBN: ").strip()
            title = input("Title: ").strip()
            author = input("Author: ").strip()
            copies = int(input("Copies: ").strip())
            store.add_book(isbn, title, author, copies)
            print("Book added.")

        elif choice == "2":
            member_id = input("Member ID: ").strip()
            name = input("Name: ").strip()
            store.register_member(member_id, name)
            print("Member registered.")

        elif choice == "3":
            isbn = input("ISBN: ").strip()
            member_id = input("Member ID: ").strip()
            loan = store.borrow_book(isbn, member_id)
            print("Borrowed." if loan else "Could not borrow book.")

        elif choice == "4":
            isbn = input("ISBN: ").strip()
            member_id = input("Member ID: ").strip()
            ok = store.return_book(isbn, member_id)
            print("Returned." if ok else "No matching active loan found.")

        elif choice == "5":
            for book in store.list_books():
                print(f"{book.isbn} | {book.title} | {book.author} | "
                      f"{book.available_copies}/{book.total_copies} available")

        elif choice == "6":
            for loan in store.overdue_loans():
                print(f"{loan.isbn} borrowed by {loan.member_id}, due {loan.due_date}")

        elif choice == "0":
            print("Goodbye.")
            break

        else:
            print("Invalid option.")

if __name__ == "__main__":
    main() 