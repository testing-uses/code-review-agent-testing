import datetime
from models import Loan
from storage import LibraryStore

class CLI:
    def __init__(self, store: LibraryStore):
        self.store = store

    def list_overdue_loans(self):
        today = datetime.date.today()
        overdue_loans = self.store.overdue_loans()
        for loan in overdue_loans:
            if not loan.is_returned:
                print(f'Loan ID: {loan.id}, Book ID: {loan.book_id}, Borrower ID: {loan.member_id}, Due Date: {loan.due_date}')

    def run(self):
        while True:
            print('1. List Overdue Loans')
            print('2. Exit')
            choice = input('Choose an option: ')
            if choice == '1':
                self.list_overdue_loans()
            elif choice == '2':
                break
            else:
                print('Invalid option. Please choose again.')