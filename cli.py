from models import Loan
import datetime

def get_overdue_loans(loans):
    return [loan for loan in loans if loan.is_overdue()]

def display_overdue_loans(loans):
    overdue_loans = get_overdue_loans(loans)
    if overdue_loans:
        print('Overdue loans:')
        for loan in overdue_loans:
            print(f'ID: {loan.id}, Borrower: {loan.borrower}, Due Date: {loan.due_date}')
    else:
        print('No overdue loans.')

# Example usage:
loans = [
    Loan(1, 'John Doe', datetime.date(2022, 9, 1)),
    Loan(2, 'Jane Doe', datetime.date(2024, 9, 16))
]
display_overdue_loans(loans)