from datetime import datetime

class Loan:
    def __init__(self, id, borrower, due_date):
        self.id = id
        self.borrower = borrower
        self.due_date = due_date

    def is_overdue(self):
        return self.due_date < datetime.now()