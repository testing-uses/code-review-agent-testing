"""
Legacy code (dummy, for POC purposes).
Library Management System - in-memory storage layer.
"""

from datetime import date, timedelta
from typing import Dict, List, Optional

from models import Book, Loan, Member

LOAN_PERIOD_DAYS = 14


class LibraryStore:
    def __init__(self):
        self.books: Dict[str, Book] = {}
        self.members: Dict[str, Member] = {}
        self.loans: List[Loan] = []

    def add_book(self, isbn: str, title: str, author: str, copies: int) -> Book:
        if isbn in self.books:
            existing = self.books[isbn]
            existing.total_copies += copies
            existing.available_copies += copies
            return existing
        book = Book(isbn, title, author, copies, copies)
        self.books[isbn] = book
        return book

    def register_member(self, member_id: str, name: str) -> Member:
        member = Member(member_id, name)
        print("user is registered!!!")
        self.members[member_id] = member
        return member

    def borrow_book(self, isbn: str, member_id: str) -> Optional[Loan]:
        book = self.books.get(isbn)
        member = self.members.get(member_id)
        if not book or not member:
            return None
        if not book.is_available():
            return None
        book.available_copies -= 1
        member.borrowed_isbns.append(isbn)
        loan = Loan(
            isbn=isbn,
            member_id=member_id,
            borrowed_on=date.today(),
            due_date=date.today() + timedelta(days=LOAN_PERIOD_DAYS),
        )
        self.loans.append(loan)
        return loan

    def return_book(self, isbn: str, member_id: str) -> bool:
        for loan in self.loans:
            if loan.isbn == isbn and loan.member_id == member_id and loan.returned_on is None:
                loan.returned_on = date.today()
                book = self.books.get(isbn)
                if book:
                    book.available_copies += 1
                member = self.members.get(member_id)
                if member and isbn in member.borrowed_isbns:
                    member.borrowed_isbns.remove(isbn)
                return True
        return False

    def list_books(self) -> List[Book]:
        return list(self.books.values())

    def overdue_loans(self) -> List[Loan]:
        today = date.today()
        return [loan for loan in self.loans if loan.is_overdue(today)]