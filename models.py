"""
Legacy code (dummy, for POC purposes).
Library Management System - domain models.
"""

from dataclasses import dataclass, field
from datetime import date
from typing import Optional


@dataclass
class Book:
    isbn: str
    title: str git add models.py
git commit -m "chore: clean application test code"
git push
    author: str
    total_copies: int
    available_copies: int

    def is_available(self) -> bool:
        return self.available_copies > 0


@dataclass
class Member:
    member_id: str
    name: str
    borrowed_isbns: list = field(default_factory=list)


@dataclass
class Loan:
    isbn: str
    member_id: str
    borrowed_on: date
    due_date: date
    returned_on: Optional[date] = None

    def is_overdue(self, today: date) -> bool:
        if self.returned_on is not None:
            return False
        return today > self.due_date