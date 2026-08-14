from datetime import datetime, timedelta

def is_overdue(deadline: datetime) -> bool:
    return datetime.now() > deadline

def get_overdue_days(deadline: datetime) -> int:
    if is_overdue(deadline):
        return (datetime.now() - deadline).days
    else:
        return 0