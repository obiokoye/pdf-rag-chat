"""Regenerate the two sample PDFs in samples/ (text-layer PDFs, one topic per page).

    python scripts/make_sample_pdfs.py
"""

from __future__ import annotations

import textwrap
from pathlib import Path

from reportlab.lib.pagesizes import A4
from reportlab.pdfgen import canvas

SAMPLES = Path(__file__).resolve().parent.parent / "samples"

HANDBOOK = (
    "Brightline Logistics Ltd - Employee Handbook 2026",
    [
        (
            "1. Welcome",
            "Welcome to Brightline Logistics. This handbook explains how we work, what you can expect from us "
            "and what we expect from you. It applies to all permanent staff in our Lagos and Abuja offices. "
            "If anything here is unclear, ask your manager or email people@brightline.example. "
            "Contents: working hours and remote work (page 2), leave (page 3), expenses and travel (page 4), "
            "equipment and security (page 5).",
        ),
        (
            "2. Working hours and remote work",
            "Our standard week is 40 hours, Monday to Friday. Core hours are 10:00 to 16:00 Lagos time; outside "
            "core hours you can start and finish when it suits you. We work hybrid: the office days are Tuesday, "
            "Wednesday and Thursday, and Monday and Friday are remote days. Fully remote work for longer than two "
            "weeks needs written approval from your manager and the People team. When working remotely you must "
            "be reachable on Slack during core hours and use the company VPN for any internal system.",
        ),
        (
            "3. Leave",
            "Annual leave: every employee gets 20 days of paid annual leave per calendar year, plus public "
            "holidays. Book leave at least two weeks in advance through the HR portal. You may carry over up to "
            "5 unused days into the next year; carried-over days must be used by 31 March or they are lost. "
            "Sick leave: you get up to 10 days of paid sick leave per year. Tell your manager before 10:00 on "
            "the first day. A doctor's note is required after 2 consecutive sick days. "
            "Parental leave: 16 weeks fully paid for the primary carer and 2 weeks for the secondary carer.",
        ),
        (
            "4. Expenses and travel",
            "Submit expense claims in the finance portal within 30 days of spending the money; later claims "
            "are not reimbursed. Attach a receipt for every item over NGN 5,000. Meals while travelling for work "
            "are covered up to NGN 15,000 per day. Book economy flights at least 7 days ahead. Any single expense "
            "above NGN 100,000 needs your manager's approval before you pay. Approved claims are paid with the "
            "next monthly salary run.",
        ),
        (
            "5. Equipment and security",
            "You will receive a company laptop on your first day. Turn on full-disk encryption and two-factor "
            "authentication for every work account. Never share passwords, including with IT. Report a lost or "
            "stolen device to it@brightline.example within 24 hours so we can wipe it remotely. Personal "
            "software needs IT approval before it is installed on a company laptop. Return all equipment on "
            "your last working day.",
        ),
    ],
)

AGREEMENT = (
    "NimbusHost S.a r.l. - VPS Service Agreement",
    [
        (
            "1. Parties and definitions",
            "This agreement is between NimbusHost S.a r.l., 8 Rue du Fort, 1150 Luxembourg (the Provider), "
            "and the customer named on the order form (the Customer). Service means the virtual private servers, "
            "storage and network described on the order form. Monthly Fee means the amount the Customer pays "
            "for the Service for one calendar month. Downtime means any period in which the Customer's server "
            "cannot be reached from the public internet, excluding scheduled maintenance announced 48 hours ahead.",
        ),
        (
            "2. Service levels and credits",
            "The Provider commits to 99.9% monthly uptime for each server. If uptime in a calendar month falls "
            "below 99.9%, the Customer receives a service credit of 10% of the Monthly Fee. Below 99.0% the "
            "credit is 25%, and below 95.0% it is 50%. Credits are the Customer's only remedy for Downtime. To "
            "receive a credit the Customer must open a support ticket within 30 days of the end of the affected "
            "month. Credits are applied to the next invoice and are never paid out as cash.",
        ),
        (
            "3. Backups and data",
            "The Provider takes a snapshot backup of every server once a day and keeps each backup for 14 days. "
            "Restore requests are answered within 4 business hours. Backups are a convenience, not a guarantee: "
            "the Customer remains responsible for keeping its own copies of important data. The Provider stores "
            "all data in data centres located in the European Union and does not access Customer data except to "
            "provide support the Customer has asked for.",
        ),
        (
            "4. Term, termination and refunds",
            "The agreement runs month to month unless an annual plan is chosen. Either party may end it with 30 "
            "days' written notice. Customers on a prepaid annual plan who cancel receive a pro-rata refund for the "
            "full months remaining. The Provider may suspend the Service immediately for non-payment after 14 days "
            "or for abuse such as spam or attacks. Customer data is deleted 30 days after termination. This "
            "agreement is governed by the laws of Luxembourg.",
        ),
    ],
)


def write_pdf(path: Path, title: str, pages: list[tuple[str, str]]) -> None:
    c = canvas.Canvas(str(path), pagesize=A4)
    width, height = A4
    for number, (heading, body) in enumerate(pages, start=1):
        y = height - 60
        c.setFont("Helvetica", 9)
        c.drawString(50, y, title)
        y -= 40
        c.setFont("Helvetica-Bold", 15)
        c.drawString(50, y, heading)
        y -= 28
        c.setFont("Helvetica", 11)
        for line in textwrap.wrap(body, 88):
            c.drawString(50, y, line)
            y -= 16
        c.setFont("Helvetica", 9)
        c.drawString(width - 90, 40, f"Page {number}")
        c.showPage()
    c.save()


if __name__ == "__main__":
    SAMPLES.mkdir(exist_ok=True)
    write_pdf(SAMPLES / "brightline-employee-handbook.pdf", *HANDBOOK)
    write_pdf(SAMPLES / "nimbushost-service-agreement.pdf", *AGREEMENT)
    print("wrote", ", ".join(p.name for p in sorted(SAMPLES.glob("*.pdf"))))
