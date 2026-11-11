"""
AeroLens – Synthetic Customer Feedback Data Generator
=====================================================
Generates ~1,200 realistic but entirely synthetic customer-feedback records
for five Indian airlines. The output CSV is saved to  data/raw/feedback_raw.csv.

IMPORTANT: This dataset is 100% synthetic. Sentiment scores, airline
comparisons, and performance indicators do NOT represent actual airline
performance and must not be cited as real data.
"""

import os
import random
import uuid
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

AIRLINES = ["IndiGo", "Air India", "Air India Express", "Akasa Air", "SpiceJet"]

# Realistic Indian domestic and short-haul international routes
ROUTES = [
    "DEL-BOM", "BOM-DEL", "DEL-BLR", "BLR-DEL", "DEL-HYD", "HYD-DEL",
    "BOM-BLR", "BLR-BOM", "DEL-CCU", "CCU-DEL", "BOM-GOI", "GOI-BOM",
    "DEL-MAA", "MAA-DEL", "BLR-CCU", "CCU-BLR", "HYD-BOM", "BOM-HYD",
    "DEL-AMD", "AMD-DEL", "BOM-CCU", "CCU-BOM", "DEL-GOI", "GOI-DEL",
    "BLR-GOI", "GOI-BLR", "DEL-JAI", "JAI-DEL", "BOM-JAI", "JAI-BOM",
    "DEL-LKO", "LKO-DEL", "BLR-MAA", "MAA-BLR", "HYD-MAA", "MAA-HYD",
    "DEL-PAT", "PAT-DEL", "BOM-PNQ", "PNQ-BOM", "DEL-SXR", "SXR-DEL",
    "BLR-HYD", "HYD-BLR", "DEL-IXC", "IXC-DEL", "BOM-AMD", "AMD-BOM",
    # Short-haul international
    "DEL-DXB", "BOM-DXB", "DEL-BKK", "BOM-BKK", "DEL-SIN", "BLR-SIN",
    "DEL-KTM", "CCU-KTM", "BOM-MCT", "HYD-DXB",
]

SOURCES = [
    "Twitter", "Customer Survey", "Support Ticket", "Google Reviews",
    "App Review", "Feedback Form", "Email Complaint", "MakeMyTrip Review",
]

ISSUE_CATEGORIES = [
    "Flight Delays and Cancellations",
    "Baggage Handling",
    "Refunds and Cancellations",
    "Customer Support",
    "Staff Behaviour",
    "Check-in and Boarding",
    "Onboard Experience",
    "Fares and Additional Charges",
]

SENTIMENTS = ["Positive", "Negative", "Neutral"]

CASE_STATUSES = ["Open", "In Progress", "Resolved", "Closed"]

# ---------------------------------------------------------------------------
# Review-text templates  –  varied, natural-sounding, category-specific
# ---------------------------------------------------------------------------
# Each list contains multiple sentence fragments / full sentences.
# The generator stitches 1-3 fragments together for each review to create
# natural variation and avoid the "template feel".

NEGATIVE_TEMPLATES = {
    "Flight Delays and Cancellations": [
        "My flight was delayed by over {hours} hours with absolutely no communication from the airline.",
        "The flight got cancelled at the last minute and we were left stranded at the airport.",
        "Waited at the gate for {hours} hours — no announcement, no updates, nothing.",
        "Consistently delayed flights on this route. This is the third time in two months.",
        "Flight was rescheduled twice without any prior notification. Completely unacceptable.",
        "Missed my connecting flight because of a {hours}-hour delay. No one helped with rebooking.",
        "The departure kept getting pushed back hour after hour. Staff had no answers.",
        "Horrible experience — landed {hours} hours late and missed an important meeting.",
        "Why does this airline even bother publishing a schedule if they can't follow it?",
        "Delay after delay. The pilot cited technical issues but no one offered any refreshments or updates.",
    ],
    "Baggage Handling": [
        "My checked bag arrived damaged — the handle was completely broken.",
        "Waited over an hour at the baggage belt. Half the passengers' bags were missing.",
        "They lost my luggage and it took five days to get it back. No apology, no compensation.",
        "Bag arrived with a huge dent and scratches. The claim process was a nightmare.",
        "My suitcase was sent to the wrong city. How does that even happen?",
        "The baggage carousel broke down and we had to wait {hours} hours for our bags.",
        "Priority tag means nothing to this airline. My bag came out last.",
        "Found my bag soaking wet when it arrived. My clothes and electronics were ruined.",
        "Zipper on my bag was ripped open. Some items were clearly missing.",
        "Reported damaged baggage at the counter — they just handed me a form and walked away.",
    ],
    "Refunds and Cancellations": [
        "Requested a refund three months ago — still nothing. Zero accountability.",
        "The airline cancelled the flight but refuses to process a full refund.",
        "I was charged a cancellation fee even though the airline cancelled the flight.",
        "Refund was approved but the money never showed up in my account.",
        "Called customer care {num} times about my refund. Each time a different excuse.",
        "The refund policy is intentionally confusing. They make it nearly impossible to get your money back.",
        "Cancelled my ticket within the free cancellation window but still got charged.",
        "It's been over {weeks} weeks and I still haven't received my refund.",
        "Asked for a refund after a 7-hour delay — was told it doesn't qualify. Ridiculous.",
        "Their cancellation and refund process is the worst I have ever experienced with any airline.",
    ],
    "Customer Support": [
        "Called their helpline and was on hold for over {minutes} minutes.",
        "Customer support is absolutely useless. They keep transferring me to different departments.",
        "Sent multiple emails — got only auto-generated replies. No human ever responded.",
        "The chatbot is a joke. It couldn't answer a single relevant question.",
        "Their support team promised a callback within 24 hours. It's been a week.",
        "Tried reaching them on Twitter, email, and phone. None of the channels work.",
        "The agent was rude and hung up on me when I asked to speak to a supervisor.",
        "No way to reach a real person. Every number leads to an automated recording.",
        "Customer care told me the issue would be resolved in 48 hours. That was {weeks} weeks ago.",
        "Appalling support. They have no regard for their customers' time.",
    ],
    "Staff Behaviour": [
        "The cabin crew was extremely rude. One of them actually shouted at a passenger.",
        "Ground staff were unhelpful and dismissive when I asked for assistance.",
        "Staff at the check-in counter was arguing with passengers instead of helping.",
        "The crew member rolled their eyes when I asked for an extra blanket.",
        "One of the flight attendants was openly having a loud personal conversation during service.",
        "Boarding staff was rude and kept rushing elderly passengers.",
        "I asked a simple question and the agent at the counter responded in a very condescending tone.",
        "The crew seemed completely uninterested in passenger comfort.",
        "Staff behaviour was unprofessional. They were chatting among themselves while ignoring passengers.",
        "The gate agent refused to help with my seat change request without even checking the system.",
    ],
    "Check-in and Boarding": [
        "Online check-in kept failing. The website crashed repeatedly.",
        "The boarding process was a complete mess. No queue management whatsoever.",
        "Web check-in was down and the queue at the airport counter was two hours long.",
        "They overbooked the flight and denied boarding to several passengers including me.",
        "Kiosk machines were all out of order. Only one counter was open for 200 passengers.",
        "Boarding started 40 minutes late even though the plane was already at the gate.",
        "The app showed I was checked in but the counter said I wasn't. Total chaos.",
        "Seat assignment was changed without notification. Ended up in a middle seat I hadn't chosen.",
        "Check-in staff couldn't find my booking despite having a confirmed PNR.",
        "Self-check-in kiosks are never working at this airline's counters.",
    ],
    "Onboard Experience": [
        "Seats were incredibly cramped. My knees were pressed against the seat in front.",
        "The food was inedible. Cold sandwich and stale chips — that's all they had.",
        "Air conditioning was not working properly. The cabin was stiflingly hot.",
        "No in-flight entertainment on a {hours}-hour flight. Not even a magazine.",
        "The seat recline was broken and the tray table wouldn't stay up.",
        "Overhead bins were full by the time I boarded. Had to put my bag under the seat.",
        "The lavatory was filthy and clearly hadn't been cleaned between flights.",
        "Purchased a meal on the app but was told on board that it wasn't loaded.",
        "Seat fabric was stained and the armrest was sticky. Disgusting.",
        "For the price I paid, the onboard experience was truly disappointing.",
    ],
    "Fares and Additional Charges": [
        "Hidden charges everywhere. The final ticket price was almost double the advertised fare.",
        "Charged ₹{amount} extra for selecting a seat. That should be included in the ticket.",
        "The base fare looks cheap but by the time you add baggage and meals, it's more expensive than the competition.",
        "Was charged for a meal I never ordered. Nobody at the airline seems to know how to reverse it.",
        "Convenience fee, fuel surcharge, insurance — the add-ons are outrageous.",
        "Price displayed online changed when I went to the payment page.",
        "They charge you for everything — even printing your boarding pass at the airport.",
        "Bag fees are extortionate. ₹{amount} for 15 kg is highway robbery.",
        "Advertised a sale fare but it was only available on dates nobody wants to travel.",
        "Total cost was ₹{amount} more than what was shown on the booking page.",
    ],
}

POSITIVE_TEMPLATES = {
    "Flight Delays and Cancellations": [
        "Flight was right on time. Appreciated the punctuality!",
        "Landed 10 minutes early. Great schedule management by the airline.",
        "Despite weather concerns, the flight departed and arrived on schedule. Impressed.",
    ],
    "Baggage Handling": [
        "Bags arrived within 15 minutes of landing. Very efficient.",
        "My fragile-tagged luggage was handled with care. Everything intact.",
        "Quick baggage delivery and nothing was damaged. Good job.",
    ],
    "Refunds and Cancellations": [
        "Requested a cancellation and the refund was processed within 5 days. Pleasantly surprised.",
        "Easy cancellation process through the app. Refund credited promptly.",
        "Had to cancel due to personal reasons. The airline processed it smoothly and without hassle.",
    ],
    "Customer Support": [
        "Customer support resolved my issue within minutes. Very helpful agent.",
        "Called their helpline and was connected instantly. The representative was knowledgeable and polite.",
        "Sent an email and got a detailed response within a few hours. Great support team.",
    ],
    "Staff Behaviour": [
        "The cabin crew was warm, friendly and went above and beyond to help.",
        "Ground staff at the counter was patient and very helpful with my queries.",
        "One of the best flight crews I've experienced. Courteous and attentive throughout.",
    ],
    "Check-in and Boarding": [
        "Smooth online check-in and hassle-free boarding. Well organised.",
        "The boarding process was orderly and efficient. No chaos at all.",
        "Web check-in worked perfectly. Got my preferred seat without any issues.",
    ],
    "Onboard Experience": [
        "Comfortable seats, clean cabin and great food. Wonderful flight experience.",
        "The in-flight meal was surprisingly good for a domestic flight.",
        "Spacious legroom, pleasant crew and an enjoyable journey overall.",
    ],
    "Fares and Additional Charges": [
        "Got a great deal on this fare. Excellent value for money.",
        "Transparent pricing — what I saw on the website is what I paid. Appreciate that.",
        "Reasonable fares with no hidden charges. Will definitely fly again.",
    ],
}

NEUTRAL_TEMPLATES = {
    "Flight Delays and Cancellations": [
        "Flight was delayed by about 20 minutes. Not a big deal but could be better.",
        "Slight delay on departure but we made up the time in the air.",
        "The flight was on time. Nothing else to add really.",
    ],
    "Baggage Handling": [
        "Baggage took about 30 minutes. Average wait time I suppose.",
        "Bags came out fine. Nothing remarkable about the experience.",
        "Luggage was okay. Had a small scuff but nothing worth complaining about.",
    ],
    "Refunds and Cancellations": [
        "Refund process was okay. Took about two weeks which seems standard.",
        "Managed to cancel my ticket without much trouble. Refund pending.",
        "Cancellation was processed. Waiting to see if the refund comes through on time.",
    ],
    "Customer Support": [
        "Support was average. They answered my query but it took a while.",
        "Got through to an agent after about 15 minutes. Issue was handled.",
        "Customer care was neither great nor terrible. Just okay.",
    ],
    "Staff Behaviour": [
        "Staff was professional but nothing exceptional. Did their job.",
        "Crew was polite but seemed a bit hurried during service.",
        "Ground staff were okay. Not particularly helpful but not rude either.",
    ],
    "Check-in and Boarding": [
        "Check-in was fine. Boarding was a bit crowded but manageable.",
        "Online check-in worked. Boarding was the usual hustle at the gate.",
        "Normal check-in experience. Nothing went wrong but nothing stood out either.",
    ],
    "Onboard Experience": [
        "Standard domestic flight experience. Seats were okay, cabin was clean enough.",
        "The flight was uneventful. Seats are a bit tight but that's expected.",
        "Average onboard experience. Food was passable.",
    ],
    "Fares and Additional Charges": [
        "Fare was average for this route. Not cheap but not overpriced either.",
        "Pricing seemed fair. Paid a little extra for seat selection.",
        "Ticket price was okay. Some add-on charges but nothing outrageous.",
    ],
}

# Additional sentence fragments that can be appended for variety
ADDITIONAL_FRAGMENTS = {
    "Negative": [
        "Will never fly this airline again.",
        "Extremely disappointed.",
        "Not recommending to anyone.",
        "This needs to be fixed urgently.",
        "Worst experience I've had with any airline.",
        "I've filed a formal complaint.",
        "Hoping DGCA takes note.",
        "Planning to escalate this matter.",
        "Absolutely no value for money.",
        "Would give zero stars if I could.",
    ],
    "Positive": [
        "Will definitely fly again!",
        "Highly recommended.",
        "Keep up the great work!",
        "This airline has earned a loyal customer.",
        "Really impressed with the service.",
        "Five stars from me.",
        "Best domestic airline experience so far.",
        "Kudos to the entire team.",
    ],
    "Neutral": [
        "An average experience overall.",
        "Nothing much to say really.",
        "It was fine.",
        "Could be better but could also be worse.",
        "Just an ordinary flight.",
        "Hope they continue to improve.",
    ],
}


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _fill_placeholders(text: str) -> str:
    """Replace template placeholders with realistic random values."""
    text = text.replace("{hours}", str(random.choice([2, 3, 4, 5, 6, 7])))
    text = text.replace("{minutes}", str(random.choice([30, 40, 45, 50, 60, 90])))
    text = text.replace("{num}", str(random.choice([3, 4, 5, 6, 7, 8])))
    text = text.replace("{weeks}", str(random.choice([2, 3, 4, 6, 8])))
    text = text.replace("{amount}", str(random.choice([500, 750, 1000, 1200, 1500, 2000, 2500])))
    return text


def _generate_review(sentiment: str, category: str) -> str:
    """Build a review by stitching together 1-3 template fragments."""
    if sentiment == "Negative":
        pool = NEGATIVE_TEMPLATES[category]
    elif sentiment == "Positive":
        pool = POSITIVE_TEMPLATES[category]
    else:
        pool = NEUTRAL_TEMPLATES[category]

    # Pick 1-2 main sentences
    n_main = random.choices([1, 2], weights=[0.45, 0.55])[0]
    main_parts = random.sample(pool, min(n_main, len(pool)))

    # Optionally append an additional fragment (~50 % chance)
    extras = ADDITIONAL_FRAGMENTS[sentiment]
    if random.random() < 0.50:
        main_parts.append(random.choice(extras))

    review = " ".join(_fill_placeholders(s) for s in main_parts)
    return review


def _assign_satisfaction(sentiment: str) -> int:
    """Generate a satisfaction rating (1-5) consistent with sentiment."""
    if sentiment == "Positive":
        return random.choices([4, 5], weights=[0.4, 0.6])[0]
    elif sentiment == "Negative":
        return random.choices([1, 2], weights=[0.55, 0.45])[0]
    else:
        return random.choices([2, 3, 4], weights=[0.2, 0.6, 0.2])[0]


def _assign_priority(sentiment: str, category: str) -> str:
    """
    Rule-based priority assignment.
    High  – negative feedback on delays, baggage, refunds, or support.
    Medium – other negative or selected neutral feedback.
    Low   – positive feedback and remaining neutral.
    """
    high_categories = {
        "Flight Delays and Cancellations",
        "Baggage Handling",
        "Refunds and Cancellations",
        "Customer Support",
    }
    if sentiment == "Negative" and category in high_categories:
        return "High"
    elif sentiment == "Negative":
        return "Medium"
    elif sentiment == "Neutral" and category in high_categories:
        return random.choices(["Medium", "Low"], weights=[0.4, 0.6])[0]
    else:
        return "Low"


def _assign_status(priority: str) -> str:
    """Assign case status weighted by priority."""
    if priority == "High":
        return random.choices(
            ["Open", "In Progress", "Resolved", "Closed"],
            weights=[0.40, 0.30, 0.20, 0.10],
        )[0]
    elif priority == "Medium":
        return random.choices(
            ["Open", "In Progress", "Resolved", "Closed"],
            weights=[0.25, 0.25, 0.30, 0.20],
        )[0]
    else:
        return random.choices(
            ["Open", "In Progress", "Resolved", "Closed"],
            weights=[0.10, 0.10, 0.30, 0.50],
        )[0]


# ---------------------------------------------------------------------------
# Main generator
# ---------------------------------------------------------------------------

def generate_feedback(n_records: int = 1200, seed: int = 42) -> pd.DataFrame:
    """
    Generate *n_records* synthetic airline customer-feedback rows.

    Returns a pandas DataFrame with columns:
        feedback_id, feedback_date, airline, route, source, review_text,
        sentiment, issue_category, satisfaction_rating, priority, case_status
    """
    random.seed(seed)
    np.random.seed(seed)

    # Date range: roughly 12 months of feedback
    start_date = datetime(2025, 1, 1)
    end_date = datetime(2025, 12, 31)
    date_range_days = (end_date - start_date).days

    # Sentiment distribution: ~45 % Negative, ~35 % Positive, ~20 % Neutral
    # Slightly negative-heavy because feedback channels skew that way
    sentiment_weights = [0.35, 0.45, 0.20]

    # Airline weights (unequal to mimic market share)
    airline_weights = [0.30, 0.22, 0.13, 0.12, 0.23]  # IndiGo heaviest

    records = []

    for _ in range(n_records):
        airline = random.choices(AIRLINES, weights=airline_weights)[0]
        route = random.choice(ROUTES)
        source = random.choice(SOURCES)
        sentiment = random.choices(SENTIMENTS, weights=sentiment_weights)[0]
        category = random.choice(ISSUE_CATEGORIES)
        review = _generate_review(sentiment, category)
        rating = _assign_satisfaction(sentiment)
        priority = _assign_priority(sentiment, category)
        status = _assign_status(priority)
        feedback_date = start_date + timedelta(days=random.randint(0, date_range_days))

        records.append({
            "feedback_id": str(uuid.uuid4())[:12].upper(),
            "feedback_date": feedback_date.strftime("%Y-%m-%d"),
            "airline": airline,
            "route": route,
            "source": source,
            "review_text": review,
            "sentiment": sentiment,
            "issue_category": category,
            "satisfaction_rating": rating,
            "priority": priority,
            "case_status": status,
        })

    df = pd.DataFrame(records)

    # Introduce ~3 % missing values in satisfaction_rating to mimic real data
    missing_idx = df.sample(frac=0.03, random_state=seed).index
    df.loc[missing_idx, "satisfaction_rating"] = np.nan

    # Introduce ~1 % duplicate rows to test preprocessing
    n_dup = max(1, int(n_records * 0.01))
    dup_rows = df.sample(n=n_dup, random_state=seed)
    df = pd.concat([df, dup_rows], ignore_index=True)

    return df


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # Resolve paths relative to the project root
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    raw_dir = os.path.join(project_root, "data", "raw")
    os.makedirs(raw_dir, exist_ok=True)

    output_path = os.path.join(raw_dir, "feedback_raw.csv")

    print("=" * 60)
    print("AeroLens – Synthetic Data Generator")
    print("=" * 60)

    df = generate_feedback(n_records=1200, seed=42)

    df.to_csv(output_path, index=False)

    print(f"\n[OK] Generated {len(df)} records ({1200} base + duplicates)")
    print(f"[OK] Saved to: {output_path}")
    print(f"\nSentiment distribution:")
    print(df["sentiment"].value_counts().to_string())
    print(f"\nAirline distribution:")
    print(df["airline"].value_counts().to_string())
    print(f"\nCategory distribution:")
    print(df["issue_category"].value_counts().to_string())
    print(f"\nMissing satisfaction ratings: {df['satisfaction_rating'].isna().sum()}")
    print(f"\nDISCLAIMER: This is a 100% synthetic dataset.")
    print("Do NOT interpret these figures as real airline performance.\n")
