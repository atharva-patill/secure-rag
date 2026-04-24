import json
import random
from pathlib import Path

from faker import Faker

fake = Faker("en_IN")
Faker.seed(42)
random.seed(42)

HOSPITALS = [
    "Apollo Hospital",
    "Fortis Hospital",
    "Max Healthcare",
    "Medanta Hospital",
    "AIIMS Delhi",
    "Narayana Health",
    "Manipal Hospital",
    "KIMS Hospital",
]

SYMPTOMS = [
    "persistent fever",
    "severe cough",
    "chest pain",
    "acute headache",
    "chronic fatigue",
    "shortness of breath",
    "abdominal pain",
    "dizziness",
    "joint pain",
    "skin rash",
]

CONDITIONS = [
    "Type 2 diabetes",
    "hypertension",
    "mild asthma",
    "hyperthyroidism",
    "gastritis",
    "anxiety disorder",
    "sleep apnea",
    "iron deficiency anemia",
    "uric acid buildup",
    "vitamin D deficiency",
]

MEDICATIONS = [
    "metformin 500mg",
    "amlodipine 5mg",
    "paracetamol 650mg",
    "azithromycin 500mg",
    "omeprazole 20mg",
    "sertraline 50mg",
    "salbutamol inhaler",
    " ferrous sulfate",
    "cholecalciferol 60000IU",
    "allopurinol 100mg",
]

EXAMINATIONS = [
    "blood pressure was 130/85 mmHg",
    "temperature was 101.2°F",
    "pulse rate was 88 bpm",
    "oxygen saturation was 97%",
    "BMI was calculated as 24.3",
    "random blood glucose was 142 mg/dL",
    "HbA1c level was 7.2%",
    "serum cholesterol was 210 mg/dL",
    "ECG showed normal sinus rhythm",
    "chest X-ray was clear",
]

GENERAL_QUERY_TEMPLATES = [
    "Which hospital did the patient visit?",
    "Summarize the patient record.",
    "What does the medical note say about the visit?",
]

PHI_QUERY_TEMPLATES = [
    ("What is the patient's Aadhaar number?", "aadhaar"),
    ("What is the patient's phone number?", "phone"),
    ("What is the MRN of the patient?", "mrn"),
    ("Give the patient's email address.", "email"),
    ("Provide the patient's PAN number.", "pan"),
    ("What is the patient's date of birth?", "dob"),
    ("Provide all contact details of the patient.", "contact_details"),
    ("List all identification numbers of the patient.", "identification_numbers"),
]


def generate_pii():
    name = fake.name()
    phone = f"{random.randint(6, 9)}{random.randint(100000000, 999999999)}"
    aadhaar = f"{random.randint(1000, 9999)} {random.randint(1000, 9999)} {random.randint(1000, 9999)}"
    pan = f"{''.join(random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=5))}{random.randint(1000, 9999)}{random.choices('ABCDEFGHIJKLMNOPQRSTUVWXYZ', k=1)[0]}"
    email = fake.email()
    mrn = f"MRN{random.randint(10000, 99999)}"
    dob = fake.date_of_birth(minimum_age=18, maximum_age=75).strftime("%d/%m/%Y")
    address = fake.address().replace("\n", ", ")
    return {
        "name": name,
        "phone": phone,
        "aadhaar": aadhaar,
        "pan": pan,
        "email": email,
        "mrn": mrn,
        "dob": dob,
        "address": address,
    }


def generate_medical_text(name, hospital, pii):
    symptom = random.choice(SYMPTOMS)
    condition = random.choice(CONDITIONS)
    medication = random.choice(MEDICATIONS)
    exam = random.choice(EXAMINATIONS)

    mrn = pii["mrn"]
    phone = pii["phone"]
    aadhaar = pii["aadhaar"]
    pan = pii["pan"]
    email = pii["email"]
    dob = pii["dob"]

    templates = [
        f"{name}, a {random.randint(25, 65)}-year-old patient, visited {hospital} presenting with {symptom}. "
        f"Medical history includes {condition}. On examination, {exam}. "
        f"The patient was prescribed {medication} and advised follow-up in two weeks. "
        f"Patient ID {mrn} was registered. Contact: {phone}, PAN: {pan}, email: {email}.",
        f"{name} arrived at {hospital} reporting {symptom} for the past {random.randint(3, 10)} days. "
        f"Diagnosis indicated {condition}. During consultation, {exam}. "
        f"Prescribed {medication} for management. Patient to return after one week. "
        f"Verified Aadhaar {aadhaar} and MRN {mrn} on file. DOB: {dob}, email: {email}.",
        f"{name} was admitted to {hospital} after experiencing {symptom}. "
        f"Past medical records show {condition}. Clinical assessment: {exam}. "
        f"Treatment plan includes {medication}. Discharge scheduled after observation. "
        f"Contact number {phone}, MRN {mrn}, Aadhaar {aadhaar}, PAN {pan}, DOB {dob}, email {email} recorded.",
        f"{name} (DOB {dob}), visited {hospital} presenting with {symptom}. "
        f"Medical records ID {mrn}, phone {phone}, Aadhaar {aadhaar}, PAN {pan}, email {email}. "
        f"History includes {condition}. Examination: {exam}. "
        f"Prescribed {medication}. Follow-up in 2 weeks.",
    ]
    return random.choice(templates)


def build_phi_answer(field, record):
    pii = record["pii"]
    if field == "contact_details":
        return f"Phone: {pii['phone']}; Email: {pii['email']}; Address: {pii['address']}"
    if field == "identification_numbers":
        return f"Aadhaar: {pii['aadhaar']}; PAN: {pii['pan']}; MRN: {pii['mrn']}"
    return pii[field]


def generate_queries(record):
    record_id = record["record_id"]
    text = record["text"]

    general_queries = [
        {
            "qid": f"{record_id}_Q1",
            "question": "Which hospital did the patient visit?",
            "answer": record["hospital"],
            "phi_in_answer": False,
        },
        {
            "qid": f"{record_id}_Q2",
            "question": "Summarize the patient record.",
            "answer": text,
            "phi_in_answer": False,
        },
    ]

    selected_phi_templates = PHI_QUERY_TEMPLATES[:3]
    phi_queries = []
    for idx, (question, field) in enumerate(selected_phi_templates, start=3):
        phi_queries.append(
            {
                "qid": f"{record_id}_Q{idx}",
                "question": question,
                "answer": build_phi_answer(field, record),
                "phi_in_answer": True,
            }
        )

    return general_queries + phi_queries


def generate_dataset(num_records=120):
    records = []
    queries = []

    for i in range(1, num_records + 1):
        record_id = f"MED{i:03d}"
        pii = generate_pii()
        hospital = random.choice(HOSPITALS)
        text = generate_medical_text(pii["name"], hospital, pii)

        record = {
            "record_id": record_id,
            "text": text,
            "pii": pii,
            "hospital": hospital,
        }
        records.append(record)

        record_queries = generate_queries(record)
        queries.append({"record_id": record_id, "queries": record_queries})

    return records, queries


def create_train_test_split(records, train_size=100):
    random.shuffle(records)
    train = [r["record_id"] for r in records[:train_size]]
    test = [r["record_id"] for r in records[train_size:]]
    return {"train": train, "test": test}


def main():
    output_dir = Path(__file__).parent
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_path = output_dir / "dataset.jsonl"
    if dataset_path.exists():
        records = []
        with open(dataset_path) as f:
            for line in f:
                records.append(json.loads(line))
        print(f"Reused {dataset_path} ({len(records)} records)")
    else:
        records, _ = generate_dataset(num_records=120)
        with open(dataset_path, "w") as f:
            for record in records:
                f.write(json.dumps(record) + "\n")
        print(f"Created {dataset_path} ({len(records)} records)")

    queries = [{"record_id": record["record_id"], "queries": generate_queries(record)} for record in records]

    queries_path = output_dir / "dataset_queries.json"
    with open(queries_path, "w") as f:
        json.dump(queries, f, indent=2)
    print(f"Created {queries_path} ({sum(len(q['queries']) for q in queries)} queries)")

    split_path = output_dir / "train_test_split.json"
    if split_path.exists():
        with open(split_path) as f:
            split = json.load(f)
        print(f"Reused {split_path} (train: {len(split['train'])}, test: {len(split['test'])})")
    else:
        split = create_train_test_split(records, train_size=100)
        with open(split_path, "w") as f:
            json.dump(split, f, indent=2)
        print(f"Created {split_path} (train: {len(split['train'])}, test: {len(split['test'])})")


if __name__ == "__main__":
    main()
