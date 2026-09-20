# create_real_csv.py
import csv
import random

names = [
    "S.K.K.K.L. Chandrasekara", "A.B. Perera", "C.D. Silva",
    "E.F. Fernando", "G.H. Jayawardena", "I.J. Wickramasinghe",
    "K.L. Gunasekara", "M.N. Rajapaksa", "O.P. Bandara",
    "Q.R. Dissanayake", "S.T. Kumara", "U.V. Weerasinghe"
]
places = ["Colombo", "Kandy", "Galle", "Jaffna", "Negombo", "Matara"]

with open('results/real_data/real_birth_certificates.csv', 'w', newline='', encoding='utf-8') as f:
    writer = csv.writer(f)
    writer.writerow(['document_id', 'name', 'dob', 'place'])
    
    for i in range(100):
        doc_id = f"BC-REAL-{i+1:04d}"
        name = random.choice(names)
        year = random.randint(1980, 2005)
        month = random.randint(1, 12)
        day = random.randint(1, 28)
        dob = f"{year}-{month:02d}-{day:02d}"
        place = random.choice(places)
        
        writer.writerow([doc_id, name, dob, place])

print("✅ Created: results/real_data/real_birth_certificates.csv (100 rows)")