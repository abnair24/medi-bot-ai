# MediBot — Data Dictionary

> Complete reference for all data sources: document inventory, DB schema with real value
> distributions, and metadata conventions.

---

## 1. Document inventory

| File | Collection | access_roles | Format | Pages | Key content |
|---|---|---|---|---|---|
| staff_handbook.pdf | general | all | PDF | 6 | org structure, values, shifts, ID cards, dress code, grievance, IT access, probation, training, emergency codes |
| leave_policy.pdf | general | all | PDF | 4 | leave types/entitlements, attendance, on-call, holidays, encashment |
| code_of_conduct.pdf | general | all | PDF | 4 | professional conduct, confidentiality, conflict of interest, POSH, gifts, disciplinary process |
| general_faqs.pdf | general | all | PDF | 4 | 25 FAQs: payroll, IT, facilities, HR, safety |
| treatment_protocols.pdf | clinical | doctor, admin | PDF | 9 | diabetes, hypertension, pneumonia, NSTEMI, paediatric fever, dengue, COPD, drug interactions |
| drug_formulary.pdf | clinical | doctor, admin | PDF | 6 | antimicrobials, cardiovascular, analgesics, respiratory, GI/endocrine, renal adjustment, high-alert, LASA, IV fluids |
| diagnostic_reference.pdf | clinical | doctor, admin | PDF | 5 | haematology, biochemistry, urine, ECG, ABG, thyroid/cardiac markers, radiology, microbiology, tumour markers, critical values |
| icu_nursing_procedures.pdf | nursing | nurse, doctor, admin | PDF | 8 | 7 SOPs: CVC care, ventilator, NGT, IV cannula, restraint, suctioning, pressure injury |
| infection_control.pdf | nursing | nurse, doctor, admin | PDF | 5 | hand hygiene, PPE, standard/transmission precautions, HAI surveillance, outbreak management, waste segregation, sharps, stewardship, isolation signage |
| billing_codes.pdf | billing | billing_executive, admin | PDF | 6 | top 30 ICD-10 codes + packages, procedure codes, insurer panel, exclusions, room rent sub-limits, co-pay |
| claim_submission_guide.md | billing | billing_executive, admin | MD | — | cashless/reimbursement/enhancement/rejection processes, escalation matrix, fraud prevention, KPIs |
| equipment_manual.pdf | equipment | technician, admin | PDF | 8 | BM-500 monitor, DriveFlow IP-200 pump, SterilPro 3000 autoclave, RadiPro MX-150 X-ray, fault codes, PM calendar, commissioning, decontamination |

---

## 2. Database schema

### claims (85 rows)

| Column | Type | Description | Distinct values |
|---|---|---|---|
| claim_id | TEXT PK | Format: CLM-2024-XXXX | 85 unique |
| patient_id | TEXT | Format: PAT-XXXXX | — |
| patient_name | TEXT | — | — |
| department | TEXT | Medical department | nephrology, cardiology, neurology, gynaecology, orthopaedics, general_medicine, emergency |
| claim_type | TEXT | — | cashless, reimbursement |
| diagnosis_code | TEXT | ICD-10 | — |
| insurer | TEXT | — | Star Health, HDFC Ergo, ICICI Lombard, New India Assurance, United India, Bajaj Allianz, Niva Bupa, Care Health |
| claimed_amount | REAL | ₹ | — |
| approved_amount | REAL (nullable) | ₹, NULL if not yet approved | — |
| status | TEXT | — | pending (17), approved (44), rejected (12), submitted (4), escalated (8) |
| submitted_date | TEXT | YYYY-MM-DD | — |
| resolved_date | TEXT (nullable) | YYYY-MM-DD | — |

### maintenance_tickets (78 rows)

| Column | Type | Description | Distinct values |
|---|---|---|---|
| ticket_id | TEXT PK | Format: TKT-2024-XXXX | 78 unique |
| equipment_name | TEXT | — | SterilPro 3000, DriveFlow IP-200, RadiPro MX-150, BM-500 Monitor, ElectroCautery EC-90, HemaCount HC-20 |
| equipment_id | TEXT | Format: EQ-XX-XXXX | — |
| category | TEXT | — | sterilisation, infusion, radiology, monitoring, surgical, laboratory |
| campus | TEXT | — | Hyderabad Central, Bengaluru Onco, Pune Speciality, Secunderabad, Mysuru Clinic Hub |
| issue_type | TEXT | — | preventive_maintenance, sensor_failure, battery_replacement, fault_reported, calibration_due |
| fault_code | TEXT (nullable) | Maps to equipment_manual codes | — |
| raised_by | TEXT | — | — |
| raised_date | TEXT | YYYY-MM-DD | — |
| resolved_date | TEXT (nullable) | — | — |
| status | TEXT | — | open (11), in_progress (15), resolved (42), escalated (10) |
| resolution_note | TEXT (nullable) | — | 8 distinct non-null values |

### Pre-computed reference answers

| Query | Answer | Notes |
|---|---|---|
| COUNT(*) FROM claims WHERE status='escalated' | 8 | |
| department with most claims | cardiology (20) | |
| category with most open+in_progress tickets | radiology (8) and monitoring (8) — **tied** | |
| SUM(approved_amount) | 24,42,700 | |
| rejection rate | 14.1% (12/85) | |
| COUNT(*) WHERE status='resolved' (tickets) | 42 | |
| insurer with highest SUM(claimed_amount) | Bajaj Allianz (13,53,000) | |

---

## 3. Cross-document references

Some entities appear in multiple collections — important for RBAC testing:

| Entity | clinical | billing | nursing | equipment | general |
|---|---|---|---|---|---|
| NSTEMI / I21.4 | treatment protocol | package ₹1,20,000 + claim guide example | — | — | — |
| Antimicrobials | drug formulary (doses) | — | stewardship (nursing role) | — | — |
| Infection control | — | — | full guidelines | equipment decontamination | — |
| Emergency codes | — | — | — | — | staff handbook |
