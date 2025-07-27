# Blood types and ABO compatibility
BLOOD_TYPES = ['O', 'A', 'B', 'AB']
BLOOD_TYPE_PROBS = [0.4814, 0.3373, 0.1428, 0.0385]  # From Saidman et al. (more precise)

# PRA levels (positive crossmatch probs) and their distribution
PRA_LEVELS = [0.05, 0.45, 0.90]  # Low, medium, high
PRA_PROBS = [0.7019, 0.20, 0.0981]

def is_abo_compatible(donor_bt: str, recipient_bt: str) -> bool:
    if donor_bt == 'O':
        return True
    if donor_bt == recipient_bt:
        return True
    if donor_bt in ['A', 'B'] and recipient_bt == 'AB':
        return True
    return False

def generate_pair() -> Dict[str, Any]:
    """
    Generate an incompatible patient-donor pair.
    - Sample blood types for patient and donor.
    - Sample cPRA (PRA level) for the patient.
    - If ABO incompatible: always incompatible, keep.
    - If ABO compatible: incompatible with probability cPRA (positive crossmatch), keep only then.
    """
    while True:
        patient_bt = np.random.choice(BLOOD_TYPES, p=BLOOD_TYPE_PROBS)
        donor_bt = np.random.choice(BLOOD_TYPES, p=BLOOD_TYPE_PROBS)
        cpra = np.random.choice(PRA_LEVELS, p=PRA_PROBS)
        if is_abo_compatible(donor_bt, patient_bt):
            if np.random.uniform() > cpra:  # Compatible (negative crossmatch), discard
                continue
        # Else: incompatible (either ABO or positive crossmatch), keep
        return {'patient_bt': patient_bt, 'donor_bt': donor_bt, 'cPRA': cpra}
    
def generate_altruistic() -> Dict[str, Any]:
    bt = np.random.choice(BLOOD_TYPES, p=BLOOD_TYPE_PROBS)
    return {'donor_bt': bt}