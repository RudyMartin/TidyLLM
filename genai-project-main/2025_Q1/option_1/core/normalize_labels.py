
def normalize_label(label):
    label = label.strip().lower()
    if label in ['correct', '✔️', 'true', 'yes']:
        return 'Correct'
    elif label in ['missing info', 'incomplete', 'missing']:
        return 'Missing Info'
    elif label in ['inconsistent', 'wrong', 'no', 'false']:
        return 'Inconsistent'
    else:
        return 'Other'
