# Manually define categorical mappings (same as used during training)
category_mappings = {
    "change": {"No": 1, "Ch": 0},
    "gender": {"Male": 1, "Female": 0, "Unknown/Other": 2},
    "age": {"Age 0-20": 1, "Age 20-70": 0, "Age 70-100": 2},
    "diabetesMed": {"Yes": 1, "No": 0}
    }

# Function to encode inputs using the dictionary
def encode_input(data):
    encoded_values = []
    
    for key, mapping in category_mappings.items():
        value = getattr(data, key)
        encoded_values.append(mapping.get(value, -1))  # Use -1 for unknown values

    return encoded_values
