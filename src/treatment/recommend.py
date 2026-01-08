def recommend_treatment(stage):
    if stage == "Non_Demented":
        return {
            "Stage": "Normal",
            "Treatment": ["No medication required"],
            "Recommendations": [
                "Mediterranean diet",
                "Exercise 150 min/week",
                "7+ hours sleep"
            ]
        }

    if stage == "Very_Mild_Demented":
        return {
            "Stage": "MCI / Very Mild",
            "Treatment": ["Lifestyle modification"],
            "Recommendations": [
                "Omega-3 supplementation",
                "Cognitive therapy",
                "Follow-up MRI in 12–18 months"
            ]
        }

    if stage == "Mild_Demented":
        return {
            "Stage": "Mild Alzheimer's",
            "Treatment": [
                "Donepezil 5–10 mg/day",
                "Rivastigmine patch"
            ],
            "Recommendations": [
                "Caregiver education",
                "Driving assessment",
                "MRI in 12 months"
            ]
        }

    if stage == "Moderate_Demented":
        return {
            "Stage": "Moderate Alzheimer's",
            "Treatment": [
                "Memantine 10–20 mg/day",
                "Combination with Donepezil"
            ],
            "Recommendations": [
                "Home safety evaluation",
                "Occupational therapy",
                "Neurology follow-up"
            ]
        }
