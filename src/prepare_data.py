"""
Data Preparation & Harmonization Pipeline for Vanishing Voices (Language Extinction Predictor)
Merges UNESCO Endangered Languages Atlas with curated Safe World Languages Corpus.
Implements category-aware imputation and spatial hotspot indexing.
"""

import os
import json
import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

# -------------------------------------------------------------
# Curated Safe World Languages Corpus (Diverse Global Coverage)
# -------------------------------------------------------------
SAFE_LANGUAGES = [
    # Asia
    {"Name in English": "Mandarin Chinese", "Number of speakers": 1120000000, "Latitude": 35.8617, "Longitude": 104.1954, "Countries": "China, Taiwan, Singapore", "Country codes alpha 3": "CHN, TWN, SGP", "Degree of endangerment": "Safe"},
    {"Name in English": "Hindi", "Number of speakers": 602000000, "Latitude": 28.6139, "Longitude": 77.2090, "Countries": "India", "Country codes alpha 3": "IND", "Degree of endangerment": "Safe"},
    {"Name in English": "Bengali", "Number of speakers": 272000000, "Latitude": 23.6850, "Longitude": 90.3563, "Countries": "Bangladesh, India", "Country codes alpha 3": "BGD, IND", "Degree of endangerment": "Safe"},
    {"Name in English": "Telugu", "Number of speakers": 96000000, "Latitude": 17.3850, "Longitude": 78.4867, "Countries": "India", "Country codes alpha 3": "IND", "Degree of endangerment": "Safe"},
    {"Name in English": "Marathi", "Number of speakers": 95000000, "Latitude": 19.7515, "Longitude": 75.7139, "Countries": "India", "Country codes alpha 3": "IND", "Degree of endangerment": "Safe"},
    {"Name in English": "Tamil", "Number of speakers": 85000000, "Latitude": 13.0827, "Longitude": 80.2707, "Countries": "India, Sri Lanka, Singapore, Malaysia", "Country codes alpha 3": "IND, LKA, SGP, MYS", "Degree of endangerment": "Safe"},
    {"Name in English": "Urdu", "Number of speakers": 231000000, "Latitude": 30.3753, "Longitude": 69.3451, "Countries": "Pakistan, India", "Country codes alpha 3": "PAK, IND", "Degree of endangerment": "Safe"},
    {"Name in English": "Gujarati", "Number of speakers": 62000000, "Latitude": 22.2587, "Longitude": 71.1924, "Countries": "India", "Country codes alpha 3": "IND", "Degree of endangerment": "Safe"},
    {"Name in English": "Kannada", "Number of speakers": 59000000, "Latitude": 15.3173, "Longitude": 75.7139, "Countries": "India", "Country codes alpha 3": "IND", "Degree of endangerment": "Safe"},
    {"Name in English": "Malayalam", "Number of speakers": 38000000, "Latitude": 10.8505, "Longitude": 76.2711, "Countries": "India", "Country codes alpha 3": "IND", "Degree of endangerment": "Safe"},
    {"Name in English": "Odia", "Number of speakers": 35000000, "Latitude": 20.9517, "Longitude": 85.0985, "Countries": "India", "Country codes alpha 3": "IND", "Degree of endangerment": "Safe"},
    {"Name in English": "Punjabi", "Number of speakers": 113000000, "Latitude": 31.1471, "Longitude": 75.3412, "Countries": "Pakistan, India", "Country codes alpha 3": "PAK, IND", "Degree of endangerment": "Safe"},
    {"Name in English": "Japanese", "Number of speakers": 125000000, "Latitude": 36.2048, "Longitude": 138.2529, "Countries": "Japan", "Country codes alpha 3": "JPN", "Degree of endangerment": "Safe"},
    {"Name in English": "Korean", "Number of speakers": 82000000, "Latitude": 35.9078, "Longitude": 127.7669, "Countries": "South Korea, North Korea", "Country codes alpha 3": "KOR, PRK", "Degree of endangerment": "Safe"},
    {"Name in English": "Vietnamese", "Number of speakers": 85000000, "Latitude": 14.0583, "Longitude": 108.2772, "Countries": "Vietnam", "Country codes alpha 3": "VNM", "Degree of endangerment": "Safe"},
    {"Name in English": "Thai", "Number of speakers": 61000000, "Latitude": 15.8700, "Longitude": 100.9925, "Countries": "Thailand", "Country codes alpha 3": "THA", "Degree of endangerment": "Safe"},
    {"Name in English": "Indonesian", "Number of speakers": 199000000, "Latitude": -0.7893, "Longitude": 113.9213, "Countries": "Indonesia", "Country codes alpha 3": "IDN", "Degree of endangerment": "Safe"},
    {"Name in English": "Javanese", "Number of speakers": 68000000, "Latitude": -7.6145, "Longitude": 110.7122, "Countries": "Indonesia", "Country codes alpha 3": "IDN", "Degree of endangerment": "Safe"},
    {"Name in English": "Tagalog (Filipino)", "Number of speakers": 82000000, "Latitude": 12.8797, "Longitude": 121.7740, "Countries": "Philippines", "Country codes alpha 3": "PHL", "Degree of endangerment": "Safe"},
    {"Name in English": "Burmese", "Number of speakers": 33000000, "Latitude": 21.9162, "Longitude": 95.9560, "Countries": "Myanmar", "Country codes alpha 3": "MMR", "Degree of endangerment": "Safe"},
    {"Name in English": "Persian (Farsi)", "Number of speakers": 77000000, "Latitude": 32.4279, "Longitude": 53.6880, "Countries": "Iran, Afghanistan, Tajikistan", "Country codes alpha 3": "IRN, AFG, TJK", "Degree of endangerment": "Safe"},
    {"Name in English": "Turkish", "Number of speakers": 88000000, "Latitude": 38.9637, "Longitude": 35.2433, "Countries": "Turkey, Cyprus", "Country codes alpha 3": "TUR, CYP", "Degree of endangerment": "Safe"},
    {"Name in English": "Arabic", "Number of speakers": 374000000, "Latitude": 23.8859, "Longitude": 45.0792, "Countries": "Saudi Arabia, Egypt, Iraq, UAE, Morocco, Algeria", "Country codes alpha 3": "SAU, EGY, IRQ, ARE, MAR, DZA", "Degree of endangerment": "Safe"},

    # Europe
    {"Name in English": "English", "Number of speakers": 1456000000, "Latitude": 52.3555, "Longitude": -1.1743, "Countries": "United Kingdom, United States, Canada, Australia, New Zealand", "Country codes alpha 3": "GBR, USA, CAN, AUS, NZL", "Degree of endangerment": "Safe"},
    {"Name in English": "Spanish", "Number of speakers": 548000000, "Latitude": 40.4637, "Longitude": -3.7492, "Countries": "Spain, Mexico, Colombia, Argentina, Peru", "Country codes alpha 3": "ESP, MEX, COL, ARG, PER", "Degree of endangerment": "Safe"},
    {"Name in English": "French", "Number of speakers": 310000000, "Latitude": 46.2276, "Longitude": 2.2137, "Countries": "France, Canada, Belgium, Switzerland, Senegal", "Country codes alpha 3": "FRA, CAN, BEL, CHE, SEN", "Degree of endangerment": "Safe"},
    {"Name in English": "German", "Number of speakers": 134000000, "Latitude": 51.1657, "Longitude": 10.4515, "Countries": "Germany, Austria, Switzerland", "Country codes alpha 3": "DEU, AUT, CHE", "Degree of endangerment": "Safe"},
    {"Name in English": "Russian", "Number of speakers": 258000000, "Latitude": 61.5240, "Longitude": 105.3188, "Countries": "Russian Federation, Belarus, Kazakhstan", "Country codes alpha 3": "RUS, BLR, KAZ", "Degree of endangerment": "Safe"},
    {"Name in English": "Portuguese", "Number of speakers": 258000000, "Latitude": 39.3999, "Longitude": -8.2245, "Countries": "Portugal, Brazil, Angola, Mozambique", "Country codes alpha 3": "PRT, BRA, AGO, MOZ", "Degree of endangerment": "Safe"},
    {"Name in English": "Italian", "Number of speakers": 68000000, "Latitude": 41.8719, "Longitude": 12.5674, "Countries": "Italy, Switzerland", "Country codes alpha 3": "ITA, CHE", "Degree of endangerment": "Safe"},
    {"Name in English": "Polish", "Number of speakers": 45000000, "Latitude": 51.9194, "Longitude": 19.1451, "Countries": "Poland", "Country codes alpha 3": "POL", "Degree of endangerment": "Safe"},
    {"Name in English": "Ukrainian", "Number of speakers": 33000000, "Latitude": 48.3794, "Longitude": 31.1656, "Countries": "Ukraine", "Country codes alpha 3": "UKR", "Degree of endangerment": "Safe"},
    {"Name in English": "Dutch", "Number of speakers": 25000000, "Latitude": 52.1326, "Longitude": 5.2913, "Countries": "Netherlands, Belgium, Suriname", "Country codes alpha 3": "NLD, BEL, SUR", "Degree of endangerment": "Safe"},
    {"Name in English": "Romanian", "Number of speakers": 24000000, "Latitude": 45.9432, "Longitude": 24.9668, "Countries": "Romania, Moldova", "Country codes alpha 3": "ROU, MDA", "Degree of endangerment": "Safe"},
    {"Name in English": "Greek", "Number of speakers": 13500000, "Latitude": 39.0742, "Longitude": 21.8243, "Countries": "Greece, Cyprus", "Country codes alpha 3": "GRC, CYP", "Degree of endangerment": "Safe"},
    {"Name in English": "Czech", "Number of speakers": 10700000, "Latitude": 49.8175, "Longitude": 15.4730, "Countries": "Czech Republic", "Country codes alpha 3": "CZE", "Degree of endangerment": "Safe"},
    {"Name in English": "Hungarian", "Number of speakers": 12600000, "Latitude": 47.1625, "Longitude": 19.5033, "Countries": "Hungary", "Country codes alpha 3": "HUN", "Degree of endangerment": "Safe"},
    {"Name in English": "Swedish", "Number of speakers": 10000000, "Latitude": 60.1282, "Longitude": 18.6435, "Countries": "Sweden, Finland", "Country codes alpha 3": "SWE, FIN", "Degree of endangerment": "Safe"},

    # Africa
    {"Name in English": "Swahili", "Number of speakers": 200000000, "Latitude": -6.3690, "Longitude": 34.8888, "Countries": "Tanzania, Kenya, Uganda, DRC", "Country codes alpha 3": "TZA, KEN, UGA, COD", "Degree of endangerment": "Safe"},
    {"Name in English": "Hausa", "Number of speakers": 77000000, "Latitude": 9.0820, "Longitude": 8.6753, "Countries": "Nigeria, Niger, Ghana", "Country codes alpha 3": "NGA, NER, GHA", "Degree of endangerment": "Safe"},
    {"Name in English": "Yoruba", "Number of speakers": 45000000, "Latitude": 7.3775, "Longitude": 3.9470, "Countries": "Nigeria, Benin", "Country codes alpha 3": "NGA, BEN", "Degree of endangerment": "Safe"},
    {"Name in English": "Igbo", "Number of speakers": 31000000, "Latitude": 5.4763, "Longitude": 7.0259, "Countries": "Nigeria", "Country codes alpha 3": "NGA", "Degree of endangerment": "Safe"},
    {"Name in English": "Amharic", "Number of speakers": 57000000, "Latitude": 9.1450, "Longitude": 40.4897, "Countries": "Ethiopia", "Country codes alpha 3": "ETH", "Degree of endangerment": "Safe"},
    {"Name in English": "Oromo", "Number of speakers": 37000000, "Latitude": 8.5410, "Longitude": 39.2680, "Countries": "Ethiopia, Kenya", "Country codes alpha 3": "ETH, KEN", "Degree of endangerment": "Safe"},
    {"Name in English": "Zulu", "Number of speakers": 27000000, "Latitude": -28.5306, "Longitude": 30.8958, "Countries": "South Africa", "Country codes alpha 3": "ZAF", "Degree of endangerment": "Safe"},
    {"Name in English": "Xhosa", "Number of speakers": 19000000, "Latitude": -32.2968, "Longitude": 26.4194, "Countries": "South Africa", "Country codes alpha 3": "ZAF", "Degree of endangerment": "Safe"},
    {"Name in English": "Shona", "Number of speakers": 15000000, "Latitude": -19.0154, "Longitude": 29.1549, "Countries": "Zimbabwe, Mozambique", "Country codes alpha 3": "ZWE, MOZ", "Degree of endangerment": "Safe"},
    {"Name in English": "Somali", "Number of speakers": 22000000, "Latitude": 5.1521, "Longitude": 46.1996, "Countries": "Somalia, Ethiopia, Kenya", "Country codes alpha 3": "SOM, ETH, KEN", "Degree of endangerment": "Safe"},
    {"Name in English": "Afrikaans", "Number of speakers": 17000000, "Latitude": -30.5595, "Longitude": 22.9375, "Countries": "South Africa, Namibia", "Country codes alpha 3": "ZAF, NAM", "Degree of endangerment": "Safe"},
    {"Name in English": "Lingala", "Number of speakers": 40000000, "Latitude": -4.0383, "Longitude": 21.7587, "Countries": "DRC, Republic of Congo", "Country codes alpha 3": "COD, COG", "Degree of endangerment": "Safe"},

    # Americas & Oceania (Vital Indigenous & National Languages)
    {"Name in English": "Guarani", "Number of speakers": 6500000, "Latitude": -23.4425, "Longitude": -58.4438, "Countries": "Paraguay, Bolivia, Argentina, Brazil", "Country codes alpha 3": "PRY, BOL, ARG, BRA", "Degree of endangerment": "Safe"},
    {"Name in English": "Quechua (Cusco-Collao)", "Number of speakers": 3800000, "Latitude": -13.5319, "Longitude": -71.9675, "Countries": "Peru, Bolivia", "Country codes alpha 3": "PER, BOL", "Degree of endangerment": "Safe"},
    {"Name in English": "Aymara", "Number of speakers": 2000000, "Latitude": -16.5000, "Longitude": -68.1500, "Countries": "Bolivia, Peru", "Country codes alpha 3": "BOL, PER", "Degree of endangerment": "Safe"},
    {"Name in English": "Fijian", "Number of speakers": 450000, "Latitude": -17.7134, "Longitude": 178.0650, "Countries": "Fiji", "Country codes alpha 3": "FJI", "Degree of endangerment": "Safe"},
    {"Name in English": "Samoan", "Number of speakers": 510000, "Latitude": -13.7590, "Longitude": -172.1046, "Countries": "Samoa, American Samoa", "Country codes alpha 3": "WSM, ASM", "Degree of endangerment": "Safe"},
    {"Name in English": "Tongan", "Number of speakers": 180000, "Latitude": -21.1789, "Longitude": -175.1982, "Countries": "Tonga", "Country codes alpha 3": "TON", "Degree of endangerment": "Safe"},
    {"Name in English": "Hebrew", "Number of speakers": 9300000, "Latitude": 31.0461, "Longitude": 34.8516, "Countries": "Israel", "Country codes alpha 3": "ISR", "Degree of endangerment": "Safe"},
    {"Name in English": "Finnish", "Number of speakers": 5800000, "Latitude": 61.9241, "Longitude": 25.7482, "Countries": "Finland, Sweden", "Country codes alpha 3": "FIN, SWE", "Degree of endangerment": "Safe"},
    {"Name in English": "Danish", "Number of speakers": 6000000, "Latitude": 56.2639, "Longitude": 9.5018, "Countries": "Denmark, Greenland", "Country codes alpha 3": "DNK, GRL", "Degree of endangerment": "Safe"},
    {"Name in English": "Norwegian", "Number of speakers": 5400000, "Latitude": 60.4720, "Longitude": 8.4689, "Countries": "Norway", "Country codes alpha 3": "NOR", "Degree of endangerment": "Safe"}
]


def assign_macro_region(lat: float, lon: float) -> str:
    """Classifies a coordinate into a macro continent/region."""
    if -60 <= lat <= 15 and -90 <= lon <= -30:
        return "South America"
    elif 10 <= lat <= 85 and -170 <= lon <= -50:
        return "North America"
    elif 35 <= lat <= 75 and -15 <= lon <= 45:
        return "Europe"
    elif -35 <= lat <= 38 and -20 <= lon <= 55:
        return "Africa"
    elif -50 <= lat <= 0 and 110 <= lon <= 180:
        return "Oceania"
    elif 0 <= lat <= 75 and 45 <= lon <= 180:
        return "Asia"
    else:
        return "Other"


def load_and_clean_data(raw_csv_path: str) -> pd.DataFrame:
    """
    Cleans raw UNESCO dataset and merges with safe language corpus.
    Applies domain-informed imputation and builds engineered feature signals.
    """
    df_raw = pd.read_csv(raw_csv_path)

    # 1. Clean Coordinates (Impute the 3 missing points with reasonable country centroids)
    df_raw.loc[df_raw['Name in English'] == 'South Italian', 'Latitude'] = 40.9798
    df_raw.loc[df_raw['Name in English'] == 'South Italian', 'Longitude'] = 15.2490
    df_raw['Latitude'] = df_raw['Latitude'].fillna(0.0)
    df_raw['Longitude'] = df_raw['Longitude'].fillna(0.0)

    # 2. Impute Number of Speakers using Endangerment Category Medians
    # Extinct languages have 0 living speakers.
    category_medians = df_raw.groupby('Degree of endangerment')['Number of speakers'].median().to_dict()
    category_medians['Extinct'] = 0.0

    def impute_speakers(row):
        val = row['Number of speakers']
        status = row['Degree of endangerment']
        if pd.isnull(val):
            return category_medians.get(status, 100.0)
        return float(val)

    df_raw['Number of speakers'] = df_raw.apply(impute_speakers, axis=1)

    # 3. Add Safe Languages Corpus
    df_safe = pd.DataFrame(SAFE_LANGUAGES)

    # Combine datasets
    columns_to_keep = [
        'Name in English', 'Number of speakers', 'Latitude', 'Longitude',
        'Countries', 'Country codes alpha 3', 'Degree of endangerment'
    ]
    df_combined = pd.concat([df_raw[columns_to_keep], df_safe[columns_to_keep]], ignore_index=True)
    df_combined['Countries'] = df_combined['Countries'].fillna('Unknown')
    df_combined['Country codes alpha 3'] = df_combined['Country codes alpha 3'].fillna('UNK')

    # 4. Feature Engineering
    # A. Log Population
    df_combined['log_speakers'] = np.log10(np.maximum(df_combined['Number of speakers'], 0) + 1.0)

    # B. Country Breadth & Transnational Presence
    def count_countries(country_str):
        if not country_str or pd.isnull(country_str) or country_str == 'Unknown':
            return 1
        return max(1, len(str(country_str).split(',')))

    df_combined['num_countries'] = df_combined['Countries'].apply(count_countries)
    df_combined['is_transnational'] = (df_combined['num_countries'] > 1).astype(int)
    df_combined['speakers_per_country'] = df_combined['Number of speakers'] / df_combined['num_countries']
    df_combined['log_speakers_per_country'] = np.log10(np.maximum(df_combined['speakers_per_country'], 0) + 1.0)

    # C. Geographic Zone & Latitude Distance
    df_combined['abs_latitude'] = df_combined['Latitude'].abs()
    # 0 = Tropical (<=23.5°), 1 = Temperate (23.5° to 55°), 2 = High Latitude / Boreal (>55°)
    df_combined['climate_zone'] = pd.cut(
        df_combined['abs_latitude'],
        bins=[-0.1, 23.5, 55.0, 95.0],
        labels=[0, 1, 2]
    ).astype(int)

    # D. Macro Region
    df_combined['macro_region'] = [
        assign_macro_region(lat, lon)
        for lat, lon in zip(df_combined['Latitude'], df_combined['Longitude'])
    ]

    # E. Spatial Risk Hotspot Density (BallTree with Haversine Distance)
    # Computes how many endangered languages exist within a ~500 km radius
    rad_coords = np.radians(df_combined[['Latitude', 'Longitude']].values)
    tree = BallTree(rad_coords, metric='haversine')
    # 500 km / 6371 km radius of earth = ~0.0785 radians
    radius_rad = 500.0 / 6371.0
    # Query neighbor count (subtract 1 for self)
    counts = tree.query_radius(rad_coords, r=radius_rad, count_only=True)
    df_combined['nearby_language_density'] = np.maximum(0, counts - 1)

    # 5. Define Targets
    # Target 1: Binary Risk (0 = Safe/Vulnerable, 1 = Threatened/Critically Endangered/Extinct)
    high_risk_classes = {'Critically endangered', 'Severely endangered', 'Definitely endangered', 'Extinct'}
    df_combined['target_binary'] = df_combined['Degree of endangerment'].apply(
        lambda x: 1 if x in high_risk_classes else 0
    )

    # Target 2: 4-Level Ordinal Risk Tier
    # 0: Safe (Global / National vitality)
    # 1: Vulnerable (Domain restricted / minor decline)
    # 2: Endangered (Severely / Definitely endangered)
    # 3: Critically Endangered / Extinct
    def get_ordinal_tier(deg):
        if deg == 'Safe':
            return 0
        elif deg == 'Vulnerable':
            return 1
        elif deg in ('Definitely endangered', 'Severely endangered'):
            return 2
        elif deg in ('Critically endangered', 'Extinct'):
            return 3
        return 1

    df_combined['target_ordinal'] = df_combined['Degree of endangerment'].apply(get_ordinal_tier)

    return df_combined


if __name__ == "__main__":
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    raw_path = os.path.join(base_dir, "data", "languages.csv")
    out_path = os.path.join(base_dir, "data", "languages_clean.csv")

    print(f"Loading raw data from: {raw_path}")
    df_clean = load_and_clean_data(raw_path)
    df_clean.to_csv(out_path, index=False)
    print(f"Processed dataset saved to: {out_path}")
    print(f"Total Rows: {len(df_clean)}, Features: {df_clean.shape[1]}")
    print("\nTarget Binary Distribution:")
    print(df_clean['target_binary'].value_counts(normalize=True).round(3))
    print("\nTarget Ordinal Tier Distribution:")
    print(df_clean['target_ordinal'].value_counts().sort_index())
