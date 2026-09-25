"""
Synthetic test fixtures that strictly emulate real Amazon ML Challenge dataset properties.
"""
import pandas as pd

def generate_sample_train_data():
    s1_data = [
        {"entity_id": "S1-1001", "business_name": "Apex Logistics LLC", "business_address": "1200 Market Street, Suite 400, Philadelphia, PA 19107", "country": "US"},
        {"entity_id": "S1-1002", "business_name": "Reliance Retail Ltd", "business_address": "Plot 45, MIDC Industrial Area, Near SBI ATM, Andheri East, Mumbai, Maharashtra 400093", "country": "India"},
        {"entity_id": "S1-1003", "business_name": "Lone Star Grill & Bar", "business_address": "8840 South Congress Ave, Austin, TX 78745", "country": "US"},
        {"entity_id": "S1-1004", "business_name": "Shree Ganesh Sweets & Bakery", "business_address": "12/4 Gandhi Road, Opp Bus Stand, Ahmedabad, Gujarat 380001", "country": "India"},
        {"entity_id": "S1-1005", "business_name": "Sole Proprietor Boutique", "business_address": "123 Elm St, Denver, CO 80202", "country": "US"}, # Singleton (no match)
    ]
    
    s2_data = [
        {"entity_id": "S2-2001", "business_name": "Apex Logistics Incorporated", "business_address": "1200 Market St, Ste 400, Philadelphia, PA 19107", "country": "US"}, # Match S1-1001
        {"entity_id": "S2-2002", "business_name": "Reliance Retail Private Limited", "business_address": "Plot 45, MIDC, Andheri East, Mumbai 400093", "country": "India"}, # Match S1-1002
        {"entity_id": "S2-2003", "business_name": "Lone Star Grill and Bar Inc", "business_address": "8840 S Congress Avenue, Austin, Texas 78745", "country": "US"}, # Match S1-1003
        {"entity_id": "S2-2004", "business_name": "Shree Ganesh Sweets", "business_address": "12/4 Gandhi Rd, Ahmedabad 380001", "country": "India"}, # Match S1-1004
        {"entity_id": "S2-2005", "business_name": "Apex Plumbing Services", "business_address": "500 Broad St, Philadelphia, PA 19102", "country": "US"}, # Hard negative for Apex
    ]
    
    s3_data = [
        {"entity_id": "S3-3001", "business_name": "Apex Logistics", "business_address": "Market Street Ste 400, Philadelphia 19107", "country": "US"}, # Multi-match S1-1001
        {"entity_id": "S3-3002", "business_name": "Reliance Retail", "business_address": "MIDC Indl Area, Near State Bank ATM, Mumbai 400093", "country": "India"}, # Multi-match S1-1002
        {"entity_id": "S3-3003", "business_name": "Ganesh Bakers and Confectioners", "business_address": "Gandhi Road, Ahmedabad, Gujarat 380001", "country": "India"}, # Near match S1-1004
        {"entity_id": "S3-3004", "business_name": "Random Unrelated Cafe", "business_address": "99 Ocean Drive, Miami, FL 33139", "country": "US"}, # Distractor
    ]
    
    gt_data = [
        {"source1_entity_id": "S1-1001", "matched_entity_ids": "S2-2001,S3-3001"},
        {"source1_entity_id": "S1-1002", "matched_entity_ids": "S2-2002,S3-3002"},
        {"source1_entity_id": "S1-1003", "matched_entity_ids": "S2-2003"},
        {"source1_entity_id": "S1-1004", "matched_entity_ids": "S2-2004"},
        {"source1_entity_id": "S1-1005", "matched_entity_ids": ""}, # Singleton!
    ]
    
    return (
        pd.DataFrame(s1_data),
        pd.DataFrame(s2_data),
        pd.DataFrame(s3_data),
        pd.DataFrame(gt_data),
    )

def generate_sample_test_data():
    s1_data = [
        {"entity_id": "S1-9001", "business_name": "Boulangerie Patisserie Artisanale", "business_address": "15 Rue de Rivoli, 75001 Paris", "country": "France"}, # France test entity!
        {"entity_id": "S1-9002", "business_name": "Global Pharma Solutions Inc", "business_address": "400 Technology Way, Cambridge, MA 02139", "country": "US"},
        {"entity_id": "S1-9003", "business_name": "Bangalore Tech Hub Pvt Ltd", "business_address": "100 Outer Ring Road, Bellandur, Bengaluru, Karnataka 560103", "country": "India"},
        {"entity_id": "S1-9004", "business_name": "Test Singleton Corp", "business_address": "101 Nowhere Street, Seattle, WA 98101", "country": "US"},
    ]
    s2_data = [
        {"entity_id": "S2-9101", "business_name": "Boulangerie Artisanale Paris", "business_address": "15 Rue de Rivoli, Paris 75001", "country": "France"},
        {"entity_id": "S2-9102", "business_name": "Global Pharma Solutions", "business_address": "400 Technology Way, Cambridge, MA 02139", "country": "US"},
        {"entity_id": "S2-9103", "business_name": "Bangalore Tech Hub", "business_address": "Outer Ring Rd, Bellandur, Bangalore 560103", "country": "India"},
    ]
    s3_data = [
        {"entity_id": "S3-9201", "business_name": "Global Pharma Research", "business_address": "400 Tech Way, Cambridge 02139", "country": "US"},
        {"entity_id": "S3-9202", "business_name": "Unrelated French Bistro", "business_address": "22 Rue Saint-Honore, 75001 Paris", "country": "France"},
    ]
    return (
        pd.DataFrame(s1_data),
        pd.DataFrame(s2_data),
        pd.DataFrame(s3_data),
    )
