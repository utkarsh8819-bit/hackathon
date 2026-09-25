from src.normalization import normalize_business_name, normalize_country
from src.address_parser import normalize_address, extract_postal_code, extract_house_or_building


def test_business_name_normalization():
    res1 = normalize_business_name("Apex Logistics & Transport, LLC")
    assert "llc" in res1["name_norm"]
    assert "and" in res1["name_norm"]
    assert "apex" in res1["core_tokens"]
    assert "transport" in res1["core_tokens"]
    assert len(res1["phonetic_key"]) > 0

    res2 = normalize_business_name("Reliance Retail Pvt. Ltd.")
    assert "private limited" in res2["name_norm"]
    assert "reliance" in res2["core_tokens"]
    assert "retail" in res2["core_tokens"]


def test_address_normalization_us():
    addr = "1200 Market St, Ste 400, Philadelphia, PA 19107"
    res = normalize_address(addr)
    assert res["postal_code"] == "19107"
    assert "street" in res["addr_norm"]
    assert "suite" in res["addr_norm"]
    assert "1200" in res["house_number"]
    assert "philadelphia" in res["distinctive_tokens"]


def test_address_normalization_india_with_landmark():
    addr = "Plot 45, MIDC Industrial Area, Near SBI ATM, Andheri East, Mumbai, Maharashtra 400093"
    res = normalize_address(addr)
    assert res["postal_code"] == "400093"
    assert "near sbi atm" in res["landmark"]
    assert "plot 45" in res["house_number"]
    assert "andheri" in res["distinctive_tokens"]
    assert "mumbai" in res["distinctive_tokens"]


def test_address_normalization_france():
    addr = "15 Rue de Rivoli, 75001 Paris"
    res = normalize_address(addr)
    assert res["postal_code"] == "75001"
    assert "paris" in res["distinctive_tokens"]
    assert "15" in res["house_number"]


def test_country_normalization_open_set():
    assert normalize_country("us") == "US"
    assert normalize_country("  India  ") == "INDIA"
    assert normalize_country("France") == "FRANCE"
    assert normalize_country("New Zealand") == "NEW ZEALAND"
