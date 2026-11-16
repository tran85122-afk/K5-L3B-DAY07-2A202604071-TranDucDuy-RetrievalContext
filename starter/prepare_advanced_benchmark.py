"""Prepare and validate the Day 7 Advanced Multimodal Benchmark (18 dev + 30 test + 30 fresh test_v2)."""
import argparse
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

STARTER_DIR = Path(__file__).resolve().parent
DEFAULT_CHUNKS = STARTER_DIR / "data" / "advanced" / "chunks_advanced.json"
DEFAULT_BASE_CASES = STARTER_DIR / "data" / "cases.json"
DEFAULT_OUT_DIR = STARTER_DIR / "data" / "advanced"

BENCHMARK_CASES: list[dict[str, Any]] = [
    # =========================================================================
    # DEV SPLIT (18 cases: 3 table, 3 diagram, 3 chart, 3 multihop, 3 text, 3 unanswerable)
    # =========================================================================
    {
        "id": "adv_d01",
        "split": "dev",
        "primary_group": "table",
        "answerable": True,
        "question": (
            "According to Table 1.1 (PDF page 48), what examples are listed under "
            "'Traditional' vs 'New' data collection methods for 'Public intent data' "
            "compared with 'Private intent data'?"
        ),
        "gold_chunk_ids": ["wdr2021:p48:vis_table_1_1"],
        "gold_pages": [48],
        "gold_elements": ["Table 1.1"],
        "expected_answer": (
            "In Table 1.1: (1) Traditional + Public intent data includes census, national accounts, "
            "household surveys, enterprise surveys, labor force surveys, surveys of personal finance, "
            "and administrative records; (2) Traditional + Private intent data includes any survey conducted "
            "by private entities (including public opinion surveys deployed by private entities) and "
            "administrative data from company financial accounts; (3) New + Public intent data includes "
            "location data from satellite imaging, digital identification, facial recognition from public "
            "cameras, and public procurement data from e-government platforms; (4) New + Private intent data "
            "includes just-in-time digital data on individual behavior/choices from digital platforms in the private sector."
        ),
        "required_facts": [
            "Traditional + Public: census, national accounts, household/enterprise/labor force/personal finance surveys, administrative records",
            "Traditional + Private: private surveys / public opinion surveys, administrative data from company financial accounts",
            "New + Public: satellite imaging location data, digital identification, public camera facial recognition, e-government procurement data",
            "New + Private: just-in-time digital data on individual behavior/choices from private digital platforms",
        ],
        "verify_tokens": ["Census", "national accounts", "satellite imaging", "Just-in-time digital data", "company financial accounts"],
    },
    {
        "id": "adv_d02",
        "split": "dev",
        "primary_group": "table",
        "answerable": True,
        "question": (
            "In Table 2.2 on NSO data dissemination practices (2019), what percentages are reported across "
            "Low-income, Lower-middle-income, Upper-middle-income, and High-income groups for "
            "'NSO uses advance release calendar' and 'NSO has conducted user satisfaction survey'?"
        ),
        "gold_chunk_ids": ["wdr2021:p84:vis_table_2_2"],
        "gold_pages": [84],
        "gold_elements": ["Table 2.2"],
        "expected_answer": (
            "In Table 2.2 (2019, % of population): 'NSO uses advance release calendar' is 30% in low-income, "
            "75% in lower-middle-income, 92% in upper-middle-income, and 98% in high-income countries. "
            "'NSO has conducted user satisfaction survey' is 10% in low-income, 20% in lower-middle-income, "
            "19% in upper-middle-income, and 33% in high-income countries."
        ),
        "required_facts": [
            "Advance release calendar: Low=30%, Lower-mid=75%, Upper-mid=92%, High=98%",
            "User satisfaction survey: Low=10%, Lower-mid=20%, Upper-mid=19%, High=33%",
        ],
        "verify_tokens": ["30", "75", "92", "98", "10", "20", "19", "33"],
    },
    {
        "id": "adv_d03",
        "split": "dev",
        "primary_group": "table",
        "answerable": True,
        "question": (
            "According to Table 8.1, what are the data management decisions listed specifically under the "
            "'Store', 'Transfer/share', and 'Destroy or reuse' stages of the data life cycle?"
        ),
        "gold_chunk_ids": ["wdr2021:p288:vis_table_8_1"],
        "gold_pages": [288],
        "gold_elements": ["Table 8.1"],
        "expected_answer": (
            "In Table 8.1: (1) Store: encrypt data, use secure servers, back up and archive data; "
            "(2) Transfer/share: verify whether consent allows for data to be shared, deidentify data if appropriate, "
            "sign confidentiality agreements for use of identified data, and publish data via bulk downloads or APIs; "
            "(3) Destroy or reuse: keep records of destruction processes, and verify that consent for use is still valid."
        ),
        "required_facts": [
            "Store: encrypt data, secure servers, back up and archive",
            "Transfer/share: verify consent, deidentify, confidentiality agreements, bulk downloads or APIs",
            "Destroy or reuse: keep records of destruction processes, verify consent is still valid",
        ],
        "verify_tokens": ["Encrypt data", "secure servers", "Deidentify data", "confidentiality agreements", "destruction processes"],
    },
    {
        "id": "adv_d04",
        "split": "dev",
        "primary_group": "diagram",
        "answerable": True,
        "question": (
            "According to Figure O.2, what are the three foundational pillars of the social contract for data, "
            "and how does the diagram define each pillar?"
        ),
        "gold_chunk_ids": ["wdr2021:p26:vis_figure_o_2"],
        "gold_pages": [26],
        "gold_elements": ["Figure O.2"],
        "expected_answer": (
            "Figure O.2 defines three pillars of the social contract for data: "
            "(1) VALUE — the full value of data materializes when systems enable the use and reuse of data for different purposes; "
            "(2) TRUST — a trust environment is created when the rights and interests that all stakeholders have in data are safeguarded; "
            "(3) EQUITY — all share equitably in the benefits of data when investments and regulations create a level playing field."
        ),
        "required_facts": [
            "VALUE: full value materializes when systems enable use and reuse for different purposes",
            "TRUST: rights and interests that all stakeholders have in data are safeguarded",
            "EQUITY: all share equitably in benefits when investments and regulations create a level playing field",
        ],
        "verify_tokens": ["VALUE", "TRUST", "EQUITY", "safeguarded", "create a level", "playing"],
    },
    {
        "id": "adv_d05",
        "split": "dev",
        "primary_group": "diagram",
        "answerable": True,
        "question": (
            "According to the diagram in Figure 1.2 ('The data life cycle'), what are the sequential stages "
            "from data creation to end of life, and what internal circular feedback mechanism connects them?"
        ),
        "gold_chunk_ids": ["wdr2021:p50:vis_figure_1_2"],
        "gold_pages": [50],
        "gold_elements": ["Figure 1.2"],
        "expected_answer": (
            "Figure 1.2 depicts the seven stages of the data life cycle: Create/receive -> Process -> Store -> "
            "Transfer/share -> Analyze and use -> Archive and preserve -> Destroy, with a central circular 'Reuse' loop "
            "illustrating repeated reuse and repurposing of data without depleting them."
        ),
        "required_facts": [
            "7 stages: Create/receive, Process, Store, Transfer/share, Analyze and use, Archive and preserve, Destroy",
            "Internal loop: Reuse",
        ],
        "verify_tokens": ["Create/receive", "Process", "Store", "Archive and", "Destroy", "Reuse"],
    },
    {
        "id": "adv_d06",
        "split": "dev",
        "primary_group": "diagram",
        "answerable": True,
        "question": (
            "In Figure 2.1, what four data feature groups maximize the value of public intent data, "
            "and what three specific bulleted attributes appear under each group?"
        ),
        "gold_chunk_ids": ["wdr2021:p75:vis_figure_2_1"],
        "gold_pages": [75],
        "gold_elements": ["Figure 2.1"],
        "expected_answer": (
            "Figure 2.1 lists four feature groups with three attributes each: "
            "(1) Adequate coverage: Completeness, Timeliness, Frequency; "
            "(2) High quality: Granularity, Accuracy, Comparability; "
            "(3) Easy to use: Accessibility, Understandability, Interoperability; "
            "(4) Safe to use: Impartiality, Confidentiality, Appropriateness."
        ),
        "required_facts": [
            "Adequate coverage: Completeness, Timeliness, Frequency",
            "High quality: Granularity, Accuracy, Comparability",
            "Easy to use: Accessibility, Understandability, Interoperability",
            "Safe to use: Impartiality, Confidentiality, Appropriateness",
        ],
        "verify_tokens": ["Completeness", "Timeliness", "Granularity", "Comparability", "Interoperability", "Impartiality", "Appropriateness"],
    },
    {
        "id": "adv_d07",
        "split": "dev",
        "primary_group": "chart",
        "answerable": True,
        "question": (
            "According to the bar values in Figure 1.1, what percentage of the world's population was enumerated "
            "in a census in 1850, 1950, 1960, 1970, and 2010?"
        ),
        "gold_chunk_ids": ["wdr2021:p46:vis_figure_1_1"],
        "gold_pages": [46],
        "gold_elements": ["Figure 1.1"],
        "expected_answer": (
            "In Figure 1.1, the share of the world's population counted in a census was 11% in 1850, "
            "82% in 1950, 94% in 1960, 71% in 1970, and 90% in 2010."
        ),
        "required_facts": [
            "1850 = 11%",
            "1950 = 82%",
            "1960 = 94%",
            "1970 = 71%",
            "2010 = 90%",
        ],
        "verify_tokens": ["11%", "82%", "94%", "71%", "90%"],
    },
    {
        "id": "adv_d08",
        "split": "dev",
        "primary_group": "chart",
        "answerable": True,
        "question": (
            "Based on the 2017/18 household surveys in 22 developing countries behind Figure 5.5, "
            "what were the top three reasons cited for not taking up data/internet services and their exact percentages?"
        ),
        "gold_chunk_ids": ["wdr2021:p184:vis_figure_5_5"],
        "gold_pages": [184],
        "gold_elements": ["Figure 5.5"],
        "expected_answer": (
            "For Figure 5.5 (2017/18 household surveys across 22 developing countries), the three most frequently cited "
            "reasons for not taking up data services are digital literacy (69 percent), affordability concerns (15 percent), "
            "and relevance issues (12 percent)."
        ),
        "required_facts": [
            "Digital literacy: 69%",
            "Affordability concerns: 15%",
            "Relevance issues: 12%",
        ],
        "verify_tokens": ["69 percent", "15 percent", "relevance issues (12", "digital literacy"],
    },
    {
        "id": "adv_d09",
        "split": "dev",
        "primary_group": "chart",
        "answerable": True,
        "question": (
            "According to Box 5.1 and Figure B5.1.1 (panels a and b), how does the proportion of students graduating "
            "with ICT degrees compare across country income groups overall versus the share of ICT professionals "
            "in total employment in high-income vs low-income countries?"
        ),
        "gold_chunk_ids": ["wdr2021:p194:vis_figure_b5_1_1"],
        "gold_pages": [194],
        "gold_elements": ["Figure B5.1.1"],
        "expected_answer": (
            "According to Box 5.1 and Figure B5.1.1: across country income groups, the share of students graduating "
            "with ICT degrees does not diverge widely and typically falls in the 7–8 percent range overall (with marked "
            "gender gaps between men and women in panel a). In contrast, ICT professionals account for 2.1 percent of "
            "total employment in high-income nations compared with only 0.1 percent in low-income countries (panel b)."
        ),
        "required_facts": [
            "ICT graduates share: 7-8 percent range overall across income groups",
            "ICT professionals in total employment: 2.1 percent in high-income vs 0.1 percent in low-income countries",
        ],
        "verify_tokens": ["7–8 percent", "2.1 percent", "0.1 per-"],
    },
    {
        "id": "adv_d10",
        "split": "dev",
        "primary_group": "multihop",
        "answerable": True,
        "question": (
            "Combining Table 2.1 and Figure 2.7, compare Low-income vs High-income countries across "
            "(a) average data openness score, (b) availability of open terms of use/license (%), and "
            "(c) percentage of countries with a fully funded national statistical plan (2019)."
        ),
        "gold_chunk_ids": ["wdr2021:p83:vis_table_2_1", "wdr2021:p86:vis_figure_2_7"],
        "gold_pages": [83, 86],
        "gold_elements": ["Table 2.1", "Figure 2.7"],
        "expected_answer": (
            "Comparing Low-income vs High-income countries: (a) Table 2.1 reports an openness score (0–100) of 38 for "
            "low-income vs 66 for high-income; (b) Table 2.1 shows open terms of use/license at 11% for low-income vs "
            "44% for high-income; (c) Figure 2.7 shows 0% of low-income countries had a fully funded national statistical "
            "plan in 2019 compared with 93% of high-income countries."
        ),
        "required_facts": [
            "Openness score (Table 2.1): Low-income=38 vs High-income=66",
            "Open terms of use/license (Table 2.1): Low-income=11% vs High-income=44%",
            "Fully funded statistical plan (Figure 2.7): Low-income=0% vs High-income=93%",
        ],
        "verify_tokens": ["38", "66", "11", "44", "0%", "93%"],
    },
    {
        "id": "adv_d11",
        "split": "dev",
        "primary_group": "multihop",
        "answerable": True,
        "question": (
            "Combine Figure 1.4 and Table 6.1: list the four data governance layers shown in Figure 1.4, and "
            "state the Global average regulatory scores (0–100) in Table 6.1 for all Safeguards vs Enablers categories."
        ),
        "gold_chunk_ids": ["wdr2021:p59:vis_figure_1_4", "wdr2021:p227:vis_table_6_1"],
        "gold_pages": [59, 227],
        "gold_elements": ["Figure 1.4", "Table 6.1"],
        "expected_answer": (
            "Figure 1.4 identifies four data governance layers at national and international levels: "
            "Infrastructure policies, Laws and regulations, Economic policies, and Institutions. "
            "In Table 6.1 (Laws and regulations, Global average scores 0–100): Safeguards are Cybersecurity and cybercrime = 56, "
            "Personal data = 44, and Nonpersonal data = 38; Enablers are E-commerce and e-transactions = 73, "
            "Public intent data = 50, and Private intent data = 17."
        ),
        "required_facts": [
            "Figure 1.4 layers: Infrastructure policies, Laws and regulations, Economic policies, Institutions",
            "Table 6.1 Global Safeguards: Cybersecurity=56, Personal data=44, Nonpersonal data=38",
            "Table 6.1 Global Enablers: E-commerce=73, Public intent data=50, Private intent data=17",
        ],
        "verify_tokens": ["Infrastructure", "Economic", "Institutions", "56", "44", "38", "73", "50", "17"],
    },
    {
        "id": "adv_d12",
        "split": "dev",
        "primary_group": "multihop",
        "answerable": True,
        "question": (
            "Combining Figure 7.5 (page 259) and Table 7.1 (page 261), what are the three broad regulatory models "
            "for cross-border personal data transfers, how do they sit on the openness spectrum, and what is each "
            "model's policy goal and regulatory cost level?"
        ),
        "gold_chunk_ids": ["wdr2021:p259:vis_figure_7_5", "wdr2021:p261:vis_table_7_1"],
        "gold_pages": [259, 261],
        "gold_elements": ["Figure 7.5", "Table 7.1"],
        "expected_answer": (
            "Across Figure 7.5 and Table 7.1, cross-border data flow models span a spectrum from data closure to data openness: "
            "(1) Limited transfers model: policy goal = Cybersecurity and other security concerns, regulatory costs = Higher, "
            "digital trade flows = Limited by transfer approval or data localization requirements; "
            "(2) Conditional transfers model: policy goal = Protection of personal data, regulatory costs = Medium, "
            "flows = Subject to regulatory conditions; "
            "(3) Open transfers model: policy goal = Business freedom, regulatory costs = Lower, flows = Largely open."
        ),
        "required_facts": [
            "Limited transfers: Cybersecurity/security concerns, Higher cost, transfer approval/data localization",
            "Conditional transfers: Protection of personal data, Medium cost, subject to regulatory conditions",
            "Open transfers: Business freedom, Lower cost, largely open",
        ],
        "verify_tokens": ["Limited transfers", "Conditional transfers", "Open transfers", "Business freedom", "Data openness"],
    },
    {
        "id": "adv_d13",
        "split": "dev",
        "primary_group": "text",
        "answerable": True,
        "question": (
            "According to Chapter 3 (page 115), what three factors explain the gap between having network access "
            "and actually being online, and what specific agricultural use of IoT devices is described there?"
        ),
        "gold_chunk_ids": ["wdr2021:p115:p0", "wdr2021:p115:p1"],
        "gold_pages": [115],
        "gold_elements": ["Text p.115"],
        "expected_answer": (
            "On page 115, the gap between the number of people with network access and those actually online is described "
            "as a function of three factors: affordability, the existence of local content, and digital skills. "
            "In agriculture, platforms matching buyers and sellers of agricultural commodities use Internet of Things (IoT) "
            "devices to accurately measure variables such as commodity weight."
        ),
        "required_facts": [
            "3 factors: affordability, local content, digital skills",
            "Agricultural IoT use: accurately measure variables such as commodity weight",
        ],
        "verify_tokens": ["affordability", "local content", "digital skills", "commodity weight"],
    },
    {
        "id": "adv_d14",
        "split": "dev",
        "primary_group": "text",
        "answerable": True,
        "question": (
            "On page 224, what UK legislation grants researchers access to deidentified data, which three cities "
            "attempted subnational public-private data sharing mandates, and how does country adoption of private-intent "
            "reuse enablers compare with public-sector enablers?"
        ),
        "gold_chunk_ids": ["wdr2021:p224:p0", "wdr2021:p224:p1"],
        "gold_pages": [224],
        "gold_elements": ["Text p.224"],
        "expected_answer": (
            "Page 224 cites the UK Digital Economy Act (2017) enabling researchers to gain access to deidentified data; "
            "notes that New York, San Francisco, and São Paulo made subnational legal attempts requiring private platforms "
            "to share commercial data for regulatory and smart-city purposes; and states that surveyed countries adopted "
            "on average less than 20 percent of good practices for private sector reuse of data — less than half the uptake "
            "of enablers for public sector data."
        ),
        "required_facts": [
            "UK Digital Economy Act (2017)",
            "Cities: New York, San Francisco, São Paulo",
            "<20% average adoption of private-data reuse enablers (less than half public-sector level)",
        ],
        "verify_tokens": ["Digital Economy Act (2017)", "New York", "San Francisco", "less than 20 percent", "less than half"],
    },
    {
        "id": "adv_d15",
        "split": "dev",
        "primary_group": "text",
        "answerable": True,
        "question": (
            "In Chapter 7 (page 265), why are Convention 108, the Budapest Convention on Cybercrime, and UNCITRAL "
            "model laws highlighted as alternatives or complements for regulatory harmonization?"
        ),
        "gold_chunk_ids": ["wdr2021:p265:p0", "wdr2021:p265:p1"],
        "gold_pages": [265],
        "gold_elements": ["Text p.265"],
        "expected_answer": (
            "Page 265 explains that unlike WTO rules or PTAs, cooperative instruments such as the Council of Europe's "
            "modernized Convention 108 (lacking binding nature) and the Budapest Convention on Cybercrime (lacking enforcement "
            "mechanisms) offer a less demanding channel for countries to adopt key data governance principles. "
            "Additionally, model laws such as those by UNCITRAL on electronic signatures provide softer approaches that "
            "encourage and facilitate regulatory harmonization."
        ),
        "required_facts": [
            "Convention 108 lacks binding nature; Budapest Cybercrime Convention lacks enforcement mechanisms -> less demanding channel",
            "UNCITRAL model laws on electronic signatures offer softer approaches facilitating regulatory harmonization",
        ],
        "verify_tokens": ["Convention 108", "Budapest Convention", "UNCITRAL", "electronic signatures"],
    },
    {
        "id": "adv_d16",
        "split": "dev",
        "primary_group": "unanswerable",
        "answerable": False,
        "question": (
            "What administrative fine amounts (in euros and global turnover percentage) does Chapter 6 of WDR 2021 "
            "report for noncompliance under EU AI Act Regulation 2024/1689?"
        ),
        "gold_chunk_ids": [],
        "gold_pages": [],
        "gold_elements": [],
        "expected_answer": (
            "Cannot answer from WDR 2021 evidence: EU AI Act Regulation 2024/1689 and its fine tiers are post-2021 "
            "external facts not present in the document."
        ),
        "required_facts": ["Explicit abstention citing absence of EU AI Act Regulation 2024/1689 in WDR 2021"],
        "abstention_reason": "Post-2021 EU AI Act Regulation 2024/1689 is absent from WDR 2021.",
        "forbidden_terms": ["2024/1689", "EU AI Act"],
    },
    {
        "id": "adv_d17",
        "split": "dev",
        "primary_group": "unanswerable",
        "answerable": False,
        "question": (
            "According to Table 5.1, how many sovereign AI GPU clusters are targeted to be built in Hanoi and "
            "Ho Chi Minh City during the 'Optimizing the system' stage?"
        ),
        "gold_chunk_ids": [],
        "gold_pages": [],
        "gold_elements": [],
        "expected_answer": (
            "Cannot answer from WDR 2021 evidence: Table 5.1 provides generic maturity-stage infrastructure "
            "recommendations and contains no sovereign AI GPU cluster targets for Hanoi or Ho Chi Minh City."
        ),
        "required_facts": ["Explicit abstention citing absence of GPU cluster counts for Hanoi/Ho Chi Minh City in Table 5.1"],
        "abstention_reason": "Fabricated city-specific GPU cluster targets not present in Table 5.1.",
        "forbidden_terms": ["Ho Chi Minh", "GPU cluster"],
    },
    {
        "id": "adv_d18",
        "split": "dev",
        "primary_group": "unanswerable",
        "answerable": False,
        "question": (
            "In Figure 7.6, how many jurisdictions had formally ratified the 2023 OECD BEPS Pillar One "
            "Multilateral Convention?"
        ),
        "gold_chunk_ids": [],
        "gold_pages": [],
        "gold_elements": [],
        "expected_answer": (
            "Cannot answer from WDR 2021 evidence: Figure 7.6 shows indirect VAT revenue potential for six East Asian "
            "countries (2015–2025) and contains no 2023 OECD BEPS Pillar One ratification count."
        ),
        "required_facts": ["Explicit abstention citing absence of 2023 Pillar One ratification count in WDR 2021"],
        "abstention_reason": "Post-2021 treaty ratification statistics absent from Figure 7.6 / WDR 2021.",
        "forbidden_terms": ["Multilateral Convention"],
    },

    # =========================================================================
    # TEST SPLIT (30 cases: 6 table, 5 diagram, 5 chart, 4 multihop, 4 text, 6 unanswerable)
    # =========================================================================
    {
        "id": "adv_t01",
        "split": "test",
        "primary_group": "table",
        "answerable": True,
        "question": (
            "According to Table 2.1, report the values across all four income groups (Low-income, Lower-middle-income, "
            "Upper-middle-income, High-income) for all five openness indicators, and calculate the percentage-point gap "
            "between High-income and Low-income countries for 'Available in machine readable format (%)'."
        ),
        "gold_chunk_ids": ["wdr2021:p83:vis_table_2_1"],
        "gold_pages": [83],
        "gold_elements": ["Table 2.1"],
        "expected_answer": (
            "In Table 2.1 (Low / Lower-middle / Upper-middle / High-income): "
            "Openness score (0–100) = 38 / 47 / 50 / 66; "
            "Available in machine readable format (%) = 37 / 51 / 61 / 81 (High minus Low gap = 81% - 37% = 44 percentage points); "
            "Available in nonproprietary format (%) = 75 / 85 / 81 / 84; "
            "Download options available (%) = 56 / 68 / 68 / 78; "
            "Open terms of use/license (%) = 11 / 19 / 22 / 44."
        ),
        "required_facts": [
            "Openness score: 38, 47, 50, 66",
            "Machine readable (%): 37, 51, 61, 81 and High-Low gap = 44 percentage points",
            "Nonproprietary (%): 75, 85, 81, 84",
            "Download options (%): 56, 68, 68, 78",
            "Open terms/license (%): 11, 19, 22, 44",
        ],
        "verify_tokens": ["38", "47", "50", "66", "37", "51", "61", "81", "75", "85", "84", "56", "68", "78", "11", "19", "22", "44"],
    },
    {
        "id": "adv_t02",
        "split": "test",
        "primary_group": "table",
        "answerable": True,
        "question": (
            "List all seven research areas in Table 4.1 ('Selected research gaps to be addressed to advance the use "
            "of private intent data for development'), and explain why footnote b uses the term 'deidentification' "
            "instead of 'anonymization'."
        ),
        "gold_chunk_ids": ["wdr2021:p162:vis_table_4_1"],
        "gold_pages": [162],
        "gold_elements": ["Table 4.1"],
        "expected_answer": (
            "Table 4.1 lists seven research areas: (1) Societal impacts, (2) Quality standards, (3) External validity, "
            "(4) Machine learning, (5) Training and validation data, (6) Deidentification, and (7) Capturing longitudinal change. "
            "Footnote b explains that 'deidentification' is used instead of 'anonymization' because even after data are "
            "processed to deidentify individuals, they may become re-identifiable in the future as computing and machine "
            "learning advance — so data may never be truly anonymized."
        ),
        "required_facts": [
            "All 7 research areas: Societal impacts, Quality standards, External validity, Machine learning, Training and validation data, Deidentification, Capturing longitudinal change",
            "Footnote b: data may become identifiable in the future as computing and ML advance, so data may never be truly anonymized",
        ],
        "verify_tokens": ["Societal impacts", "Quality standards", "External validity", "Machine learning", "Training and validation data", "Deidentification", "Capturing longitudinal", "never be truly anonymized"],
    },
    {
        "id": "adv_t03",
        "split": "test",
        "primary_group": "table",
        "answerable": True,
        "question": (
            "According to Table 5.1, compare the sequenced data infrastructure recommendations for 'Connecting people' "
            "versus 'Connecting countries' across all three country maturity stages."
        ),
        "gold_chunk_ids": ["wdr2021:p197:vis_table_5_1"],
        "gold_pages": [197],
        "gold_elements": ["Table 5.1"],
        "expected_answer": (
            "Table 5.1 sequences recommendations across three stages: "
            "(1) Establishing fundamentals: Connecting people = eliminate coverage gaps by lowering costs via wholesale/retail "
            "competition, infrastructure sharing, and targeted state support; Connecting countries = ensure adequate international "
            "bandwidth and a competitive market for international gateways and ISPs. "
            "(2) Initiating data flows: Connecting people = narrow usage gaps via digital literacy campaigns, basic education, "
            "lower taxes/duties on low-end handsets, and local handset assembly support; Connecting countries = create the first "
            "domestic IXP with all domestic players, allow additional IXPs, and cache popular international content locally. "
            "(3) Optimizing the system: Connecting people = upgrade digital networks to the latest generation for higher speed, "
            "efficiency, and consumption; Connecting countries = support colocation data centers integrated with IXPs to provide "
            "cloud service on-ramps."
        ),
        "required_facts": [
            "Establishing fundamentals: eliminate coverage gaps / competition / state support vs international bandwidth & competitive gateways/ISPs",
            "Initiating data flows: digital literacy / handset taxes vs domestic IXP & local content caching",
            "Optimizing the system: upgrade to latest generation networks vs colocation data centers integrated with IXPs & cloud on-ramps",
        ],
        "verify_tokens": ["Establishing", "Initiating data", "Optimizing the", "international", "domestic IXP", "colocation", "cloud services"],
    },
    {
        "id": "adv_t04",
        "split": "test",
        "primary_group": "table",
        "answerable": True,
        "question": (
            "Using Table 6.1, report the average regulatory scores (0–100) for High-income, Low-income, and Global "
            "groups across all three Safeguards sub-indicators and all three Enablers sub-indicators."
        ),
        "gold_chunk_ids": ["wdr2021:p227:vis_table_6_1"],
        "gold_pages": [227],
        "gold_elements": ["Table 6.1"],
        "expected_answer": (
            "In Table 6.1 (scores 0–100): "
            "High-income: Safeguards = Cybersecurity and cybercrime 73, Personal data 59, Nonpersonal data 43; "
            "Enablers = E-commerce and e-transactions 86, Public intent data 69, Private intent data 30. "
            "Low-income: Safeguards = Cybersecurity 39, Personal data 31, Nonpersonal data 47; "
            "Enablers = E-commerce 59, Public intent data 28, Private intent data 3. "
            "Global: Safeguards = Cybersecurity 56, Personal data 44, Nonpersonal data 38; "
            "Enablers = E-commerce 73, Public intent data 50, Private intent data 17."
        ),
        "required_facts": [
            "High-income: 73, 59, 43 | 86, 69, 30",
            "Low-income: 39, 31, 47 | 59, 28, 3",
            "Global: 56, 44, 38 | 73, 50, 17",
        ],
        "verify_tokens": ["73", "59", "43", "86", "69", "30", "39", "31", "47", "28", "56", "44", "38", "50", "17"],
    },
    {
        "id": "adv_t05",
        "split": "test",
        "primary_group": "table",
        "answerable": True,
        "question": (
            "Compare the three cross-border personal data models in Table 7.1 across every column: "
            "'Model', 'Policy goal', 'Regulatory costs', and 'Digital trade flows'."
        ),
        "gold_chunk_ids": ["wdr2021:p261:vis_table_7_1"],
        "gold_pages": [261],
        "gold_elements": ["Table 7.1"],
        "expected_answer": (
            "In Table 7.1: "
            "(1) Limited transfers: Policy goal = Cybersecurity and other security concerns; Regulatory costs = Higher; "
            "Digital trade flows = Limited by transfer approval or data localization requirements. "
            "(2) Conditional transfers: Policy goal = Protection of personal data; Regulatory costs = Medium; "
            "Digital trade flows = Subject to regulatory conditions. "
            "(3) Open transfers: Policy goal = Business freedom; Regulatory costs = Lower; "
            "Digital trade flows = Largely open."
        ),
        "required_facts": [
            "Limited transfers: Cybersecurity and other security concerns | Higher | Limited by transfer approval or data localization requirements",
            "Conditional transfers: Protection of personal data | Medium | Subject to regulatory conditions",
            "Open transfers: Business freedom | Lower | Largely open",
        ],
        "verify_tokens": ["Limited transfers", "Cybersecurity and other security concerns", "Conditional transfers", "Protection of personal data", "Open transfers", "Business freedom", "Largely open"],
    },
    {
        "id": "adv_t06",
        "split": "test",
        "primary_group": "table",
        "answerable": True,
        "question": (
            "In Table 8.3 ('Snapshot of common data intermediary structures'), list all five rows with their exact "
            "Purposes, Types, and concrete Examples."
        ),
        "gold_chunk_ids": ["wdr2021:p298:vis_table_8_3"],
        "gold_pages": [298],
        "gold_elements": ["Table 8.3"],
        "expected_answer": (
            "Table 8.3 covers three overarching Purposes across five structure types: "
            "(1) Create commercial value and data markets -> Data aggregators, data brokers -> Acxiom, Experian; "
            "(2) Exchange data to solve public problems through collaborative structures (objective: collaborative within sectors) -> "
            "Data pools -> Global Data Synchronization Network (GDSN); "
            "(3) Exchange data to solve public problems (create and manage shared/interoperable research data assets & computing infra) -> "
            "Data commons, data clubs -> The Open Commons Consortium (OCC), UK Biobank; "
            "(4) Exchange data to solve public problems (contribute data for collective benefits) -> Data cooperatives -> "
            "LunaDNA (community-owned platform for health research); "
            "(5) Enable trusted sharing and use of sensitive data through enhanced accountability mechanisms -> Data trusts -> "
            "Platform Info Exchange, UK (Mozilla Data Futures Lab pilot)."
        ),
        "required_facts": [
            "Data aggregators, data brokers: Acxiom, Experian",
            "Data pools: GDSN (Global Data Synchronization Network)",
            "Data commons, data clubs: Open Commons Consortium (OCC), UK Biobank",
            "Data cooperatives: LunaDNA",
            "Data trusts: Platform Info Exchange, UK (Mozilla Data Futures Lab pilot)",
        ],
        "verify_tokens": ["Acxiom, Experian", "GDSN", "UK Biobank", "LunaDNA", "Mozilla Data Futures", "Data trusts"],
    },
    {
        "id": "adv_t07",
        "split": "test",
        "primary_group": "diagram",
        "answerable": True,
        "question": (
            "According to the theory of change diagram in Figure O.1 (PDF page 24), which three main channels generate/use "
            "data, what positive outcomes vs negative risks appear on the right-hand side, and what do the two-way 'Reuse' arrows represent?"
        ),
        "gold_chunk_ids": ["wdr2021:p24:vis_figure_o_1"],
        "gold_pages": [24],
        "gold_elements": ["Figure O.1"],
        "expected_answer": (
            "Figure O.1 traces data produced/collected via Individuals (civil society & academia), More data on individuals "
            "(Government & International organizations), and Production process in firms (Private sector) through data analytics/processing "
            "and reuse toward development. Positive outcomes (green) are: Greater transparency, Greater accountability, Better policy making "
            "and service delivery, and Increased business opportunities. Negative risks (red) are: Criminal activity/dark net, "
            "Political surveillance, Market concentration, Widening inequality, and Discrimination. "
            "The two-way 'Reuse' arrows represent reusing and repurposing data collected for commercial purposes for public policy, and vice versa."
        ),
        "required_facts": [
            "Three channels: Individuals / Government & International organizations / Private sector",
            "Positive (green): Greater transparency, Greater accountability, Better policy making & service delivery, Increased business opportunities",
            "Negative (red): Criminal activity/dark net, Political surveillance, Market concentration, Widening inequality, Discrimination",
            "Two-way Reuse arrows: bidirectional commercial <-> public policy reuse and repurposing",
        ],
        "verify_tokens": ["Greater transparency", "Better policy making", "dark net", "Political surveillance", "Market concentration", "Discrimination", "Reuse"],
    },
    {
        "id": "adv_t08",
        "split": "test",
        "primary_group": "diagram",
        "answerable": True,
        "question": (
            "In Figure O.6 ('What happens in an integrated national data system?'), what are the five actions on data, "
            "five participant groups, four policy pillars, and foundational elements sustaining the system?"
        ),
        "gold_chunk_ids": ["wdr2021:p37:vis_figure_o_6"],
        "gold_pages": [37],
        "gold_elements": ["Figure O.6"],
        "expected_answer": (
            "Figure O.6 specifies: (1) Five actions — data are produced, protected, open, quality controlled, and used and reused; "
            "(2) Five participant groups — government entities, civil society and individuals, academic institutions, the private sector, "
            "and international organizations; (3) Four pillars — infrastructure policies, laws and regulations, economic policies, and institutions; "
            "(4) Foundation elements — human capital, data demand, trust, incentives, and funding."
        ),
        "required_facts": [
            "5 actions: produced, protected, open, quality controlled, used and reused",
            "5 participants: government entities, civil society and individuals, academic institutions, private sector, international organizations",
            "4 pillars: infrastructure policies, laws and regulations, economic policies, institutions",
            "Foundation: human capital, data demand, trust, incentives, funding",
        ],
        "verify_tokens": ["quality controlled", "civil society", "academic", "infrastructure", "human capital", "trust", "incentives"],
    },
    {
        "id": "adv_t09",
        "split": "test",
        "primary_group": "diagram",
        "answerable": True,
        "question": (
            "According to Figure 3.1 ('The role of data in the production process: Pathways to development'), what two "
            "primary pathways lead from data in the production process, and what intermediate and final development outcomes result?"
        ),
        "gold_chunk_ids": ["wdr2021:p114:vis_figure_3_1"],
        "gold_pages": [114],
        "gold_elements": ["Figure 3.1"],
        "expected_answer": (
            "Figure 3.1 shows two primary pathways: (1) Better data-driven decision-making, leading to quality improvements, "
            "cost reduction and process optimization, and product innovation; and (2) Data-driven matching of demand and supply "
            "and transaction facilitation, leading to data-driven intermediation and lower transaction costs. "
            "Together these yield development outcomes: boosting opportunities for lagging regions and populations, creating new "
            "export opportunities, promoting productivity and growth, creating new markets and disrupting incumbents' market power, "
            "and supporting local entrepreneurship ecosystems (with related opportunities including access to 'free' services, "
            "COVID-19 adaptation, and support for the green agenda)."
        ),
        "required_facts": [
            "Pathway 1: Better data-driven decision-making -> quality improvements, cost reduction & process optimization, product innovation",
            "Pathway 2: Data-driven matching of demand and supply & transaction facilitation -> intermediation & lower transaction costs",
            "Outcomes: lagging regions, new exports, productivity & growth, new markets/disrupt incumbents, local entrepreneurship",
        ],
        "verify_tokens": ["Better data-driven", "matching of demand", "Product", "lower", "lagging regions", "incumbents"],
    },
    {
        "id": "adv_t10",
        "split": "test",
        "primary_group": "diagram",
        "answerable": True,
        "question": (
            "Based on the architecture diagram in Figure 5.1 ('The data infrastructure supply chain'), describe the three "
            "segments, their distance scales, their roles, and key facilities/technologies shown in each segment."
        ),
        "gold_chunk_ids": ["wdr2021:p178:vis_figure_5_1"],
        "gold_pages": [178],
        "gold_elements": ["Figure 5.1"],
        "expected_answer": (
            "Figure 5.1 breaks the data infrastructure supply chain into three segments: "
            "(1) Cross-border network (international links, >1,000 km): connects service centers to the worldwide internet "
            "using subsea cables, terrestrial fiber, satellite links/earth stations, IXPs, and colocation data centers; "
            "(2) Middle-mile network (distribution, 10–1,000 km): connects bigger cities and routes data to service centers "
            "using fiber-optic backbones, fiber or microwave backbones, satellite links, IXPs, and colocation data centers; "
            "(3) Last mile (access, <1–5 km): distributes internet locally to homes and businesses via cellular towers, "
            "fiber, DSL, coaxial, wireless (Wi-Fi, LTE, TV white space), satellite DTH, and balloons/drones."
        ),
        "required_facts": [
            "Cross-border network (international links): >1,000 km (subsea cable, earth station, terrestrial fiber, IXP, colocation data center)",
            "Middle-mile network (distribution): 10-1,000 km (fiber-optic/microwave backbone, IXP, colocation data center)",
            "Last mile (access): <1-5 km (cellular towers, fiber, DSL, coaxial, Wi-Fi/LTE/TV white space, satellite DTH, balloons and drones)",
        ],
        "verify_tokens": ["Cross-border network", ">1,000 km", "Middle-mile network", "10–1,000 km", "Last mile", "<1–5 km", "Subsea cable", "TV white space"],
    },
    {
        "id": "adv_t11",
        "split": "test",
        "primary_group": "diagram",
        "answerable": True,
        "question": (
            "In Figure 9.3 ('Steps to integrating the public sector into the national data system'), what concrete steps "
            "are grouped under (1) Establishing fundamentals, (2) Initiating data flows, and (3) Optimizing the system?"
        ),
        "gold_chunk_ids": ["wdr2021:p330:vis_figure_9_3"],
        "gold_pages": [330],
        "gold_elements": ["Figure 9.3"],
        "expected_answer": (
            "Figure 9.3 organizes public sector integration steps into three maturity stages: "
            "(1) Establishing fundamentals: Policies (recognize an integrated national data system in a high-level strategic document); "
            "Laws and regulations (robust data protection regulations); Human capital and funding (strengthen NSO technical capacity "
            "and financing); Institutions (create technical units in ministries for administrative data coordination); "
            "Incentives (appropriately remunerate public-sector data scientists and statisticians). "
            "(2) Initiating data flows: Data demand (establish a culture of data use in ministries and among policy makers/legislators); "
            "Incentives (prioritize open data and common standards throughout the data life cycle); Infrastructure policies "
            "(establish a secure integrated digital platform to store and provide access to deidentified public intent data). "
            "(3) Optimizing the system: Institutions (charge a government unit with overseeing and reporting on strategy implementation); "
            "Incentives (empower the NSO to play an active role); "
            "Policies (define clear institutional mandates across government institutions)."
        ),
        "required_facts": [
            "Establishing fundamentals: strategic document, data protection regulations, NSO capacity & financing, ministry technical units, remunerate data scientists/statisticians",
            "Initiating data flows: culture of data use, open data & common standards, secure digital platform for deidentified public intent data",
            "Optimizing the system: oversight unit, empower NSO, clear institutional mandates",
        ],
        "verify_tokens": ["high-level strategic document", "robust data protection", "culture of data use", "digital platform", "overseeing", "remunerated"],
    },
    {
        "id": "adv_t12",
        "split": "test",
        "primary_group": "chart",
        "answerable": True,
        "question": (
            "Using Figure 3.5 (panels a and b), what share of Low-income, High-income, and Middle-income countries "
            "have Facebook, Google, and YouTube among the top 10 visited websites, and what is the global average daily "
            "time per user spent on Facebook.com, Google.com, Youtube.com, and Others?"
        ),
        "gold_chunk_ids": ["wdr2021:p118:vis_figure_3_5"],
        "gold_pages": [118],
        "gold_elements": ["Figure 3.5"],
        "expected_answer": (
            "In Figure 3.5: Panel a shows Facebook, Google, and YouTube appear in the top 10 visited websites in "
            "100% of low-income countries, 94% of high-income countries, and 78% of middle-income countries. "
            "Panel b shows global average daily time per user is 17.42 minutes on Facebook.com, 13.12 minutes on Google.com, "
            "13.04 minutes on Youtube.com, and 7.01 minutes on Others (average)."
        ),
        "required_facts": [
            "Panel a: Low-income=100%, High-income=94%, Middle-income=78%",
            "Panel b: Facebook.com=17.42 min, Google.com=13.12 min, Youtube.com=13.04 min, Others=7.01 min",
        ],
        "verify_tokens": ["100%", "94%", "78%", "17.42", "13.12", "13.04", "7.01"],
    },
    {
        "id": "adv_t13",
        "split": "test",
        "primary_group": "chart",
        "answerable": True,
        "question": (
            "According to the AI Talent Tracker charts in Figure 4.2, compare the shares of the United States, China, "
            "and Europe in Panel a ('Location of employment') versus Panel b ('Location of undergraduate study'), "
            "and note India's undergraduate share in Panel b."
        ),
        "gold_chunk_ids": ["wdr2021:p157:vis_figure_4_2"],
        "gold_pages": [157],
        "gold_elements": ["Figure 4.2"],
        "expected_answer": (
            "In Figure 4.2: in Panel a (location of employment), the United States accounts for 59%, China 11%, and Europe 10% "
            "(alongside Others 10%, Canada 6%, UK 4%). In Panel b (location of undergraduate study), China accounts for 29%, "
            "the United States 20%, Europe 18%, and India 8% (Others 10%, Canada 5%, UK 4%, Iran 3%, Israel 3%), "
            "highlighting a large net inflow of AI specialists into the United States."
        ),
        "required_facts": [
            "Panel a employment: US=59%, China=11%, Europe=10%",
            "Panel b undergrad study: China=29%, US=20%, Europe=18%, India=8%",
        ],
        "verify_tokens": ["59%", "11%", "10%", "29%", "20%", "18%", "India=8%"],
    },
    {
        "id": "adv_t14",
        "split": "test",
        "primary_group": "chart",
        "answerable": True,
        "question": (
            "From Figure 5.3 (panels a and b), how did global Connected, Usage gap, and 3G Coverage gap shares shift "
            "between 2015 and 2018, and which regions represent the 2018 extremes in coverage gap and usage gap?"
        ),
        "gold_chunk_ids": ["wdr2021:p181:vis_figure_5_3"],
        "gold_pages": [181],
        "gold_elements": ["Figure 5.3"],
        "expected_answer": (
            "In Figure 5.3 Panel a, globally from 2015 to 2018: Connected population rose from 41% (2015) -> 45% (2016) -> "
            "48% (2017) -> 50% (2018); the 3G coverage gap shrank from 21% (2015) -> 15% (2016) -> 11% (2017) -> 8% (2018); "
            "and the usage gap increased from 37% (2015) -> 40% (2016) -> 41% (2017) -> 42% (2018). "
            "In Panel b (2018 regional extremes): North America had a <1% coverage gap versus 29% in Sub-Saharan Africa, "
            "while South Asia had the highest usage gap at 64% (with 28% connected and 8% coverage gap)."
        ),
        "required_facts": [
            "2015: Connected=41%, Usage gap=37%, Coverage gap=21%",
            "2018: Connected=50%, Usage gap=42%, Coverage gap=8%",
            "2018 regional extremes: North America <1% vs Sub-Saharan Africa 29% coverage gap; South Asia 64% usage gap",
        ],
        "verify_tokens": ["41%", "37%", "21%", "50%", "42%", "North America", "29%", "64%"],
    },
    {
        "id": "adv_t15",
        "split": "test",
        "primary_group": "chart",
        "answerable": True,
        "question": (
            "According to Figure 5.6 (2018, 119 economies), report mobile data consumption (GB per capita per month) across "
            "all four country income groups and the World average (Panel a), and contrast the highest and lowest regions (Panel b)."
        ),
        "gold_chunk_ids": ["wdr2021:p186:vis_figure_5_6"],
        "gold_pages": [186],
        "gold_elements": ["Figure 5.6"],
        "expected_answer": (
            "In Figure 5.6 (2018, GB per capita per month): Panel a shows High-income = 7.1, Upper-middle-income = 2.8, "
            "Lower-middle-income = 1.3, Low-income = 0.2, and World = 3.7 GB. "
            "In Panel b across regions, Middle East and North Africa is highest at 7.4 GB (followed by Europe & Central Asia 5.0, "
            "North America 4.5, East Asia & Pacific 4.2, Latin America & Caribbean 2.4, South Asia 1.1) and Sub-Saharan Africa "
            "is lowest at 0.3 GB."
        ),
        "required_facts": [
            "Panel a: High=7.1, Upper-mid=2.8, Lower-mid=1.3, Low=0.2, World=3.7 GB/capita/month",
            "Panel b highest vs lowest: Middle East and North Africa=7.4 vs Sub-Saharan Africa=0.3 GB",
        ],
        "verify_tokens": ["7.1", "2.8", "1.3", "0.2", "3.7", "7.4", "0.3"],
    },
    {
        "id": "adv_t16",
        "split": "test",
        "primary_group": "chart",
        "answerable": True,
        "question": (
            "In Figure 7.2 (finalized digital platform antitrust cases as of January 2020), compare the total number of cases "
            "and top sectoral shares (%) between High-income countries (Panel a) and Middle-income countries (Panel b)."
        ),
        "gold_chunk_ids": ["wdr2021:p250:vis_figure_7_2"],
        "gold_pages": [250],
        "gold_elements": ["Figure 7.2"],
        "expected_answer": (
            "In Figure 7.2: Panel a covers 62 finalized cases in high-income countries, led by Retail e-commerce (21%), "
            "followed by Online search and advertising (15%), Software and operating systems (15%), Others (15%), Tourism (13%), "
            "Social media and communications (10%), Passenger transport (8%), and Online real estate platforms (5%). "
            "Panel b covers 40 cases in middle-income countries, led by Passenger transport (35%), Retail e-commerce (23%), "
            "Software and operating systems (18%), Others (10%), Mobile financial services (8%), and Online search and advertising (8%)."
        ),
        "required_facts": [
            "High-income (62 cases): Retail e-commerce 21%, Online search & ads 15%, Software/OS 15%, Passenger transport 8%",
            "Middle-income (40 cases): Passenger transport 35%, Retail e-commerce 23%, Software/OS 18%, Mobile financial 8%",
        ],
        "verify_tokens": ["62 cases", "40 cases", "21%", "35%", "23%", "18%"],
    },
    {
        "id": "adv_t17",
        "split": "test",
        "primary_group": "multihop",
        "answerable": True,
        "question": (
            "Combining Figure 8.3 (page 293) and Figures 8.4 and 8.5 (page 296), compare Low-income vs High-income countries on: "
            "(1) established dedicated data governance entities and separate-agency share, "
            "(2) established data protection authorities, and (3) operational antitrust authorities."
        ),
        "gold_chunk_ids": ["wdr2021:p293:vis_figure_8_3", "wdr2021:p296:vis_figure_8_4"],
        "gold_pages": [293, 296],
        "gold_elements": ["Figure 8.3", "Figure 8.4", "Figure 8.5"],
        "expected_answer": (
            "Combining page 293 and page 296: "
            "(1) Dedicated data governance entity (Figure 8.3 Panel a, 198 economies): Low-income = 0% established (0% in progress) "
            "vs High-income = 53% established (plus 8% in progress), and in Panel b (58 countries), 56% of High-income entities are "
            "separate agencies (vs 0% in Lower-middle-income); "
            "(2) Data protection authority (Figure 8.4, 198 economies): Low-income = 3% established (24% in progress/planned) vs "
            "High-income = 81% established (5% in progress/planned); "
            "(3) Antitrust authority (Figure 8.5, 218 economies): Low-income = 52% operational (17% planned) vs "
            "High-income = 75% operational (4% planned)."
        ),
        "required_facts": [
            "Figure 8.3: Low=0% vs High=53% established data governance entity (56% separate agency in High)",
            "Figure 8.4: Low=3% vs High=81% established data protection authority",
            "Figure 8.5: Low=52% vs High=75% operational antitrust authority",
        ],
        "verify_tokens": ["53%", "56%", "81%", "75%", "52%"],
    },
    {
        "id": "adv_t18",
        "split": "test",
        "primary_group": "multihop",
        "answerable": True,
        "question": (
            "Combining Figure 8.2 (page 290) and Table 8.2 (page 292), identify all four thematic clusters of data "
            "governance functions, the specific bulleted sub-functions under each cluster, and the indicative actors/institutions "
            "assigned to each cluster."
        ),
        "gold_chunk_ids": ["wdr2021:p290:vis_figure_8_2", "wdr2021:p292:vis_table_8_2"],
        "gold_pages": [290, 292],
        "gold_elements": ["Figure 8.2", "Table 8.2"],
        "expected_answer": (
            "Across Figure 8.2 and Table 8.2, the four thematic governance clusters are: "
            "(1) Strategic planning — functions: develop strategies and establish institutional arrangements; actors: centralized "
            "data governance agency/unit (e.g., NSO, digital economy ministry) or decentralized ministry units, plus CSOs, universities, research institutions; "
            "(2) Rule making and implementation — functions: legislate/regulate, set standards, provide clarification and guidance; "
            "actors: national legislature, sector-specific regulators (telecom, banking/securities), industry associations, CSOs, IRBs, SSOs, international organizations (World Bank, IMF, UN, WTO); "
            "(3) Compliance — functions: enforce, audit, arbitrate, remedy; actors: watchdogs/umpires (data protection authority, ATI agency, antitrust authority, consumer protection agency, audit body, courts, ombudsperson, CSIRT); "
            "(4) Learning and evidence — functions: monitor and evaluate, anticipate and manage risks; actors: knowledge community (M&E units/bodies, CSOs/NGOs, MDBs, academia, think tanks, news media, training bodies, professional associations)."
        ),
        "required_facts": [
            "Strategic planning: develop strategies, institutional arrangements -> centralized/decentralized governance units, CSOs, academia",
            "Rule making: legislate/regulate, set standards, clarification/guidance -> legislature, sector regulators, SSOs, international orgs",
            "Compliance: enforce, audit, arbitrate, remedy -> DPA, ATI, antitrust, audit body, courts, ombudsperson, CSIRT",
            "Learning and evidence: monitor & evaluate, anticipate & manage risks -> M&E units, academia, think tanks, news media, MDBs",
        ],
        "verify_tokens": ["Strategic planning", "Rule making", "Compliance", "Learning and evidence", "CSIRT", "Ombudsperson", "Industry"],
    },
    {
        "id": "adv_t19",
        "split": "test",
        "primary_group": "multihop",
        "answerable": True,
        "question": (
            "Combine the empirical data gap shown in Figure 5.6 (page 186) with the sequenced policies in Table 5.1 (page 197): "
            "how big is the 2018 mobile data consumption gap between Low-income / Sub-Saharan Africa and High-income economies, "
            "and what policies should countries in the 'Establishing fundamentals' and 'Initiating data flows' stages implement "
            "to close both coverage and usage/cost gaps?"
        ),
        "gold_chunk_ids": ["wdr2021:p186:vis_figure_5_6", "wdr2021:p197:vis_table_5_1"],
        "gold_pages": [186, 197],
        "gold_elements": ["Figure 5.6", "Table 5.1"],
        "expected_answer": (
            "In Figure 5.6 (2018), monthly mobile data consumption is 0.2 GB per capita in Low-income economies (and 0.3 GB in "
            "Sub-Saharan Africa) versus 7.1 GB in High-income economies (and 3.7 GB World average) — over a 35x income-group disparity "
            "(and >23x vs Sub-Saharan Africa). To close these gaps, Table 5.1 prescribes: "
            "at 'Establishing fundamentals', eliminating coverage gaps via wholesale/retail competition, infrastructure sharing, "
            "targeted state support, adequate international bandwidth, and competitive international gateways/ISPs; and at "
            "'Initiating data flows', narrowing usage gaps via digital literacy campaigns, basic education investment, lower handset "
            "taxes/duties, domestic IXP creation, and local caching of popular internet content."
        ),
        "required_facts": [
            "Figure 5.6: Low-income=0.2 GB (SSA=0.3 GB) vs High-income=7.1 GB (World=3.7 GB)",
            "Table 5.1 Establishing fundamentals: competition, infrastructure sharing, state support, international bandwidth/gateways",
            "Table 5.1 Initiating data flows: digital literacy, lower handset taxes/duties, domestic IXP, local content caching",
        ],
        "verify_tokens": ["7.1", "0.2", "0.3", "Establishing", "Initiating data", "domestic IXP", "digital literacy"],
    },
    {
        "id": "adv_t20",
        "split": "test",
        "primary_group": "multihop",
        "answerable": True,
        "question": (
            "Combining Table 6.1 (page 227) and Table 7.1 (page 261), contrast the Global average regulatory score for "
            "Personal data safeguards vs E-commerce enablers, and explain how the Limited, Conditional, and Open transfers "
            "models trade off personal data protection, security goals, and regulatory costs."
        ),
        "gold_chunk_ids": ["wdr2021:p227:vis_table_6_1", "wdr2021:p261:vis_table_7_1"],
        "gold_pages": [227, 261],
        "gold_elements": ["Table 6.1", "Table 7.1"],
        "expected_answer": (
            "In Table 6.1, the Global average score for Personal data safeguards is 44 (evolving level) compared with 73 for "
            "E-commerce and e-transactions enablers (moderate level; while Private intent data enablers lag at 17). "
            "Because stronger data-flow safeguards raise compliance complexity, Table 7.1 contrasts three cross-border personal data models: "
            "Limited transfers prioritizes cybersecurity and security concerns with higher regulatory costs and transfer approval/"
            "data localization restrictions; Conditional transfers prioritizes personal data protection at medium regulatory costs "
            "subject to destination safeguards/conditions; and Open transfers prioritizes business freedom with lower regulatory costs "
            "and largely open flows (with few safeguards)."
        ),
        "required_facts": [
            "Table 6.1 Global Personal data safeguard=44 vs E-commerce enabler=73 (Private intent=17)",
            "Table 7.1 Limited (cybersecurity, Higher cost, localization/approval), Conditional (personal data protection, Medium cost), Open (business freedom, Lower cost)",
        ],
        "verify_tokens": ["44", "73", "17", "Limited transfers", "Conditional transfers", "Open transfers", "Medium", "Business freedom"],
    },
    {
        "id": "adv_t21",
        "split": "test",
        "primary_group": "text",
        "answerable": True,
        "question": (
            "On PDF page 120, what did the consumer behavior study reveal about default likelihood among customers "
            "arriving from a price comparison website versus customers directed via search engine ads, and what other "
            "types of non-attributable data are cited in that paragraph?"
        ),
        "gold_chunk_ids": ["wdr2021:p120:p1"],
        "gold_pages": [120],
        "gold_elements": ["Text p.120"],
        "expected_answer": (
            "On page 120, consumers arriving from a price comparison website were roughly half as likely to default as "
            "customers directed via search engine ads, which is consistent with research on impulse shopping. The same chunk "
            "notes that firms also collect machine/device-generated data not attributable to a specific individual or business, "
            "including traffic, weather and climatic conditions, and network usage."
        ),
        "required_facts": [
            "Price comparison website users roughly half as likely to default as search-engine-ad customers (impulse shopping)",
            "Non-attributable machine/device data: traffic, weather and climatic conditions, network usage",
        ],
        "verify_tokens": ["price comparison website", "half as likely to default", "search engine ads", "impulse shopping", "weather and climatic"],
    },
    {
        "id": "adv_t22",
        "split": "test",
        "primary_group": "text",
        "answerable": True,
        "question": (
            "According to page 218, why can public sector adoption of privately developed automated decision-making systems "
            "hinder independent audits and trust, and what international ethical AI principles published in 2019 are highlighted?"
        ),
        "gold_chunk_ids": ["wdr2021:p218:p0", "wdr2021:p218:p1"],
        "gold_pages": [218],
        "gold_elements": ["Text p.218"],
        "expected_answer": (
            "On page 218, private-sector AI shows prevalent racial and gender bias, while public sector systems procured "
            "from private corporations may have underlying algorithms protected by copyright or other IPRs that restrict "
            "independent third-party audits — impairing public trust and risking discrimination without proper audits and "
            "grievance redress mechanisms. Internationally, the OECD and G-20 published two closely related ethical AI principle sets "
            "in 2019 emphasizing transparency, explainability, and inclusion of unrepresented or vulnerable groups."
        ),
        "required_facts": [
            "Copyright/IPRs on privately developed public-procured algorithms restrict third-party audits (plus racial/gender bias)",
            "OECD and G-20 ethical AI principles (2019): transparency, explainability, inclusion of unrepresented/vulnerable groups",
        ],
        "verify_tokens": ["copyright or other IPRs", "third-party audits", "OECD", "G-20", "2019", "explainability"],
    },
    {
        "id": "adv_t23",
        "split": "test",
        "primary_group": "text",
        "answerable": True,
        "question": (
            "Based on page 252, why do multisided digital platforms create unique competition concerns regarding "
            "'killer'/'zombie' acquisitions and bargaining power, and which three countries adopted complementary "
            "transaction-value merger notification thresholds?"
        ),
        "gold_chunk_ids": ["wdr2021:p252:p0", "wdr2021:p252:p2", "wdr2021:p252:p4"],
        "gold_pages": [252],
        "gold_elements": ["Text p.252"],
        "expected_answer": (
            "Page 252 explains that multisided platforms can act as buyers of services or de facto gig-economy employers "
            "(raising superior-bargaining-position concerns) and may engage in 'killer' or 'zombie' acquisitions — buying potential "
            "competitors in complementary markets before they become threats (or acquiring them primarily for data/algorithms) "
            "while those targets lack sufficient tangible assets or revenue to meet traditional merger notification thresholds. "
            "To capture such deals, Austria, Germany, and Japan adopted complementary merger thresholds based on transaction values."
        ),
        "required_facts": [
            "Platforms as buyers/de facto gig-economy employers (superior bargaining position) and killer/zombie acquisitions below revenue thresholds",
            "Countries with transaction-value merger thresholds: Austria, Germany, and Japan",
        ],
        "verify_tokens": ["gig economy", "killer", "zombie", "transaction values", "Austria", "Germany", "Japan"],
    },
    {
        "id": "adv_t24",
        "split": "test",
        "primary_group": "text",
        "answerable": True,
        "question": (
            "According to page 294, which three open data milestones were established by civil society/nongovernmental actors "
            "between 2012 and 2015, and what regional governance framework did ASEAN adopt in 2018?"
        ),
        "gold_chunk_ids": ["wdr2021:p294:p0", "wdr2021:p294:p1"],
        "gold_pages": [294],
        "gold_elements": ["Text p.294"],
        "expected_answer": (
            "Page 294 notes that nongovernmental actors established the Open Definition in 2012, standards for open data "
            "licensing in 2013, and the Open Data Charter in 2015. At the regional level, ASEAN adopted the ASEAN Framework "
            "on Digital Data Governance in 2018 to help member states strengthen and coordinate policy, regulatory, and "
            "institutional arrangements and build interoperable data governance systems."
        ),
        "required_facts": [
            "Open Definition (2012), standards for open data licensing (2013), Open Data Charter (2015)",
            "ASEAN Framework on Digital Data Governance (2018)",
        ],
        "verify_tokens": ["Open Definition in 2012", "2013", "Open Data Charter in 2015", "ASEAN Framework on Digital Data Governance", "2018"],
    },
    {
        "id": "adv_t25",
        "split": "test",
        "primary_group": "unanswerable",
        "answerable": False,
        "question": (
            "According to Chapter 4 of WDR 2021, what are the per-million-token API pricing rates and training compute "
            "in FLOPs reported for GPT-4 and Claude 3 Opus?"
        ),
        "gold_chunk_ids": [],
        "gold_pages": [],
        "gold_elements": [],
        "expected_answer": (
            "Cannot answer from WDR 2021 evidence: WDR 2021 was published in 2021 and contains no information, "
            "token pricing, or FLOPs for GPT-4 or Claude 3 Opus."
        ),
        "required_facts": ["Explicit abstention citing absence of GPT-4 and Claude 3 Opus in WDR 2021"],
        "abstention_reason": "Post-2021 LLM pricing and training compute specs do not exist in WDR 2021.",
        "forbidden_terms": ["GPT-4", "Claude 3", "per-million-token"],
    },
    {
        "id": "adv_t26",
        "split": "test",
        "primary_group": "unanswerable",
        "answerable": False,
        "question": (
            "Does WDR 2021 specify the total VND budget allocation mandated under Vietnam's National Data Center "
            "Resolution 175/NQ-CP and the 2024 Law on Data?"
        ),
        "gold_chunk_ids": [],
        "gold_pages": [],
        "gold_elements": [],
        "expected_answer": (
            "Cannot answer from WDR 2021 evidence: Vietnam's Resolution 175/NQ-CP and 2024 Law on Data budget "
            "allocations post-date WDR 2021 and are absent from the document."
        ),
        "required_facts": ["Explicit abstention citing absence of Resolution 175/NQ-CP and 2024 Law on Data in WDR 2021"],
        "abstention_reason": "Post-2021 Vietnamese legislative instruments absent from WDR 2021.",
        "forbidden_terms": ["175/NQ-CP", "Law on Data", "National Data Center"],
    },
    {
        "id": "adv_t27",
        "split": "test",
        "primary_group": "unanswerable",
        "answerable": False,
        "question": (
            "In Figure 5.3, what are the exact 2022–2024 ITU 5G population coverage percentages reported individually "
            "for Singapore, Malaysia, Indonesia, and Vietnam?"
        ),
        "gold_chunk_ids": [],
        "gold_pages": [],
        "gold_elements": [],
        "expected_answer": (
            "Cannot answer from WDR 2021 evidence: Figure 5.3 presents 2015–2018 global and regional 3G coverage/usage "
            "gaps and does not contain 2022–2024 country-level 5G coverage percentages."
        ),
        "required_facts": ["Explicit abstention citing that Figure 5.3 covers 2015-2018 3G gaps, not 2022-2024 country 5G percentages"],
        "abstention_reason": "2022-2024 ASEAN 5G coverage figures do not exist in Figure 5.3.",
        "forbidden_terms": ["2022–2024", "2024 ITU"],
    },
    {
        "id": "adv_t28",
        "split": "test",
        "primary_group": "unanswerable",
        "answerable": False,
        "question": (
            "Using Table 2.1, what are the individual country Open Data Inventory (ODIN) openness scores specifically "
            "for Thailand, Vietnam, and the Philippines?"
        ),
        "gold_chunk_ids": [],
        "gold_pages": [],
        "gold_elements": [],
        "expected_answer": (
            "Cannot answer from WDR 2021 evidence: Table 2.1 reports aggregate averages by country income group "
            "(Low, Lower-middle, Upper-middle, High-income) rather than individual country scores for Thailand, Vietnam, "
            "or the Philippines."
        ),
        "required_facts": ["Explicit abstention noting Table 2.1 reports income-group averages, not individual country ODIN scores"],
        "abstention_reason": "Table 2.1 contains income-group aggregates only, not per-country ODIN scores.",
        "forbidden_terms": ["Thailand", "Philippines"],
        "scoped_chunk": "wdr2021:p83:vis_table_2_1",
    },
    {
        "id": "adv_t29",
        "split": "test",
        "primary_group": "unanswerable",
        "answerable": False,
        "question": (
            "According to Table 8.3, what were the 2020 audited annual revenues and full-time employee headcounts "
            "of Acxiom, Experian, and LunaDNA?"
        ),
        "gold_chunk_ids": [],
        "gold_pages": [],
        "gold_elements": [],
        "expected_answer": (
            "Cannot answer from WDR 2021 evidence: Table 8.3 only lists Acxiom, Experian, and LunaDNA as illustrative "
            "intermediary examples and provides no annual revenues or employee headcounts."
        ),
        "required_facts": ["Explicit abstention noting Table 8.3 provides no revenue or employee headcounts for Acxiom/Experian/LunaDNA"],
        "abstention_reason": "Company financials and headcounts are not reported in Table 8.3 or WDR 2021.",
        "forbidden_terms": ["employee headcount", "audited annual revenue"],
    },
    {
        "id": "adv_t30",
        "split": "test",
        "primary_group": "unanswerable",
        "answerable": False,
        "question": (
            "In Figure 5.10, what is the exact annual cryptocurrency mining electricity consumption in terawatt-hours (TWh) "
            "reported per country?"
        ),
        "gold_chunk_ids": [],
        "gold_pages": [],
        "gold_elements": [],
        "expected_answer": (
            "Cannot answer from WDR 2021 evidence: Figure 5.10 discusses energy/cooling/carbon intensity of data centers "
            "and does not report annual cryptocurrency mining electricity consumption in TWh per country."
        ),
        "required_facts": ["Explicit abstention noting absence of cryptocurrency mining TWh figures in Figure 5.10"],
        "abstention_reason": "Cryptocurrency mining TWh statistics do not appear in WDR 2021.",
        "forbidden_terms": ["cryptocurrency mining", "terawatt-hours"],
    },
]


# =============================================================================
# TEST_V2 SPLIT (30 cases) — fresh, independent test set written on 2026-11-16,
# AFTER the original `test` split had been used to tune the pipeline
# (extract_diagram_xy_columns + arithmetic prompt). Per ADVANCED.md/RULES.md the
# original `test` is therefore no longer an independent evaluation; `test_v2` is
# run exactly once with the frozen pipeline (no code/prompt change afterwards).
# The report contains only 11 tables, so table cases reuse some tables but ask
# rows/columns/cells never asked in dev or in the original test.
# =============================================================================
BENCHMARK_CASES_V2: list[dict[str, Any]] = [
    # ---------------------------- table (6) ----------------------------------
    {
        "id": "adv_v01", "split": "test_v2", "primary_group": "table", "answerable": True,
        "question": (
            "In Table 2.2 (data dissemination practices by country income group), what percentages are reported "
            "for 'NSO has data portal' and 'NSO makes metadata available' in each income group, which income group "
            "has the highest data-portal share, and what is the high-income minus low-income gap for metadata?"
        ),
        "gold_chunk_ids": ["wdr2021:p84:vis_table_2_2"], "gold_pages": [84], "gold_elements": ["Table 2.2"],
        "expected_answer": (
            "Table 2.2 (2019, % of population whose NSO has the attribute): NSO has data portal = 84 (low), 91 "
            "(lower-middle), 95 (upper-middle), 92 (high) -> highest is upper-middle-income (95), not high-income. "
            "NSO makes metadata available = 63 / 91 / 97 / 100 -> high minus low = 37 percentage points."
        ),
        "required_facts": [
            "Data portal: 84 / 91 / 95 / 92 (low, lower-middle, upper-middle, high)",
            "Highest data-portal share is upper-middle-income (95)",
            "Metadata: 63 / 91 / 97 / 100; gap high-low = 37 percentage points",
        ],
        "verify_tokens": ["NSO has data portal | 84 | 91 | 95 | 92", "NSO makes metadata available | 63 | 91 | 97 | 100"],
    },
    {
        "id": "adv_v02", "split": "test_v2", "primary_group": "table", "answerable": True,
        "question": (
            "According to Table 8.1, which data management decisions are listed for the 'Create/receive', 'Process', "
            "'Analyze and use', and 'Archive and preserve' stages of the data life cycle?"
        ),
        "gold_chunk_ids": ["wdr2021:p288:vis_table_8_1"], "gold_pages": [288], "gold_elements": ["Table 8.1"],
        "expected_answer": (
            "Create/receive: determine lawful use (e.g., obtain consent for collection and sharing); collect "
            "identifications that allow merging with other datasets. Process: standardize units and categories; use "
            "widely compatible and accessible formats; validate quality (accuracy), relevance and integrity. Analyze and "
            "use: ensure reproducibility and publish code/algorithms; do not publish identifiable data; visualize and "
            "communicate insights. Archive and preserve: classify and catalog data systematically; include data "
            "dictionaries and notes on how data were created; maintain access, security and integrity over time."
        ),
        "required_facts": [
            "Create/receive: lawful use/consent; identifications for merging",
            "Process: standardize units/categories; compatible formats; validate quality/relevance/integrity",
            "Analyze and use: reproducibility/publish code; no identifiable data; visualize/communicate",
            "Archive and preserve: classify/catalog; data dictionaries/notes; maintain access, security, integrity",
        ],
        "verify_tokens": ["Determine lawful use", "Standardize units", "Ensure reproducibility",
                          "Do not publish identifiable data", "data dictionaries"],
    },
    {
        "id": "adv_v03", "split": "test_v2", "primary_group": "table", "answerable": True,
        "question": (
            "Using Table 6.1, report the six average regulatory scores for the Upper-middle-income and the "
            "Lower-middle-income groups, and identify the sub-indicator on which lower-middle-income countries score "
            "higher than upper-middle-income countries."
        ),
        "gold_chunk_ids": ["wdr2021:p227:vis_table_6_1"], "gold_pages": [227], "gold_elements": ["Table 6.1"],
        "expected_answer": (
            "Upper-middle-income: Cybersecurity and cybercrime 57, Personal data 46, Nonpersonal data 29, E-commerce and "
            "e-transactions 74, Public intent data 62, Private intent data 20. Lower-middle-income: 55, 43, 38, 72, 44, 15. "
            "Lower-middle exceeds upper-middle only on Nonpersonal data (38 vs 29)."
        ),
        "required_facts": [
            "Upper-middle: 57, 46, 29, 74, 62, 20",
            "Lower-middle: 55, 43, 38, 72, 44, 15",
            "Lower-middle > upper-middle only on Nonpersonal data (38 vs 29)",
        ],
        "verify_tokens": ["Upper-middle-income | 57 | 46 | 29 | 74 | 62 | 20", "Lower-middle-income | 55 | 43 | 38 | 72 | 44 | 15"],
    },
    {
        "id": "adv_v04", "split": "test_v2", "primary_group": "table", "answerable": True,
        "question": (
            "In Table 4.1, what specific research gaps (questions) are listed under the 'Quality standards' and "
            "'Machine learning' research areas?"
        ),
        "gold_chunk_ids": ["wdr2021:p162:vis_table_4_1"], "gold_pages": [162], "gold_elements": ["Table 4.1"],
        "expected_answer": (
            "Quality standards: (1) how standards can be created, agreed on, updated and communicated to the development "
            "community, and who needs to be part of these conversations; (2) what the standards for accuracy and precision "
            "should be for frontier applications that use private intent data or combine public and private intent data, so "
            "policy makers can trust and use results. Machine learning: (1) how the approach to machine learning and spatial "
            "feature selection must change from common tasks to specialized tasks that aid development policy; (2) which "
            "features best predict spatial variation in development outcomes in different contexts, and the trade-offs "
            "between predictive accuracy and cost."
        ),
        "required_facts": [
            "Quality standards: creating/agreeing/updating/communicating standards and who participates",
            "Quality standards: accuracy and precision standards for frontier private-intent applications",
            "Machine learning: changing ML/spatial feature selection approach for specialized development tasks",
            "Machine learning: features predicting spatial variation; predictive accuracy vs cost trade-off",
        ],
        "verify_tokens": ["Quality standards", "accuracy and precision", "spatial feature selection",
                          "predictive accuracy and cost"],
    },
    {
        "id": "adv_v05", "split": "test_v2", "primary_group": "table", "answerable": True,
        "question": (
            "What 'Objectives' column entries does Table 8.3 give for data aggregators/data brokers, for data pools, "
            "and for data trusts?"
        ),
        "gold_chunk_ids": ["wdr2021:p298:vis_table_8_3"], "gold_pages": [298], "gold_elements": ["Table 8.3"],
        "expected_answer": (
            "Data aggregators/brokers: transform raw data into more consumable information. Data pools: increase "
            "incentives for competitors to share and combine data resources for common use within sectors. Data trusts: "
            "facilitate sharing of sensitive data and provide collective bargaining power to individuals."
        ),
        "required_facts": [
            "Aggregators/brokers: transform raw data into more consumable information",
            "Data pools: incentives for competitors to share/combine data for common use within sectors",
            "Data trusts: facilitate sharing of sensitive data; collective bargaining power to individuals",
        ],
        "verify_tokens": ["Transform raw data into more", "Increase incentives for", "bargaining power to"],
    },
    {
        "id": "adv_v06", "split": "test_v2", "primary_group": "table", "answerable": True,
        "question": (
            "Table 7.2 compares key provisions on digital trade in recent trade agreements. Which agreements (with years) "
            "form its columns, which provisions form its rows, and how does the table note say the cells should be read?"
        ),
        "gold_chunk_ids": ["wdr2021:p264:vis_table_7_2"], "gold_pages": [264], "gold_elements": ["Table 7.2"],
        "expected_answer": (
            "Columns (8): US–Singapore FTA (2004), Canada–EU CETA (2014), ChAFTA (2015), CPTPP (2018), ASEAN (2019), "
            "USMCA (2020), DEPA (2020), RCEP (2020). Rows (9): nondiscrimination of digital products; e-documents and "
            "e-signatures; paperless trading; online consumer protection; privacy protection; cybersecurity; unrestricted "
            "cross-border data flows; prohibition of data localization; customs duties. Note: one symbol marks a binding "
            "provision, another marks a soft law provision, and an empty cell means no provision."
        ),
        "required_facts": [
            "8 agreements with years 2004, 2014, 2015, 2018, 2019, 2020 x3",
            "9 provision rows incl. unrestricted cross-border data flows, prohibition of data localization, customs duties",
            "Legend: binding vs soft law provision; empty = no provision",
        ],
        "verify_tokens": ["CETA (2014)", "CPTPP", "DEPA", "RCEP", "Paperless trading", "Prohibition of data",
                          "binding provision", "soft law provision"],
    },
    # ---------------------------- diagram (5) --------------------------------
    {
        "id": "adv_v07", "split": "test_v2", "primary_group": "diagram", "answerable": True,
        "question": (
            "Figure 1.3 shows three pathways along which data can foster development. Which outcomes does it show as "
            "positive impacts and which as negative impacts, and how does the figure distinguish them?"
        ),
        "gold_chunk_ids": ["wdr2021:p54:vis_figure_1_3"], "gold_pages": [54], "gold_elements": ["Figure 1.3"],
        "expected_answer": (
            "Positive impacts (green): greater transparency, greater accountability, better policy making and service "
            "delivery, increased business opportunities (leading to development). Negative impacts (red): criminal "
            "activity/dark net, political surveillance, market concentration, widening inequality, discrimination. The note "
            "states positive impacts are green and negative impacts red; the pathways run through individuals/civil "
            "society/academia, government/international organizations, and the private sector."
        ),
        "required_facts": [
            "Positive: greater transparency, greater accountability, better policy making and service delivery, increased business opportunities",
            "Negative: criminal activity/dark net, political surveillance, market concentration, widening inequality, discrimination",
            "Green = positive, red = negative",
        ],
        "verify_tokens": ["Greater transparency", "Greater accountability", "Political surveillance",
                          "Market concentration", "Widening inequality", "shown in green"],
    },
    {
        "id": "adv_v08", "split": "test_v2", "primary_group": "diagram", "answerable": True,
        "question": (
            "In Figure 1.5 (data flowing safely across stakeholders in an integrated national data system), which "
            "stakeholder groups surround the data, and which data attributes are listed?"
        ),
        "gold_chunk_ids": ["wdr2021:p61:vis_figure_1_5"], "gold_pages": [61], "gold_elements": ["Figure 1.5"],
        "expected_answer": (
            "Five stakeholder groups: academic institutions, government entities, civil society and individuals, "
            "international organizations, and the private sector. Data attributes: produced, protected, open, quality "
            "controlled, and used and reused."
        ),
        "required_facts": [
            "Stakeholders: academic institutions, government entities, civil society and individuals, international organizations, private sector",
            "Attributes: produced, protected, open, quality controlled, used and reused",
        ],
        "verify_tokens": ["Academic", "Government", "Civil society", "International", "Data attributes", "Quality controlled"],
    },
    {
        "id": "adv_v09", "split": "test_v2", "primary_group": "diagram", "answerable": True,
        "question": (
            "What features of well-functioning institutions for effective data governance are shown in Figure 8.7, and "
            "what does the figure say each feature should do?"
        ),
        "gold_chunk_ids": ["wdr2021:p301:vis_figure_8_7"], "gold_pages": [301], "gold_elements": ["Figure 8.7"],
        "expected_answer": (
            "Within a performance-enhancing environment: Coordination — create incentives for collaborative, user-centric "
            "decision-making; Leadership — mobilize data champions in positions of power to create a culture of data sharing "
            "and use; Multistakeholder governance — sustain outcomes through inclusive and legitimate stakeholder engagement; "
            "Transparency — build trust by promoting participation and accountability in policy making; Functional autonomy — "
            "promote institutional independence and autonomy for effective performance; Data literacy — invest in technical "
            "capacity, resources and data skills for civil servants."
        ),
        "required_facts": [
            "Coordination, Leadership, Multistakeholder governance, Transparency, Functional autonomy, Data literacy",
            "Correct description for each of the six features",
        ],
        "verify_tokens": ["Coordination", "Leadership", "Multistakeholder governance", "Transparency",
                          "Functional autonomy", "Data literacy", "data champions"],
    },
    {
        "id": "adv_v10", "split": "test_v2", "primary_group": "diagram", "answerable": True,
        "question": (
            "According to Figure 9.4, what steps are proposed for integrating civil society into the national data system "
            "at each of the three stages (establishing fundamentals, initiating data flows, optimizing the system)?"
        ),
        "gold_chunk_ids": ["wdr2021:p333:vis_figure_9_4"], "gold_pages": [333], "gold_elements": ["Figure 9.4"],
        "expected_answer": (
            "Establishing fundamentals: laws and regulations protecting people's rights to produce, use and disseminate data; "
            "incentives — create civil society watchdogs to monitor compliance; funding — ensure civil society has funding and "
            "resources for data production and use. Initiating data flows: human capital — improve data literacy via joint "
            "projects, training, secondments and the education system; infrastructure policies — adopt common standards that "
            "improve quality and interoperability; trust — an independent scientific quality control unit. Optimizing the "
            "system: institutions — institutionalize collaboration for systematic use of citizen-generated data in "
            "decision-making; incentives — include civil society in planning, high-level decisions and strategy setting."
        ),
        "required_facts": [
            "Fundamentals: laws on rights to produce/use/disseminate data; watchdogs; funding",
            "Initiating: data literacy; common standards; independent quality control unit",
            "Optimizing: institutionalize use of citizen-generated data; include civil society in planning/strategy",
        ],
        "verify_tokens": ["Laws and regulations", "civil society watchdogs", "secondments", "quality control unit",
                          "citizen-", "strategy setting"],
    },
    {
        "id": "adv_v11", "split": "test_v2", "primary_group": "diagram", "answerable": True,
        "question": (
            "According to Figure 9.6, what steps are recommended for integrating the private sector into the national data "
            "system at the 'Establishing fundamentals', 'Initiating data flows', and 'Optimizing the system' stages?"
        ),
        "gold_chunk_ids": ["wdr2021:p337:vis_figure_9_6"], "gold_pages": [337], "gold_elements": ["Figure 9.6"],
        "expected_answer": (
            "Establishing fundamentals: infrastructure policies — construct national transmission networks for high-speed "
            "wireless broadband; human capital — equip workers and businesses with skills and appliances to produce and use "
            "data. Initiating data flows: infrastructure policies — increase IXPs and data centers; incentives — adopt "
            "compensation and noncompensation schemes to promote data exchanges; data demand — promote the role of data "
            "stewards in business. Optimizing the system: incentives — increase interoperability through stringent data "
            "quality and privacy standards that are also pro-business; infrastructure policies — expand access to colocation "
            "data centers; trust — engage business through trusted data intermediaries."
        ),
        "required_facts": [
            "Fundamentals: national transmission networks for broadband; skills/appliances for workers and businesses",
            "Initiating: IXPs and data centers; compensation/noncompensation schemes; data stewards",
            "Optimizing: stringent pro-business quality/privacy standards; colocation data centers; trusted data intermediaries",
        ],
        "verify_tokens": ["transmission networks", "noncompensation", "data stewards", "pro-business",
                          "colocation data", "trusted data"],
    },
    # ---------------------------- chart (5) ----------------------------------
    {
        "id": "adv_v12", "split": "test_v2", "primary_group": "chart", "answerable": True,
        "question": (
            "In Figure 8.6, what share of countries in each income group has an established cybersecurity agency and what "
            "share has one in progress or planned? What is the high-income minus low-income gap for established agencies?"
        ),
        "gold_chunk_ids": ["wdr2021:p297:vis_figure_8_6"], "gold_pages": [297], "gold_elements": ["Figure 8.6"],
        "expected_answer": (
            "Share of countries with a cybersecurity agency (%), 198 economies: Low-income — established 24, in progress or "
            "planned 14; Lower-middle-income — established 66, in progress 12; Upper-middle-income — established 65, in "
            "progress 15; High-income — established 86, in progress 6. Gap for established agencies: 86 - 24 = 62 "
            "percentage points (title: only about one-quarter of low-income countries have such agencies). Bar-to-label "
            "mapping verified from the PDF text-layer coordinates (labels sit just above each bar top)."
        ),
        "required_facts": [
            "Established: 24 / 66 / 65 / 86",
            "In progress or planned: 14 / 12 / 15 / 6",
            "High minus low established gap = 62 percentage points",
        ],
        "verify_tokens": ["86", "66", "65", "24", "14", "12", "15", "In progress or planned", "Established"],
    },
    {
        "id": "adv_v13", "split": "test_v2", "primary_group": "chart", "answerable": True,
        "question": (
            "In panel a of Figure 3.6 (share of global internet traffic by firm, 2018), what percentage is shown for each "
            "named firm and for 'Others', and what combined share do the named firms account for?"
        ),
        "gold_chunk_ids": ["wdr2021:p119:vis_figure_3_6"], "gold_pages": [119], "gold_elements": ["Figure 3.6"],
        "expected_answer": (
            "Panel a (2018, % of total, Sandvine 2019): Google 12, Netflix 11, Facebook 8, Microsoft 5, Apple 4, Amazon 3, "
            "Others 57. The six named firms together account for 43 percent (12+11+8+5+4+3)."
        ),
        "required_facts": [
            "Google 12, Netflix 11, Facebook 8, Microsoft 5, Apple 4, Amazon 3, Others 57",
            "Named firms combined = 43 percent",
        ],
        "verify_tokens": ["Google, 12", "Netﬂix, 11", "Others, 57", "Facebook, 8", "Amazon, 3", "Sandvine 2019"],
    },
    {
        "id": "adv_v14", "split": "test_v2", "primary_group": "chart", "answerable": True,
        "question": (
            "Figure 2.9 has two scatter panels relating governance to statistical performance. What does each panel plot on "
            "its x-axis, which countries does each include, and what R² is reported for each?"
        ),
        "gold_chunk_ids": ["wdr2021:p89:vis_figure_2_9"], "gold_pages": [89], "gold_elements": ["Figure 2.9"],
        "expected_answer": (
            "Panel a: NSO independence score (Ibrahim Index of African Governance) vs statistical performance, African "
            "countries only, R² = 0.22. Panel b: RSF World Press Freedom Index (axis runs from 100 to 0) vs statistical "
            "performance, all countries with available data, R² = 0.23. Each x represents a country; both relations are positive."
        ),
        "required_facts": [
            "Panel a: NSO independence score, only African countries, R² = 0.22",
            "Panel b: RSF World Press Freedom Index, all countries with data, R² = 0.23",
        ],
        "verify_tokens": ["R² = 0.22", "R² = 0.23", "NSO independence score", "RSF World Press Freedom Index",
                          "only African countries"],
    },
    {
        "id": "adv_v15", "split": "test_v2", "primary_group": "chart", "answerable": True,
        "question": (
            "What does Figure B4.1.1 measure, over what period and relative to what, and which user groups are plotted as "
            "separate series?"
        ),
        "gold_chunk_ids": ["wdr2021:p143:vis_figure_b4_1_1"], "gold_pages": [143], "gold_elements": ["Figure B4.1.1"],
        "expected_answer": (
            "It shows changes in the time smartphone users spent at home in Jakarta from February 1 to November 15, 2020, "
            "relative to a baseline period (with PSBB large-scale social restrictions marked). Series are by wealth of the "
            "administrative unit: low-wealth users (0–40% of wealth), medium-wealth (>40–80%), and high-wealth (>80–100%)."
        ),
        "required_facts": [
            "Change in time spent at home in Jakarta, Feb 1 – Nov 15, 2020, relative to baseline",
            "Three wealth groups: low (0–40%), medium (>40–80%), high (>80–100%)",
        ],
        "verify_tokens": ["Jakarta", "Low-wealth", "Medium-wealth", "High-wealth", "relative to the baseline"],
    },
    {
        "id": "adv_v16", "split": "test_v2", "primary_group": "chart", "answerable": True,
        "question": (
            "Figure B1.4.1 tracks published COVID-19 articles by type of private intent data. Which data types are plotted, "
            "over what time span, from what source, and why does the note say the categories cannot simply be summed?"
        ),
        "gold_chunk_ids": ["wdr2021:p56:vis_figure_b1_4_1"], "gold_pages": [56], "gold_elements": ["Figure B1.4.1"],
        "expected_answer": (
            "Articles containing 'COVID' and: call detail records, cellphone, Facebook, Google Trends, big data, Twitter, "
            "social media; February to September 2020; source CORD-19 (COVID-19 Open Research Dataset, Semantic Scholar/"
            "Allen Institute for AI). The cumulative sum is higher than the number of articles because some articles appear "
            "in more than one category."
        ),
        "required_facts": [
            "Seven data types: call detail records, cellphone, Facebook, Google Trends, big data, Twitter, social media",
            "February–September 2020; CORD-19 source",
            "Articles can appear in more than one category",
        ],
        "verify_tokens": ["Call detail records", "Google Trends", "Twitter", "CORD-19", "more than one category"],
    },
    # ---------------------------- multihop (4) -------------------------------
    {
        "id": "adv_v17", "split": "test_v2", "primary_group": "multihop", "answerable": True,
        "question": (
            "Combine Figure 8.6 and Table 6.1: for low-income versus high-income countries, what share has an established "
            "cybersecurity agency, what is their average 'Cybersecurity and cybercrime' regulatory score, and what are the "
            "high-minus-low gaps for each measure?"
        ),
        "gold_chunk_ids": ["wdr2021:p297:vis_figure_8_6", "wdr2021:p227:vis_table_6_1"], "gold_pages": [297, 227],
        "gold_elements": ["Figure 8.6", "Table 6.1"],
        "expected_answer": (
            "Figure 8.6: established cybersecurity agency in 24% of low-income vs 86% of high-income countries (gap 62 "
            "percentage points). Table 6.1: Cybersecurity and cybercrime score 39 (low-income) vs 73 (high-income), gap 34 points."
        ),
        "required_facts": ["Agency: 24% vs 86%, gap 62 pp", "Score: 39 vs 73, gap 34 points"],
        "verify_tokens": ["24", "86", "High-income | 73", "Low-income | 39"],
    },
    {
        "id": "adv_v18", "split": "test_v2", "primary_group": "multihop", "answerable": True,
        "question": (
            "Compare the 'Establishing fundamentals' steps in Figure 9.5 (integrating academia) with those in Figure 9.7 "
            "(integrating international and regional organizations) into the national data system."
        ),
        "gold_chunk_ids": ["wdr2021:p335:vis_figure_9_5", "wdr2021:p340:vis_figure_9_7"], "gold_pages": [335, 340],
        "gold_elements": ["Figure 9.5", "Figure 9.7"],
        "expected_answer": (
            "Figure 9.5 (academia): infrastructure policies — ensure academics have adequate financial support and access to "
            "key ICT infrastructure; data demand — promote awareness of participants' datasets by building relationships "
            "between academia and government. Figure 9.7 (international/regional organizations): funding — seek support, "
            "including technical assistance, to improve weak parts of the data life cycle; human capital — increase technical "
            "capacity and data literacy; institutions — engage in regional data cooperation to save resources through "
            "economies of scale and facilitate peer-to-peer learning."
        ),
        "required_facts": [
            "9.5: financial support and ICT access for academics; awareness of datasets via academia–government relationships",
            "9.7: funding/technical assistance; technical capacity and data literacy; regional cooperation, economies of scale, peer-to-peer learning",
        ],
        "verify_tokens": ["academics have", "relationships among partners", "technical assistance", "peer-to-peer"],
    },
    {
        "id": "adv_v19", "split": "test_v2", "primary_group": "multihop", "answerable": True,
        "question": (
            "Using Figure 5.7 and Figure 5.8 together: what affordability threshold is drawn in Figure 5.7 and how are its "
            "group prices defined (year and statistic), and in Figure 5.8 what is on the x-axis, what does each point "
            "represent, and what R² is reported?"
        ),
        "gold_chunk_ids": ["wdr2021:p187:vis_figure_5_7"], "gold_pages": [187], "gold_elements": ["Figure 5.7", "Figure 5.8"],
        "expected_answer": (
            "Figure 5.7: monthly price of 1 GB of mobile data by income group; affordability threshold = 2 percent of monthly "
            "income (GNI per capita); prices are median prices of the economies in each group, data as of 2018 (Cable.co.uk). "
            "Figure 5.8: x-axis = average cost per user per GB (% of GNI per capita); each circle is a country; R² = 0.7637, "
            "showing data consumption is very sensitive to price/affordability."
        ),
        "required_facts": [
            "5.7: threshold 2% of monthly income; median prices; 2018",
            "5.8: x-axis average cost per GB as % of GNI per capita; each circle a country; R² = 0.7637",
        ],
        "verify_tokens": ["Affordability threshold", "2 percent of monthly", "median prices", "R² = 0.7637",
                          "Each circle represents a country"],
    },
    {
        "id": "adv_v20", "split": "test_v2", "primary_group": "multihop", "answerable": True,
        "question": (
            "Comparing Table 2.2 and Table 2.1: what are the low-income and high-income values for 'NSO makes metadata "
            "available' (Table 2.2) and for 'Open terms of use/license' (Table 2.1), and which indicator has the larger "
            "high-minus-low gap?"
        ),
        "gold_chunk_ids": ["wdr2021:p84:vis_table_2_2", "wdr2021:p83:vis_table_2_1"], "gold_pages": [84, 83],
        "gold_elements": ["Table 2.2", "Table 2.1"],
        "expected_answer": (
            "Metadata available (Table 2.2): 63% low-income vs 100% high-income, gap 37 percentage points. Open terms of "
            "use/license (Table 2.1): 11% vs 44%, gap 33 percentage points. The metadata gap (37) is larger by 4 points."
        ),
        "required_facts": ["Metadata 63 vs 100, gap 37", "Open license 11 vs 44, gap 33", "Metadata gap larger"],
        "verify_tokens": ["NSO makes metadata available | 63 | 91 | 97 | 100", "Open terms of use/license (%) | 11 | 19 | 22 | 44"],
    },
    # ---------------------------- text (4) -----------------------------------
    {
        "id": "adv_v21", "split": "test_v2", "primary_group": "text", "answerable": True,
        "question": (
            "Regarding WTO discussions on digital trade under the Joint Statement on Electronic Commerce, how many WTO members "
            "are involved and what share of digital trade do they represent, how many low-income versus high-income countries "
            "have joined, and which country proposed a WTO data governance framework and when?"
        ),
        "gold_chunk_ids": ["wdr2021:p264:p2", "wdr2021:p264:p3"], "gold_pages": [264], "gold_elements": [],
        "expected_answer": (
            "85 WTO members accounting for nearly 90 percent of digital trade. Only one low-income country (Burkina Faso) "
            "has joined, compared with 52 high-income countries. Japan proposed a data governance framework under the WTO "
            "during its G-20 presidency in 2019."
        ),
        "required_facts": ["85 members, ~90% of digital trade", "1 low-income (Burkina Faso) vs 52 high-income", "Japan, G-20 2019"],
        "verify_tokens": ["85 WTO members", "nearly 90 percent", "Burkina Faso", "52 high-income", "Japan"],
    },
    {
        "id": "adv_v22", "split": "test_v2", "primary_group": "text", "answerable": True,
        "question": (
            "What happened with the deidentified medical records released in Massachusetts in the 1990s, and what does the "
            "report suggest as one way to minimize such confidentiality concerns?"
        ),
        "gold_chunk_ids": ["wdr2021:p85:p0", "wdr2021:p85:p1"], "gold_pages": [85], "gold_elements": [],
        "expected_answer": (
            "The governor of Massachusetts approved releasing deidentified medical records of state employees to researchers; "
            "although name and address were removed, a researcher triangulated the data with other public information and "
            "identified the medical records of the governor and other individuals. One mitigation is to produce only "
            "appropriate data — data that measure concepts of interest, have a clear policy purpose, and are not collected "
            "excessively or to surveil individuals."
        ),
        "required_facts": [
            "Deidentified state-employee medical records released; name/address removed",
            "Re-identified (incl. governor) by triangulating with public information",
            "Mitigation: produce only appropriate data with clear policy purpose, no excessive collection/surveillance",
        ],
        "verify_tokens": ["governor of Massachusetts", "state employees", "triangulating", "clear policy purpose"],
    },
    {
        "id": "adv_v23", "split": "test_v2", "primary_group": "text", "answerable": True,
        "question": (
            "How does the report describe data brokers — what do they do, for what purposes are their profiles sold, and "
            "how many operate worldwide in an industry of what value?"
        ),
        "gold_chunk_ids": ["wdr2021:p298:p0"], "gold_pages": [298], "gold_elements": [],
        "expected_answer": (
            "Data brokers scrape public records and buy or license private data to build profiles of individuals sold for "
            "profit — often for marketing, risk mitigation (including identification verification and fraud detection), and "
            "people search pages. Some 4,000 data brokers operate worldwide in an industry valued at US$250 billion."
        ),
        "required_facts": [
            "Scrape public records, buy/license private data, build profiles",
            "Marketing, risk mitigation (ID verification, fraud detection), people search",
            "~4,000 brokers, US$250 billion",
        ],
        "verify_tokens": ["scrape public records", "fraud detection", "people search pages", "4,000 data brokers", "US$250 billion"],
    },
    {
        "id": "adv_v24", "split": "test_v2", "primary_group": "text", "answerable": True,
        "question": (
            "What example does the report give of the income gradient in adherence to international statistical standards, "
            "and which standard is by contrast used by a large share of all countries?"
        ),
        "gold_chunk_ids": ["wdr2021:p84:p3"], "gold_pages": [84], "gold_elements": [],
        "expected_answer": (
            "The International Classification of Status in Employment is used in two-thirds of high-income countries but only "
            "7 percent of low-income countries (figure 2.5). By contrast, a large share of all countries uses at least the 1993 "
            "System of National Accounts (SNA 1993)."
        ),
        "required_facts": ["ICSE: two-thirds of high-income vs 7% of low-income", "SNA 1993 widely used"],
        "verify_tokens": ["Status in Employment", "two-thirds", "7 percent", "SNA 1993"],
    },
    # ---------------------------- unanswerable (6) ---------------------------
    {
        "id": "adv_v25", "split": "test_v2", "primary_group": "unanswerable", "answerable": False,
        "question": "According to Table 7.2, which digital trade provisions are binding in the EU–Vietnam Free Trade Agreement (EVFTA)?",
        "gold_chunk_ids": [], "gold_pages": [], "gold_elements": [],
        "expected_answer": "Cannot answer: Table 7.2 covers eight agreements (US–Singapore, CETA, ChAFTA, CPTPP, ASEAN, USMCA, DEPA, RCEP); EVFTA is not included.",
        "required_facts": ["Abstain; EVFTA is not in Table 7.2 or the report"],
        "abstention_reason": "EVFTA absent from WDR 2021.",
        "forbidden_terms": ["EVFTA", "EU–Vietnam", "EU-Vietnam"],
    },
    {
        "id": "adv_v26", "split": "test_v2", "primary_group": "unanswerable", "answerable": False,
        "question": "In Figure 3.6, what share of global internet traffic in 2018 came from TikTok (ByteDance)?",
        "gold_chunk_ids": [], "gold_pages": [], "gold_elements": [],
        "expected_answer": "Cannot answer: Figure 3.6 lists Google, Netflix, Facebook, Microsoft, Apple, Amazon and Others; TikTok/ByteDance is not shown.",
        "required_facts": ["Abstain; TikTok/ByteDance not in Figure 3.6 or the report"],
        "abstention_reason": "TikTok/ByteDance absent from WDR 2021.",
        "forbidden_terms": ["TikTok", "ByteDance"],
    },
    {
        "id": "adv_v27", "split": "test_v2", "primary_group": "unanswerable", "answerable": False,
        "question": "In Figure 8.6, how did the share of low-income countries with an established cybersecurity agency change between 2015 and 2020?",
        "gold_chunk_ids": [], "gold_pages": [], "gold_elements": [],
        "expected_answer": "Cannot answer: Figure 8.6 is a single cross-section by income group (198 economies) with no 2015 vs 2020 time series.",
        "required_facts": ["Abstain; Figure 8.6 has no time series"],
        "abstention_reason": "Figure 8.6 has no year-by-year breakdown.",
        "forbidden_terms": ["2015", "2020"],
        "scoped_chunk": "wdr2021:p297:vis_figure_8_6",
    },
    {
        "id": "adv_v28", "split": "test_v2", "primary_group": "unanswerable", "answerable": False,
        "question": "What personal data breach notification deadline does India's Digital Personal Data Protection Act impose on data fiduciaries?",
        "gold_chunk_ids": [], "gold_pages": [], "gold_elements": [],
        "expected_answer": "Cannot answer: India's Digital Personal Data Protection Act (2023) post-dates WDR 2021 and is not discussed.",
        "required_facts": ["Abstain; DPDP Act absent from WDR 2021"],
        "abstention_reason": "Post-2021 Indian legislation absent from the report.",
        "forbidden_terms": ["Digital Personal Data Protection Act", "data fiduciar"],
    },
    {
        "id": "adv_v29", "split": "test_v2", "primary_group": "unanswerable", "answerable": False,
        "question": "What average regulatory scores does Table 6.1 report for the East Asia and Pacific region?",
        "gold_chunk_ids": [], "gold_pages": [], "gold_elements": [],
        "expected_answer": "Cannot answer: Table 6.1 reports averages by income group (high, upper-middle, lower-middle, low, global), not by region.",
        "required_facts": ["Abstain; Table 6.1 has income groups, not regions"],
        "abstention_reason": "No regional breakdown in Table 6.1.",
        "forbidden_terms": ["East Asia"],
        "scoped_chunk": "wdr2021:p227:vis_table_6_1",
    },
    {
        "id": "adv_v30", "split": "test_v2", "primary_group": "unanswerable", "answerable": False,
        "question": "According to Figure B4.1.1, how much did time spent at home increase for smartphone users in Surabaya during the PSBB?",
        "gold_chunk_ids": [], "gold_pages": [], "gold_elements": [],
        "expected_answer": "Cannot answer: Figure B4.1.1 covers Jakarta only; Surabaya is not in the report.",
        "required_facts": ["Abstain; Surabaya not covered"],
        "abstention_reason": "Surabaya absent from WDR 2021.",
        "forbidden_terms": ["Surabaya"],
    },
]

BENCHMARK_CASES = BENCHMARK_CASES + BENCHMARK_CASES_V2


def _normalize_q(q: str) -> str:
    return re.sub(r"\W+", " ", q.lower()).strip()


def validate_and_write(chunks_path: Path, base_cases_path: Path, out_dir: Path) -> dict[str, Any]:
    chunks = json.loads(chunks_path.read_text(encoding="utf-8"))
    by_id = {c["id"]: c for c in chunks}
    all_text_lower = "\n".join(c["text"].lower() for c in chunks)

    base_cases = json.loads(base_cases_path.read_text(encoding="utf-8"))
    base_norm = {_normalize_q(c["question"]) for c in base_cases}

    seen_norm: set[str] = set()
    split_counts = Counter()
    group_counts: dict[str, Counter] = {"dev": Counter(), "test": Counter(), "test_v2": Counter()}

    for item in BENCHMARK_CASES:
        norm = _normalize_q(item["question"])
        if norm in seen_norm:
            raise ValueError(f"Duplicate question inside advanced benchmark: {item['id']}")
        if norm in base_norm:
            raise ValueError(f"Question overlaps Base Day 7 benchmark: {item['id']}")
        seen_norm.add(norm)

        sp = item["split"]
        pg = item["primary_group"]
        split_counts[sp] += 1
        group_counts[sp][pg] += 1

        if item["answerable"]:
            combined = []
            for gid in item["gold_chunk_ids"]:
                if gid not in by_id:
                    raise KeyError(f"{item['id']} references missing chunk_id: {gid}")
                combined.append(by_id[gid]["text"])
            combined_norm = re.sub(r"\s+", " ", "\n".join(combined)).lower()
            for tok in item.get("verify_tokens", []):
                tok_norm = re.sub(r"\s+", " ", tok).lower()
                if tok_norm not in combined_norm:
                    raise AssertionError(
                        f"{item['id']} verification token {tok!r} not found in gold chunks {item['gold_chunk_ids']}"
                    )
        else:
            scoped_id = item.get("scoped_chunk")
            haystack = by_id[scoped_id]["text"].lower() if scoped_id else all_text_lower
            for term in item.get("forbidden_terms", []):
                if term.lower() in haystack:
                    raise AssertionError(
                        f"Unanswerable case {item['id']} forbidden term {term!r} appeared in corpus/scoped chunk!"
                    )

    assert split_counts["dev"] == 18, f"Expected 18 dev, got {split_counts['dev']}"
    assert split_counts["test"] == 30, f"Expected 30 test, got {split_counts['test']}"
    assert group_counts["dev"] == {"table": 3, "diagram": 3, "chart": 3, "multihop": 3, "text": 3, "unanswerable": 3}
    assert group_counts["test"] == {"table": 6, "diagram": 5, "chart": 5, "multihop": 4, "text": 4, "unanswerable": 6}
    assert split_counts["test_v2"] == 30, f"Expected 30 test_v2, got {split_counts['test_v2']}"
    assert group_counts["test_v2"] == {"table": 6, "diagram": 5, "chart": 5, "multihop": 4, "text": 4, "unanswerable": 6}

    out_dir.mkdir(parents=True, exist_ok=True)
    benchmark_out = []
    cases_out = []
    for item in BENCHMARK_CASES:
        b_row = {k: v for k, v in item.items() if k not in ("verify_tokens", "forbidden_terms", "scoped_chunk")}
        benchmark_out.append(b_row)
        cases_out.append({
            "id": item["id"],
            "split": item["split"],
            "primary_group": item["primary_group"],
            "question": item["question"],
            "answerable": item["answerable"],
            "filters": {},
        })

    (out_dir / "benchmark_advanced.json").write_text(
        json.dumps(benchmark_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    (out_dir / "cases_advanced.json").write_text(
        json.dumps(cases_out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    return {
        "total_cases": len(benchmark_out),
        "dev_groups": dict(group_counts["dev"]),
        "test_groups": dict(group_counts["test"]),
        "test_v2_groups": dict(group_counts["test_v2"]),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Build & validate Day 7 Advanced Multimodal Benchmark.")
    parser.add_argument("--chunks", type=Path, default=DEFAULT_CHUNKS)
    parser.add_argument("--base-cases", type=Path, default=DEFAULT_BASE_CASES)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUT_DIR)
    args = parser.parse_args()
    summary = validate_and_write(args.chunks, args.base_cases, args.output_dir)
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
