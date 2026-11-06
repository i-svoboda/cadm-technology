"""Protocol data for the 2026 replication: verbatim prompts, personas, structures and the published numbers.

Running this file writes the JSON files in ../protocol. Sources (texts in
phd_dissertation/sources/own_publications):
  AHP: arXiv 2402.07404v1 (05_*.txt), SSRN 5069656 (06_*.txt), TACS 7(2) (02_*.txt)
  ANP: ITS 14(1) 2026, published version (03_*_published.txt) and final manuscript (03_*_docx.txt)
Every prompt marked "verbatim" is copied from arXiv v1 (typos kept). Text marked
"reconstructed" is written for this replication where the paper gives no wording.
"""
from __future__ import annotations

import json
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "protocol"

# ---------------------------------------------------------------- AHP 2024 study
GOAL = "Secure the Corporate Datacenter from Social Engineering Attacks"

AHP_GUIDE = {
    "name": "AHP Guide",
    "description": "Guides AHP decision-making, including managing external expert inputs.",
    "instructions": (
        "As an AHP Guide, your role includes facilitating users who are working with a specific problem or "
        "question using Saaty's Analytic Hierarchy Process. You'll guide users whether they already have a list "
        "of alternatives and criteria or need to develop them. Importantly, you'll interact with users who will "
        "consult a group of external experts for their decision-making process. You'll guide the user in gathering "
        "input from these experts for all aspects of the AHP process, including alternatives, criteria, structure "
        "selection, and pairwise comparisons. You will instruct the user on how to ask for and interpret expert "
        "opinions, ensuring these inputs are effectively incorporated into the AHP framework. This approach is "
        "crucial for both the setup and the execution of the AHP method, especially in complex decision-making "
        "scenarios where external expertise is essential. Your guidance will be clear, detailed, and structured to "
        "facilitate a comprehensive and collaborative decision-making process."
    ),
    "source": "arXiv v1 p. 3-4 and SSRN Appendix A (verbatim)",
    "not_reproduced": "The original GPT also had Saaty (1990) as a knowledge file and Web Browsing, DALL-E and "
                      "Code Interpreter enabled. None of these is available in the replication.",
}

GUIDE_PROMPTS = [
    {"id": "G1", "status": "verbatim, with the goal sentence added in front (the paper names the goal but not the "
                           "message that stated it)",
     "text": f'Our goal is "{GOAL}".\n\nI want to rely on the help of a group of experts. How many experts do you '
             f'think we need for the optimal solution?'},
    {"id": "G2", "status": "verbatim",
     "text": "Give me a list of 7 high quality diverse experts that will be the best fit in helping me with selecting "
             "criteria, selecting alternatives and completing pairwise comparison of said criteria.\nApproach each "
             "expert as a separate persona, describe their professional background, as well as work-related "
             "personality and preferences that would make them optimal for this AHP model building."},
    {"id": "G3", "status": "verbatim", "text": "How many criteria levels would be optimal for our goal?"},
]
GUIDE_PUBLISHED = {
    "G1": "In summary, for a decision as critical as securing a corporate datacenter, a group of 5-7 experts from key "
          "areas would be a good balance.",
    "G3": "Given the goal of securing a datacenter against social engineering attacks, a two-level structure is often "
          "optimal. It allows for sufficient detail and specificity without becoming overly complicated.",
}

AVA_INSTRUCTIONS = (
    "As Dr. Ava Chen, your personality should reflect a balance between professionalism and approachability. Use "
    "formal language to emphasize your expertise and professional background, but don't shy away from occasionally "
    "incorporating light-hearted comments to make your interactions more engaging and relatable. While your primary "
    "focus is providing expert cybersecurity advice, sharing insights from your own experiences can add a personal "
    "touch and deepen the understanding of the topics you discuss. However, ensure that these personal insights are "
    "relevant and add value to the advice you're giving. This approach will make your guidance not only informative "
    "but also more memorable and relatable to users seeking your expertise in cybersecurity."
)

# Appendix B of arXiv v1 (= SSRN Appendix F), verbatim
AHP_PERSONAS = [
    {"id": "A1", "name": "Dr. Ava Chen", "role": "Cybersecurity Strategist", "field": "cybersecurity",
     "background": "With a Ph.D. in Cybersecurity and over 15 years of experience in cyber defense strategies, Dr. "
                   "Chen has a deep understanding of various cybersecurity threats, including social engineering.",
     "personality": "Detail-oriented and analytical, she excels in identifying subtle security vulnerabilities and "
                    "prefers data-driven approaches. Dr. Chen will be instrumental in defining criteria related to "
                    "technical security measures and evaluating alternatives for effectiveness."},
    {"id": "A2", "name": "Michael Rodriguez", "role": "Senior IT Infrastructure Architect",
     "field": "secure IT infrastructure design",
     "background": "Michael specializes in designing secure IT infrastructures. His 20 years in the field give him a "
                   "thorough understanding of the technical aspects of datacenter operations.",
     "personality": "A problem-solver who enjoys exploring innovative solutions, Michael will contribute "
                    "significantly to identifying and evaluating alternatives that involve technical infrastructure "
                    "enhancements."},
    {"id": "A3", "name": "Dr. Yara Singh", "role": "Organizational Psychologist",
     "field": "human factors in security and workplace behavior",
     "background": "Dr. Singh's expertise lies in human behavior in the workplace. Her research on social engineering "
                   "vulnerabilities within corporate environments is widely respected.",
     "personality": "Known for her empathetic and intuitive nature, she is adept at understanding human factors in "
                    "security and will provide invaluable insights into criteria and alternatives related to employee "
                    "training and awareness programs."},
    {"id": "A4", "name": "Edward Kim", "role": "Legal and Compliance Officer",
     "field": "corporate law, compliance and data privacy regulation",
     "background": "Edward has extensive experience in corporate law, with a focus on compliance and data privacy "
                   "regulations.",
     "personality": "As a meticulous and thorough professional, he is well-suited to advise on legal and "
                    "compliance-related criteria, ensuring that the chosen security measures adhere to legal "
                    "standards."},
    {"id": "A5", "name": "Anita Patel", "role": "Chief Risk Officer", "field": "risk management and mitigation",
     "background": "Anita has a strong background in risk management and mitigation strategies in large "
                   "corporations.",
     "personality": "Her strategic and forward-thinking approach will be crucial in evaluating the long-term risks "
                    "and benefits of each alternative, especially in relation to financial and reputational impacts.",
     "extra": "TACS 7(2) describes her custom GPT as combining a progressive approach to risk analysis with "
              "accessibility (plain language and analogies)."},
    {"id": "A6", "name": "Lt. Col. John Abrams (Retd.)", "role": "Physical Security Expert",
     "field": "corporate physical security",
     "background": "With a military background and experience in corporate physical security, he understands the "
                   "importance of securing physical access to sensitive areas.",
     "personality": "His practical and no-nonsense approach will ground the discussion in realistic, enforceable "
                    "physical security measures."},
    {"id": "A7", "name": "Laura García", "role": "Vendor Management Specialist",
     "field": "third-party vendor management and vendor security risk",
     "background": "Laura specializes in managing relationships with third-party vendors and has a keen understanding "
                   "of the security risks associated with external entities.",
     "personality": "Her collaborative and communicative style is effective in discussions about managing external "
                    "risks and integrating vendor-related security considerations into the overall strategy."},
]


def ahp_custom_gpt(p: dict) -> dict:
    """Custom-GPT description and instructions. Ava Chen: verbatim. Others: reconstructed on her template."""
    if p["id"] == "A1":
        desc = "Professional yet approachable Dr. Ava Chen, blending expertise with personal insights."
        instr = AVA_INSTRUCTIONS
        status = "verbatim (arXiv v1 p. 5)"
    else:
        desc = f"{p['role']} {p['name']}, blending expertise with personal insights."
        instr = (
            f"As {p['name']}, your personality should reflect the profile given below. Use formal language to "
            f"emphasize your expertise and professional background, but don't shy away from occasionally "
            f"incorporating light-hearted comments to make your interactions more engaging and relatable. While your "
            f"primary focus is providing expert advice on {p['field']}, sharing insights from your own experiences "
            f"can add a personal touch and deepen the understanding of the topics you discuss. However, ensure that "
            f"these personal insights are relevant and add value to the advice you're giving."
        )
        if p["id"] == "A5":
            instr += (" Combine a progressive approach to risk analysis with accessibility: use plain language and "
                      "analogies.")
        status = "reconstructed on the template of Dr. Ava Chen (the paper publishes only her instructions)"
    return {"description": desc, "instructions": instr, "status": status}


def ahp_developer_message(p: dict) -> str:
    g = ahp_custom_gpt(p)
    return (
        f'You are a custom GPT (a version of ChatGPT customized for a specific use case) named "{p["name"]}".\n'
        f"Description: {g['description']}\n"
        f"Instructions:\n{g['instructions']}\n\n"
        f'Persona profile written by the "AHP Guide" assistant that designed this expert:\n'
        f"{p['role']}, {p['name']}:\n"
        f"Background: {p['background']}\n"
        f"Personality/Preferences: {p['personality']}"
    )


def guide_developer_message() -> str:
    return (
        f'You are a custom GPT (a version of ChatGPT customized for a specific use case) named "{AHP_GUIDE["name"]}".\n'
        f"Description: {AHP_GUIDE['description']}\n"
        f"Instructions:\n{AHP_GUIDE['instructions']}"
    )


# Published hierarchy (arXiv v1 pp. 10-11). Ids follow TACS 7(2) for the criteria.
AHP_CRITERIA = [
    {"id": "SEA", "name": "Social Engineering Awareness",
     "subs": [("TPE", "Training Program Effectiveness"), ("ASR", "Awareness Session Regularity"),
              ("IRP", "Incident Reporting Protocol")]},
    {"id": "PAC", "name": "Physical Access Controls",
     "subs": [("BSR", "Biometric System Reliability"), ("VTS", "Visitor Tracking System"),
              ("APM", "Access Point Monitoring")]},
    {"id": "AL", "name": "Audit Trails",
     "subs": [("LAA", "Log Analysis Accuracy"), ("AFR", "Audit Frequency"), ("ATE", "Anomaly Tracking Efficiency")]},
    {"id": "BA", "name": "Behavior Analysis",
     "subs": [("UBM", "User Behavior Monitoring"), ("RTA", "Response to Anomalies"),
              ("APA", "Activity Pattern Analysis")]},
    {"id": "ORC", "name": "Operational Risk Controls",
     "subs": [("IVC", "Infrastructure Vulnerability Check"), ("DRS", "Data Redundancy Systems"),
              ("EPE", "Emergency Protocol Effectiveness")]},
    {"id": "PP", "name": "Psychological Profiling",
     "subs": [("SBA", "Staff Behavior Assessment"), ("RBP", "Risk Behavior Profiling"),
              ("COB", "Continuous Observation")]},
    {"id": "SLA", "name": "Service Level Agreements",
     "subs": [("RTC", "Response Time Commitment"), ("DPA", "Data Privacy Assurance"),
              ("BPS", "Breach Penalty Specification")]},
]
# Order as written in the alternatives prompt (arXiv v1 p. 9)
AHP_ALTERNATIVES = [
    ("BKP", "Cloud-Based Data Backup Solutions"),
    ("PHB", "Physical Barrier Reinforcement"),
    ("STF", "Security Personnel Training Update"),
    ("TRN", "Comprehensive Employee Training Programs"),
    ("IDS", "Advanced Intrusion Detection Systems"),
]

# Appendix C: initial criteria by expert (49). The paper removed "repeating criteria such as Employee
# Training, Communication Protocols, Physical Security and Access Control" to get 45. All four are in
# Dr. Chen's list, so the 45 are the 49 without those four entries of her list.
APPENDIX_C = {
    "A1": ["Employee Training", "Access Control", "Communication Protocols", "Incident Response", "Physical Security",
           "Policy Enforcement", "Monitoring Systems"],
    "A2": ["System Redundancies", "Database Security", "Cloud Infrastructure Security", "Hardware Integrity",
           "Software Updates", "Server Access Control", "Network Segmentation"],
    "A3": ["Employee Training Programs", "Behavior Analysis", "Social Engineering Awareness", "Psychological Profiling",
           "Insider Threat Monitoring", "Communication Protocols", "Staff Vigilance"],
    "A7": ["Vendor Risk Assessment", "Third-party Audits", "Supply Chain Security", "Service Level Agreements",
           "External Collaboration Security", "Information Sharing Policies", "Outsourcing Management"],
    "A4": ["Regulatory Compliance", "Legal Risk Assessment", "Policy Development", "Contractual Safeguards",
           "Audit Trails", "Reporting Mechanisms", "Intellectual Property Protection"],
    "A5": ["Risk Management Framework", "Business Continuity", "Disaster Recovery Planning", "Financial Impact Analysis",
           "Strategic Risk Evaluation", "Operational Risk Controls", "Compliance Risk Management"],
    "A6": ["Physical Access Controls", "Surveillance Systems", "Security Personnel Training", "Entry Point Security",
           "Emergency Response", "Visitor Management", "Environmental Controls"],
}
REMOVED_FROM_A1 = ["Employee Training", "Access Control", "Communication Protocols", "Physical Security"]


def published_45() -> list[str]:
    out = []
    for pid in ["A1", "A2", "A3", "A7", "A4", "A5", "A6"]:          # order of Appendix C
        for c in APPENDIX_C[pid]:
            if pid == "A1" and c in REMOVED_FROM_A1:
                continue
            out.append(c)
    assert len(out) == 45 and len(set(out)) == 45, len(out)
    return out


AHP_PUBLISHED = {
    "source": "arXiv 2402.07404v1 (12.02.2024), Table 1 and pp. 12-13; the same numbers in SSRN 5069656 and TACS 7(2)",
    "model": "GPT-4 custom GPTs in the ChatGPT interface, early 2024 (arXiv v1)",
    "criteria_order": ["SEA", "PAC", "AL", "BA", "ORC", "PP", "SLA"],
    "aggregated_criteria_matrix": [
        [1.000, 1.319, 1.104, 1.483, 1.081, 0.498, 0.369],
        [0.756, 1.000, 1.673, 1.560, 1.029, 0.937, 0.408],
        [0.904, 0.601, 1.000, 1.251, 0.701, 0.756, 0.325],
        [0.674, 0.641, 0.798, 1.000, 0.627, 0.801, 0.503],
        [0.920, 0.966, 1.426, 1.608, 1.000, 0.604, 0.526],
        [2.007, 1.068, 1.319, 1.247, 1.636, 1.000, 0.652],
        [2.712, 2.438, 3.061, 1.990, 1.883, 1.532, 1.000],
    ],
    "criteria_weights": {"SEA": 0.120, "PAC": 0.131, "AL": 0.099, "BA": 0.096, "ORC": 0.126, "PP": 0.164,
                         "SLA": 0.264},
    "criteria_weights_tacs_4dp": {"SEA": 0.1204, "PAC": 0.1305, "AL": 0.0990, "BA": 0.0961, "ORC": 0.1260,
                                  "PP": 0.1640, "SLA": 0.2640},
    "criteria_consistency": {"CI": 0.022, "CR": 0.016, "lambda_max": 7.13},
    "subcriteria_global": {"RTC": 0.1127, "DPA": 0.0866, "SBA": 0.0655, "BPS": 0.0644, "IVC": 0.0573,
                           "RBP": 0.0546, "BSR": 0.0502, "TPE": 0.0485, "COB": 0.0440, "VTS": 0.0434,
                           "AFR": 0.0385, "DRS": 0.0384, "UBM": 0.0378, "APM": 0.0375, "IRP": 0.0368,
                           "ASR": 0.0347, "RTA": 0.0340, "LAA": 0.0317, "EPE": 0.0302, "ATE": 0.0288,
                           "APA": 0.0242},
    "subcriteria_cr_range": {"min": 0.002, "min_matrix": "PP", "max": 0.02, "max_matrix": "PAC"},
    "alternatives": {"BKP": 0.1938, "PHB": 0.1254, "STF": 0.1795, "TRN": 0.2774, "IDS": 0.2240},
    "alternatives_cr_range": {"min": 0.002, "min_matrix": "UBM", "max": 0.03, "max_matrix": "IVC"},
    "individual_matrices": "most matrices had CR under 0.01; all below 0.1",
    "example_matrix_A1_SEA": {"elements": ["TPE", "ASR", "IRP"],
                              "rows": [[1, 2, 3], [0.5, 1, 2], [1 / 3, 0.5, 1]]},
    "matrix_count": {"arxiv_text": 231, "dissertation": 203, "per_expert": "1 + 7 + 21 = 29"},
    "counts": {"top_level_criteria_proposed": 49, "after_dedup": 45, "sub_criteria_proposed": 147,
               "alternatives_proposed": 35},
    "loeo_tacs_7_2": "top alternative and its gap to the runner-up unchanged in every leave-one-expert-out run",
    "cost": "about USD 4 (gpt-4-32k list prices, 4,350 tokens per expert conversation)",
}

FORMAT_NOTE = ("Give your answer as the JSON object required by the output schema.")
MATRIX_FORMAT_NOTE = (
    "Give the matrices as the JSON object required by the output schema. For each matrix give its name, the elements "
    "in the order listed above, the full matrix as rows (one row per element, in the same order), with every value "
    "written like \"1\", \"3\" or \"1/5\", and a short rationale."
)

AHP_PROMPTS = {
    "hello_criteria": {
        "status": "verbatim (two sentences of one quoted prompt, arXiv v1 p. 5)",
        "text": f'Hello, you have been tasked with being a part of group of experts working on an AHP tree that has '
                f'the following top goal: "{GOAL}".\nFirst, come up with 7 top-level criteria. Use 3 words max for '
                f'each criteria.'},
    "hello_only": {
        "status": "verbatim first sentence of the prompt above; used to open conversations that start with a matrix",
        "text": f'Hello, you have been tasked with being a part of group of experts working on an AHP tree that has '
                f'the following top goal: "{GOAL}".'},
    "score_criteria": {
        "status": "verbatim (arXiv v1 p. 6); the number 45 is replaced by the actual count",
        "text": "Can you please assign a score from 1 to 9, with 1 being lowest importance to 9 being highest "
                "importance, for each of these {n} criteria:\n{items}"},
    "subcriteria": {
        "status": "verbatim (arXiv v1 p. 6)",
        "text": "Now, come up with 3 sub-criteria for each of these criteria: {criteria}. Use 3 words max for each "
                "sub-criteria."},
    "score_subcriteria": {
        "status": "reconstructed: the paper says each expert ranked the sets of sub-criteria and the highest-scored "
                  "ones were kept; formula (1) scores criteria, sub-criteria and alternatives the same way",
        "text": "Can you please assign a score from 1 to 9, with 1 being lowest importance to 9 being highest "
                "importance, for each of these sub-criteria. They are grouped by their top-level criterion:\n{items}"},
    "alternatives": {
        "status": "reconstructed: the paper says each expert was asked to come up with 5 alternatives; the word limit "
                  "follows the criteria prompts",
        "text": "Next, come up with 5 alternatives that could satisfy our main goal ({goal}). Use 5 words max for each "
                "alternative."},
    "score_alternatives": {
        "status": "reconstructed from the description (scores 1 to 9 on whether an alternative can satisfy the goal)",
        "text": "Can you please assign a score from 1 to 9, with 1 being lowest to 9 being highest, for each of these "
                "{n} alternatives, based on whether the alternative can satisfy our main goal ({goal}):\n{items}"},
    "criteria_matrix": {
        "status": "verbatim (arXiv v1 p. 7)",
        "text": "I now need you to create a pairwise comparison matrix for the list of our top-level criteria: "
                "{criteria}.\nThe matrix should be build based on Saaty's AHP methodology. Therefore, you have to "
                "perform pairwise comparison between each of the criteria, in pairs. You have to assign value from 1/9 "
                "to 9 based on whether one criteria is less or more important to our main goal ({goal}) than the "
                "another one. If they are equally important, the score is 1.\nAs an expert, I would like you to assign "
                "weights based on your personal subjective analysis and judgement."},
    "sub_matrices": {
        "status": "verbatim (arXiv v1 Appendix A)",
        "text": "Great work. Now, next step. For each top-level criterion, we have 3 sub-criteria. The tree looks like "
                "this:\n{tree}\nI now want you to create a separate comparison matrix for each of these 7 top-level "
                "criteria, where you will be comparing their sub-criteria.\nAs an expert, I would like you to assign "
                "weights based on your personal subjective analysis and judgement."},
    "alt_matrices": {
        "status": "verbatim (arXiv v1 p. 8-9) with the next three sub-criteria in each of the seven turns",
        "text": "I want you to build pairwise comparison matrices to select best alternatives following AHP "
                "guidelines. Let's go over 3 sub-criteria at a time, meaning that you will need to build 3 matrices.\n"
                "Sub-criteria are: {subs}.\nAlternatives are: {alts}.\nThe question should sound \"Between alternative "
                "A and alternative B, which one better satisfies (or performs with respect to) this sub-criterion?\"\n"
                "As an expert, I would like you to assign weights based on your personal subjective analysis and "
                "judgement."},
    "reminder": {
        "status": "reconstructed: the paper says matrices built incorrectly were 'easily fixed with a reminder "
                  "prompt' without giving it",
        "text": "Reminder of the AHP guidelines: each matrix compares every element with every other element of its "
                "list, the diagonal is 1, and every value below the diagonal is the reciprocal of the mirrored value "
                "above it (a_ji = 1/a_ij), with values from 1/9 to 9. {problems} Please rebuild all {n} matrices of "
                "your last answer following these guidelines."},
    "reminder_generic": {
        "status": "reconstructed (used when a list or score answer is incomplete)",
        "text": "Your last answer did not fully follow the request. {problems} Please give the complete answer again."},
}

# ---------------------------------------------------------------- ANP ITS 14(1) study
ANP_GOAL = ("Prioritize security controls for a national critical infrastructure operator that must allocate limited "
            "resources, under realistic budget and staffing constraints, to reduce mission risk over the next 12-18 "
            "months against sophisticated external state adversaries")
ANP_CLUSTERS = {
    "K": {"name": "Security controls", "nodes": [
        ("MON", "ICS network monitoring and anomaly detection"),
        ("SEG", "Network segmentation and unidirectional gateways"),
        ("IDM", "Identity, MFA and privileged access management"),
        ("IR", "Incident response and staff training"),
        ("SUP", "Supply-chain security, including SBOM and code signing"),
        ("BAK", "Offline immutable backups and recovery for ICS"),
        ("PHY", "Physical and environmental hardening of sites")]},
    "R": {"name": "Evaluation criteria", "nodes": [
        ("E", "Effectiveness against the TTPs of state actors"),
        ("T", "Time to implement (less is better)"),
        ("U", "Operational impact and availability friction (less is better)"),
        ("C", "Implementation cost (less is better)"),
        ("P", "Policy and compliance fit (NIST CSF 2.0, NIS2)")]},
    "H": {"name": "Threat vectors", "nodes": [
        ("PHISH", "Targeted spear-phishing"),
        ("SC", "Supply-chain compromise"),
        ("EXP", "Remote exploitation of vulnerabilities"),
        ("INS", "Insider threat"),
        ("SAB", "Physical sabotage")]},
    "B": {"name": "Implementation constraints", "nodes": [
        ("BUD", "Budget limits"),
        ("WRK", "Workforce readiness"),
        ("LOCK", "Vendor lock-in risk")]},
}
# ITS 14(1) Table 2 (abbreviated Ukrainian text) in the English rendering of the TACS SMAA project
ANP_PERSONAS = [
    {"id": "E1", "role": "ICS engineer",
     "conditioning": "You operate PLC and DCS equipment from several vendors at a critical infrastructure operator. "
                     "Safety and availability come first for you, and you distrust intrusive scanning of live control "
                     "systems. Other things being equal, less implementation time, cost and operational impact is "
                     "better."},
    {"id": "E2", "role": "SOC lead (blue team)",
     "conditioning": "You run a 24/7 security operations centre with deep packet inspection sensors in the industrial "
                     "control network. You care about detection latency and the triage load on your analysts."},
    {"id": "E3", "role": "ICS red-team specialist",
     "conditioning": "You emulate state-level adversaries against industrial control systems. You weigh initial access "
                     "and lateral movement, and you take data diodes and jump hosts into account."},
    {"id": "E4", "role": "Supply-chain security lead",
     "conditioning": "You are responsible for SBOMs, code signing, third-party risk and the provenance of software and "
                     "firmware updates."},
    {"id": "E5", "role": "Compliance specialist",
     "conditioning": "You map security controls to the NIST CSF 2.0 functions and to NIS2 outcomes, and you weigh how "
                     "well each control meets policy and compliance requirements (criterion P)."},
    {"id": "E6", "role": "Field operations manager",
     "conditioning": "You manage technicians, work permits and maintenance windows at field sites. You weigh the "
                     "operational friction that a control creates (criterion U)."},
    {"id": "E7", "role": "Chief information security officer",
     "conditioning": "You weigh the budget (BUD), workforce readiness (WRK) and vendor lock-in (LOCK), and you care "
                     "about governance and the order in which controls are introduced."},
]


def anp_developer_message(p: dict) -> str:
    return (f"You are a virtual expert in a structured judgment study. Role: {p['role']}. {p['conditioning']}")


ANP_PUBLISHED = {
    "source": "ITS 14(1) 2026, pp. 235-252 (published version)",
    "model": "gpt-5-2025-08-07 through the OpenAI API as of 1 October 2025, temperature 0.2, top_p 1.0, structured "
             "numeric output",
    "cluster_prior_v": {"K": 0.4466, "H": 0.2549, "R": 0.2506, "B": 0.0480},
    "control_order_in_table3": ["MON", "SEG", "IDM", "IR", "SUP", "BAK", "PHY"],
    "table3_local": {
        "R|GOAL": {"vector": [0.4463, 0.1504, 0.1016, 0.1475, 0.1543], "lambda_max": 5.0030, "CI": 0.00067},
        "K|R:E": {"vector": [0.2479, 0.2737, 0.1299, 0.1710, 0.0972, 0.0506, 0.0297], "lambda_max": 7.0123, "CI": 0.00155},
        "K|R:T": {"vector": [0.0962, 0.0693, 0.1438, 0.1804, 0.1973, 0.2126, 0.1004], "lambda_max": 7.0090, "CI": 0.00114},
        "K|R:U": {"vector": [0.0920, 0.1006, 0.1618, 0.1743, 0.1710, 0.2016, 0.0987], "lambda_max": 7.0100, "CI": 0.00126},
        "K|R:C": {"vector": [0.0864, 0.0716, 0.1338, 0.1772, 0.1712, 0.2359, 0.1239], "lambda_max": 7.0143, "CI": 0.00180},
        "K|R:P": {"vector": [0.1564, 0.1817, 0.1696, 0.1291, 0.2095, 0.0936, 0.0600], "lambda_max": 7.0067, "CI": 0.00084},
    },
    "global": {"MON": 0.1948, "SEG": 0.1832, "IDM": 0.1623, "SUP": 0.1354, "IR": 0.1342, "BAK": 0.1242,
               "PHY": 0.0659},
    "consistency": {"mean_CR": 0.006, "sd_CR": 0.004, "mean_Koczkodaj": 0.034, "n_matrices": 32,
                    "all_below": {"CR": 0.10, "K": 0.10}},
    "lambda2": 0.947,
    "loeo": {"MON": [0.1948, 0.1933, 0.0028, 0.1887, 0.1964, 0], "SEG": [0.1832, 0.1828, 0.0038, 0.1769, 0.1882, 0],
             "IDM": [0.1623, 0.1643, 0.0036, 0.1583, 0.1679, 0], "SUP": [0.1354, 0.1351, 0.0040, 0.1300, 0.1406, 4],
             "IR": [0.1342, 0.1362, 0.0058, 0.1269, 0.1421, 5], "BAK": [0.1242, 0.1253, 0.0045, 0.1207, 0.1333, 1],
             "PHY": [0.0659, 0.0630, 0.0070, 0.0531, 0.0723, 0],
             "columns": ["base", "mean", "sd", "min", "max", "rank_changes_of_7"],
             "mean_abs_change": 0.0039, "mean_kendall_tau": 0.90},
    "kendall_w": {"E": 0.77, "P": 0.74, "C": 0.68, "T": 0.65, "U": 0.62, "p": "<0.001", "m": 7, "N": 7},
    "monte_carlo": {"trials": 1000, "sigma_rule": "CR of perturbed matrices about 0.05-0.10",
                    "top2_kept": 1.00, "sup_ir_swap": 0.57, "max_weight_change": 0.015},
    "knapsack": {"costs_musd": {"MON": 3.8, "SEG": 5.2, "IDM": 3.0, "IR": 1.4, "SUP": 2.0, "BAK": 1.6, "PHY": 0.8},
                 "budget_musd": 10.2, "selected": ["MON", "IDM", "IR", "SUP"], "risk_reduction_musd": 27,
                 "alpha_and_eal": "not published (ATT&CK for ICS mapping and expert calibration; EAL conventional)",
                 "cost_order_assumption": "costs listed in the order MON, SEG, IDM, IR, SUP, BAK, PHY"},
    "scenarios": {"SC": {"MON": 0.188, "SEG": 0.177, "IDM": 0.150, "SUP": 0.160, "IR": 0.136, "BAK": 0.116, "PHY": 0.073},
                  "EXP": {"MON": 0.196, "SEG": 0.198, "IDM": 0.164, "SUP": 0.121, "IR": 0.126, "BAK": 0.098, "PHY": 0.097},
                  "INS+SAB": {"MON": 0.182, "SEG": 0.169, "IDM": 0.176, "SUP": 0.110, "IR": 0.160, "BAK": 0.117,
                              "PHY": 0.086},
                  "multipliers": "not published"},
}


def main() -> None:
    OUT.mkdir(exist_ok=True)
    ahp = {
        "goal": GOAL, "guide": {**AHP_GUIDE, "developer_message": guide_developer_message()},
        "guide_prompts": GUIDE_PROMPTS, "guide_published_answers": GUIDE_PUBLISHED,
        "criteria": [{"id": c["id"], "name": c["name"],
                      "sub_criteria": [{"id": s, "name": n} for s, n in c["subs"]]} for c in AHP_CRITERIA],
        "alternatives": [{"id": a, "name": n} for a, n in AHP_ALTERNATIVES],
        "appendix_c_initial_criteria": APPENDIX_C, "removed_from_A1": REMOVED_FROM_A1,
        "published_45_criteria": published_45(),
        "prompts": AHP_PROMPTS, "format_note": FORMAT_NOTE, "matrix_format_note": MATRIX_FORMAT_NOTE,
        "published": AHP_PUBLISHED,
    }
    (OUT / "ahp_2024_protocol.json").write_text(json.dumps(ahp, indent=1, ensure_ascii=False), encoding="utf-8")
    personas = [{**p, "custom_gpt": ahp_custom_gpt(p), "developer_message": ahp_developer_message(p)}
                for p in AHP_PERSONAS]
    (OUT / "ahp_2024_personas.json").write_text(json.dumps({"source": "arXiv v1 Appendix B (profiles, verbatim)",
                                                           "personas": personas}, indent=1, ensure_ascii=False),
                                               encoding="utf-8")
    anp = {"goal": ANP_GOAL, "clusters": {k: {"name": v["name"], "nodes": [{"id": i, "name": n} for i, n in v["nodes"]]}
                                          for k, v in ANP_CLUSTERS.items()},
           "published": ANP_PUBLISHED}
    (OUT / "anp_2026_protocol.json").write_text(json.dumps(anp, indent=1, ensure_ascii=False), encoding="utf-8")
    (OUT / "anp_2026_personas.json").write_text(json.dumps(
        {"source": "ITS 14(1) Table 2, English rendering from tacs_smaa_llm_elicitation/experiments/personas/"
                   "anp_ci_controls_personas.json (BUD, WRK and LOCK named in E7 as in Table 2)",
         "personas": [{**p, "developer_message": anp_developer_message(p)} for p in ANP_PERSONAS]},
        indent=1, ensure_ascii=False), encoding="utf-8")
    print("written:", sorted(x.name for x in OUT.glob("*.json")))


if __name__ == "__main__":
    main()
