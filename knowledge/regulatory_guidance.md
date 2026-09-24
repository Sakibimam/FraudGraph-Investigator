# Regulatory guidance notes

Working notes on the public regulatory references listed with the dataset. They are
summaries written for retrieval by the investigation agent, not the documents themselves.
Each section names its source so an analyst can go back to the original.

## SAR narrative: the five essential elements
Source: FinCEN, SAR Narrative Guidance Package (https://www.fincen.gov/system/files/shared/sar_guidance_narrative.pdf)

A SAR narrative must explain who is conducting the suspicious activity, what instruments or
mechanisms were used, when the activity took place, where it took place, and why the
institution thinks the activity is suspicious. It should also describe how the activity was
carried out (the modus operandi). The narrative is the only free-text part of the report and
must stand on its own: a reader at a law-enforcement agency may not have access to the
institution's internal files. Be chronological, specific (dates, amounts, account and card
identifiers, device or IP details) and concise. Avoid internal jargon and codes the reader
cannot interpret.

## Complete and sufficient narratives
Source: FinCEN, Preparing a Complete and Sufficient SAR Narrative (https://www.fincen.gov/system/files/shared/sarnarrcompletguidfinal_112003.pdf)

Common deficiencies are narratives that are left blank, that say only "see attached", that
repeat the structured fields without explaining them, or that fail to say why the activity is
unusual for this customer. A sufficient narrative compares the activity to the customer's
expected behaviour, names every subject and linked account, gives the total amount and the
date range, and describes what the institution did (account closure, card block, monitoring).

## Supporting documentation
Source: FinCEN, SAR Supporting Documentation (FIN-2007-G003) (https://www.fincen.gov/system/files/shared/fin-2007-g003.pdf)

Supporting documentation is everything the institution relied on when deciding to file:
transaction records, device and login logs, customer correspondence, and the investigation
notes. It is not filed with the SAR but must be kept for five years and produced on request.
An internal case record that lists the evidence and the decisions is therefore a regulatory
requirement, not just good practice.

## When to file and continuing activity
Source: FinCEN, SAR Filing FAQs, October 2025 (https://www.fincen.gov/system/files/2025-10/SAR-FAQs-October-2025.pdf)

A report is expected when the institution knows, suspects or has reason to suspect that a
transaction involves funds from illegal activity, is designed to evade reporting requirements,
or has no apparent lawful purpose, above the applicable dollar thresholds. Structuring,
meaning deliberately breaking amounts to stay under a threshold, is itself reportable. The
decision not to file should also be documented with the reasoning.

## Account takeover red flags
Source: FinCEN Advisory FIN-2011-A016, Account Takeover Activity (https://www.fincen.gov/resources/advisories/fincen-advisory-fin-2011-a016)

Account takeover is the use of stolen credentials to take control of a legitimate customer's
account. Red flags: logins or transactions from new devices or unfamiliar IP ranges; changes
to contact details followed by transactions; activity inconsistent with the customer's
history mixed with normal activity; identity or address-match failures; and access through
anonymising proxies. The advisory asks institutions to describe the device and access
information in the SAR narrative.

## Money mules and imposter scams
Source: FinCEN Advisory on Imposter Scams and Money Mule Schemes (https://www.fincen.gov/system/files/advisory/2020-07-07/Advisory_%20Imposter_and_Money_Mule_COVID_19_508_FINAL.pdf)

Mule networks move stolen funds through many accounts that share infrastructure. Red flags:
many unrelated customers using the same device, email domain or address in a short window;
new accounts receiving and quickly spending funds; and coordinated activity with amounts in
a narrow band. Reports should name every linked account and the shared element.

## Identity-related suspicious activity
Source: FinCEN Financial Trend Analysis, Identity-Related Suspicious Activity, 2021 (https://www.fincen.gov/system/files/shared/FTA_Identity_Final508.pdf)

Most identity-related SARs involve impersonation, and a large share involve compromised
credentials used through online channels. Signals include device and identity mismatches,
failed authentication followed by success, and personal details that do not match the
account. Step-up authentication and customer verification are the recommended controls
before irreversible action.

## Cyber-enabled fraud flows
Source: FATF, Illicit Financial Flows from Cyber-Enabled Fraud (https://www.fatf-gafi.org/content/dam/fatf-gafi/reports/Illicit-financial-flows-cyber-enabled-fraud.pdf.coredownload.inline.pdf)

Cyber-enabled fraud is organised and industrialised: the same infrastructure (devices,
phishing kits, mule accounts) is reused across many victims. FATF recommends that
institutions look for network-level links between cases rather than treating each alert in
isolation, and share typologies. Card testing, account takeover and business email compromise
are named as common entry points.

## FFIEC red flags for card and electronic activity
Source: FFIEC BSA/AML Manual, Appendix F: Money Laundering and Terrorist Financing Red Flags (https://bsaaml.ffiec.gov/manual/Appendices/07)

Relevant red flags: transactions structured just below reporting or authorisation thresholds;
many transactions of similar amounts in a short period; activity inconsistent with the
customer's profile; use of anonymising services; and several customers sharing the same
contact or device details without an apparent reason.

## FFIEC suspicious activity reporting expectations
Source: FFIEC BSA/AML Manual, Suspicious Activity Reporting (https://bsaaml.ffiec.gov/manual/AssessingComplianceWithBSARegulatoryRequirements/04)

Institutions need a process to identify, research, decide and report. Decisions to file or
not file are documented, including who decided. Alerts should be dispositioned in a timely
way. Examiners check that the investigation is proportionate and that the narrative is
complete.
