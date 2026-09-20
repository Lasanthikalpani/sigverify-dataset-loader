\# SigVerify - Supervisor Meeting Summary



\*\*Project:\*\* Explainable AI for Signature Forgery Detection in Sri Lankan Government Documents Using Siamese Convolutional Neural Networks



\*\*Student:\*\* S.K.K.K.L. Chandrasekara



\*\*Date:\*\* 20 September 2026



\*\*Supervisor:\*\* \[Supervisor Name]



\---



\## 📌 Executive Summary



| RQ | Title | Status | Key Result |

|----|-------|--------|------------|

| \*\*RQ1\*\* | Core AI Performance | ✅ Complete | 98% accuracy |

| \*\*RQ2\*\* | Explainability \& Trust | ✅ Complete | Grad-CAM working |

| \*\*RQ3\*\* | Blockchain + Tamper Detection | ✅ Complete | 100% detection |



\*\*Supervisor's Idea Implemented:\*\* ✅ Batch blockchain with 2000 documents per block



\---



\## 🎯 RQ3: Complete Implementation



\### Supervisor's Original Idea



> "Blockchain add node (document wise) optimize it to document + large number to ledger (eg 2000 birth certificate QR verification)"



\### ✅ Fully Implemented



| Requirement | Target | Achieved |

|-------------|--------|----------|

| Documents per block | 2000 | ✅ 2000 |

| Storage reduction | ≥90% | ✅ 99.90% |

| Tamper detection | ≥95% | ✅ 100% |

| Blockchain validity | 100% | ✅ True |

| Verification speed | ≤2 seconds | ✅ <1 second |



\---



\## 📁 Test Files and Commands



\### RQ1: Siamese CNN (Signature Verification)



| File | Purpose | Command |

|------|---------|---------|

| `test\_gradcam.py` | Grad-CAM explainability test | `python test\_gradcam.py` |

| `src/model.py` | Siamese CNN model | - |

| `src/train.py` | Training script | `python src\\train.py` |



\### RQ3: Blockchain + Tamper Detection



| File | Purpose | Command |

|------|---------|---------|

| `test\_rq3\_complete.py` | Complete RQ3 test | `python test\_rq3\_complete.py` |

| `src/tamper\_generator.py` | Tamper detection (6 strategies) | `python src\\tamper\_generator.py` |

| `src/hybrid\_dataset.py` | Hybrid dataset (2000 docs) | `python src\\hybrid\_dataset.py` |

| `src/show\_documents.py` | Show/save original documents | `python src\\show\_documents.py` |

| `src/load\_real\_data.py` | Load real data from CSV | `python src\\load\_real\_data.py` |

| `create\_real\_csv.py` | Create sample CSV (100 rows) | `python create\_real\_csv.py` |



\---



\## 📊 Test Results Summary



\### RQ3: Tamper Detection (6 Strategies)



| # | Strategy | Detection Rate |

|---|----------|---------------|

| 1 | change\_date | ✅ 100% |

| 2 | change\_place | ✅ 100% |

| 3 | change\_name | ✅ 100% |

| 4 | change\_id | ✅ 100% |

| 5 | swap\_fields | ✅ 100% |

| 6 | multiple\_changes | ✅ 100% |

| | \*\*TOTAL\*\* | \*\*✅ 100%\*\* |



\### RQ3: Dataset Distribution



| Type | Count | Percentage |

|------|-------|------------|

| Synthetic | 1900 | 95% |

| Real | 100 | 5% |

| \*\*Total\*\* | \*\*2000\*\* | \*\*100%\*\* |



\### RQ3: Blockchain Results



| Metric | Value |

|--------|-------|

| Total blocks | 2 (Genesis + Batch) |

| Transactions per block | 2000 |

| Merkle Root | 914d8673479d389b... |

| Block Hash | 005ea2d41f2aa46c... |

| Storage saved | 99.90% |

| Chain valid | ✅ True |



\---



\## 🏗️ RQ3: Complete Theory



\### Layer 1: Dual Hashing



| Hash | Formula | Purpose |

|------|---------|---------|

| Content Hash | SHA256(content) | Document integrity |

| Metadata Hash | SHA256(timestamp + issuer + doc\_id) | Issuance authenticity |



\### Layer 2: Blockchain Batch Ledger



| Component | Value |

|-----------|-------|

| Batch size | 2000 documents/block |

| Data structure | Merkle tree |

| Storage reduction | 99.90% |

| Verification | O(log n) |



\### Layer 3: QR Code Verification



QR contains:

```json

{

&#x20; "content\_hash": "028826135b243c7d...",

&#x20; "metadata\_hash": "573e4be49b2911c7...",

&#x20; "signature": "375d861dc2f70e36...",

&#x20; "timestamp": "2026-09-15T22:09:52",

&#x20; "issuer": "Registrar General's Office",

&#x20; "document\_id": "BC-2026-000001"

}

