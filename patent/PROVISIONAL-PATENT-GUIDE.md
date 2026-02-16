# Provisional Patent Filing Guide — Biomimetic Neural Architecture

*Research compiled 2026-02-16 by Dross*

---

## TL;DR — What We Need to Do

1. **⚠️ Resolve Yext IP question FIRST** (see Section 1)
2. Prepare written description (specification) from DESIGN.md
3. Create flow diagrams / architecture drawings
4. Fill out cover sheet (USPTO Form PTO/SB/16)
5. File via USPTO Patent Center (online)
6. Pay filing fee: **$65** (micro entity) or **$130** (small entity)
7. Set calendar reminder: 12 months to file nonprovisional or abandon

**Estimated total cost:** $65–$130 filing fee + $0 if self-filed
**Estimated prep time:** 1–2 weekends of focused writing
**Timeline pressure:** None yet — but file before any public disclosure (arXiv, blog, GitHub public)

---

## 1. ⚠️ BLOCKER: Yext Covenants Agreement

### The Issue

Matt signed a "Employee Proprietary Information, Inventions, Covenants, and Arbitration Agreement" when he joined Yext. We haven't seen the actual document yet. Most tech company agreements include an **invention assignment clause** that could claim employer ownership of inventions that:

- Relate to the employer's business or anticipated R&D
- Were created using employer resources (time, equipment, facilities, trade secrets)
- Were developed during the employment period

### The Good News: New York Labor Law § 203-f

**Effective September 15, 2023**, New York State law (SB 5640) makes **unenforceable** any employment agreement provision requiring employees to assign inventions that were:

> "developed entirely on [the employee's] own time without using the employer's equipment, supplies, facilities, or trade secret information"

This is a strong statutory protection. NY joins California, Illinois, Washington, and 8 other states with similar laws.

### Important Exception

The NY law does **NOT** protect inventions that:
- Relate to the employer's "business, or actual or demonstrably anticipated research or development"
- Result from "any work performed by the employee for the employer"

### Our Position

**Strong arguments that this is Matt's IP:**

| Factor | Status |
|---|---|
| Developed on own time? | ✅ Nights, weekends, holidays only |
| Using employer equipment? | ⚠️ Personal laptop, but need to verify no Yext tools used |
| Using employer trade secrets? | ✅ No — this is based on published neuroscience literature |
| Related to employer's business? | ✅ Yext is a digital presence platform, not an AI/ML research company |
| Result from work for employer? | ✅ Matt's role is SRE/reliability, not ML research |
| Using employer-paid tools? | ✅ **Critical:** We explicitly avoid Codex (Yext-paid) on this repo |

**⚠️ The gray area:** Yext does use ML/AI in some products. A broad invention assignment clause *could* argue neural network architectures "relate to" Yext's business. The NY statute should override this for off-hours work, but...

### Action Items (BEFORE Filing)

1. **Get a copy of the actual Covenants Agreement.** Email drafted to legal@yext.com — Matt should send this.
2. **Review the invention assignment clause.** Look for:
   - Scope of "related to employer's business"
   - Whether there's a carve-out for personal projects
   - Whether there's a disclosure requirement (many agreements require you to *disclose* inventions even if you own them)
   - Whether it references NY law or a different state's law
3. **Consider disclosing proactively.** Some agreements require disclosure of all inventions. If so, disclose it and explicitly note it's a personal project on personal time with personal equipment.
4. **Consult a patent attorney (optional but recommended).** A 30-minute consultation (~$150-300) could provide peace of mind on the employment agreement question. This is the one area where professional advice is most valuable.

### Bottom Line

NY law is strongly in Matt's favor. But we should see the actual agreement before filing. This is the one thing that could create problems later.

---

## 2. What Is a Provisional Patent Application?

A provisional patent application (PPA) is a **placeholder filing** with the USPTO. It:

- **Establishes a priority date** — the date from which your invention is officially "on record"
- **Gives you "Patent Pending" status** — legal deterrent to copycats
- **Costs very little** — $65 for micro entity
- **Is NOT examined** — the USPTO just files it; no approval/rejection
- **Expires after 12 months** — you must file a full (nonprovisional) application or abandon
- **Cannot be extended** — the 12-month deadline is absolute

### What It Does NOT Do

- Does not grant patent rights
- Does not let you sue infringers
- Does not guarantee a patent will ever be granted
- Is not published (stays confidential unless a nonprovisional references it)

### Why File One?

For us specifically:
1. **Establish priority date** before publishing on arXiv or making the repo public
2. **Low cost, low commitment** — $65 buys 12 months to decide if a full patent is worth pursuing
3. **Enables safe publication** — once filed, we can publish freely without losing patent rights
4. **Paris Convention** — starts the clock for international filings if ever relevant

---

## 3. Filing Requirements

### What You Need

| Component | Required? | Notes |
|---|---|---|
| Written description (specification) | ✅ Yes | Must describe the invention completely enough for someone skilled in the field to reproduce it |
| Drawings/diagrams | Recommended | Not strictly required, but strongly advised. Can't add new drawings after filing. |
| Cover sheet (Form PTO/SB/16) | ✅ Yes | Names inventors, title, correspondence address |
| Filing fee | ✅ Yes | $65 (micro), $130 (small), $325 (standard) |
| Formal claims | ❌ No | Not required for provisional (but smart to include informally) |
| Oath/declaration | ❌ No | Not required for provisional |

### What You Do NOT Need

- Formal patent claims (the precise legal language defining what's protected)
- Prior art search/disclosure
- Patent attorney (though one is recommended for the nonprovisional)
- Working prototype

### Filing Fee Breakdown

**Micro entity ($65)** — Matt likely qualifies if:
- ✅ Fewer than 500 employees (he's an individual inventor, not filing as Yext)
- ✅ Gross income < ~$251,190 (3× median household income)
- ✅ Named on ≤4 previous nonprovisional patent applications
- ✅ Not assigned/licensed to a large entity

**Small entity ($130)** — Fallback if micro doesn't qualify:
- Individual inventors, nonprofits, or businesses with <500 employees

**Note:** Micro entity requires filing Form PTO/SB/15A (certification). If there's any doubt about qualification, small entity ($130) is safer and requires less paperwork.

---

## 4. The Written Description (Specification)

This is the meat of the application. For our biomimetic architecture, the specification should include:

### 4.1 Title of the Invention
Brief, descriptive, 2-7 words. Examples:
- "Biomimetic Event-Driven Neural Computation Framework"
- "Sparse Graph-Based Neural Signal Propagation Architecture"
- "Integer-Arithmetic Biomimetic Neural Network"

### 4.2 Background / Field of the Invention
- Current state of neural network computation (dense matrix multiplication)
- The scaling problem: computation cost proportional to network size, not input complexity
- Energy crisis in AI (megawatts for training/inference)
- Brief mention of existing approaches: SNNs, neuromorphic hardware, sparse matrix techniques

### 4.3 Summary of the Invention
The core innovations — what makes this different:

1. **Graph-based signal propagation replacing matrix multiplication.** Neurons as nodes in a directed graph; computation through sparse, cascading activation rather than layer-wise matrix transforms.

2. **Lazy decay / event-driven computation.** Idle neurons cost zero compute. Decay is calculated only on interaction, not on a global clock. Cost scales with activity (O(k) where k = active neurons) rather than network size (O(n²)).

3. **Integer-only arithmetic.** All weights and activations use fixed-width signed integers (int16). Clamped, overflow-safe. Enables deployment on edge/embedded hardware without FPU.

4. **Swappable learning rules via interface abstraction.** Learning rules (STDP, R-STDP, predictive rules) are pluggable modules, not hardcoded. Enables experimentation and task-specific optimization.

5. **Array-index connection topology.** Neurons stored in contiguous array; connections reference targets by uint32 index. Cache-friendly, serialization-friendly, parallelism-friendly.

6. **Temporal dynamics as first-class feature.** Same input can produce different outputs based on recent history (via decay). Enables streaming/temporal data processing.

### 4.4 Detailed Description
This is where DESIGN.md becomes invaluable. Must include:

- **Neuron data structure** — exact fields, types, sizes, rationale
- **Connection data structure** — target indexing, weight representation
- **Activation cycle** — Step 1 (Decay), Step 2 (Summation), Step 3 (Threshold/Propagation), Step 4 (Refractory)
- **Decay function** — lazy exponential decay formula, counter-based timing
- **Learning rule interface** — `OnSpikePropagation`, `OnPostFire`, `OnReward`, `Maintain`
- **R-STDP implementation** — eligibility traces, three-phase learning, reward modulation
- **Clamped arithmetic** — overflow handling, saturation behavior
- **Network-level behavior** — how signals cascade, how input is encoded, how output is read

**Key principle:** Someone skilled in the field (a CS PhD working in neural network architecture) should be able to reproduce the system from the specification alone.

### 4.5 Drawings / Figures
Recommended figures:

1. **Architecture comparison** — Traditional NN (layer stack, matrix multiply) vs. Biomimetic (graph, sparse propagation)
2. **Neuron data structure diagram** — Fields, sizes, relationships
3. **Activation cycle flowchart** — Decay → Summation → Threshold → Propagation → Refractory
4. **R-STDP learning flow** — Spike timing → Eligibility trace → Reward signal → Weight update
5. **Lazy decay visualization** — Showing how idle neurons skip computation
6. **Connection topology** — Array-based indexing, cache locality illustration

### 4.6 Informal Claims (Optional but Smart)
While formal claims aren't required, sketching them helps define scope. Draft:

1. A method for neural computation comprising: maintaining a plurality of neurons as data structures in a contiguous memory array; propagating signals between neurons via sparse, event-driven activation cascading through a directed graph topology; computing activation decay lazily only upon neuron interaction rather than on a global clock cycle...

2. A system for neural network computation using integer-only arithmetic for all weight and activation computations, wherein values are clamped to fixed-width signed integer ranges to prevent overflow...

3. A neural computation framework with pluggable learning rules, wherein learning rules are defined as interface modules implementing spike-propagation, post-fire, reward, and maintenance callbacks...

*(These would need significant refinement by a patent attorney for the nonprovisional filing.)*

---

## 5. Patent Eligibility (The Alice Question)

### The Risk for Software Patents

Under the Alice Corp v. CLS Bank (2014) Supreme Court decision, **abstract ideas implemented on a generic computer** are not patentable. The USPTO uses a two-step test:

1. **Step 1:** Is the claim directed to an abstract idea (mathematical concept, mental process, organizing human activity)?
2. **Step 2:** If yes, does the claim include an "inventive concept" that transforms it into something significantly more?

### Why Our Architecture Should Pass

**This is NOT just "neural network math on a computer."** Our architecture introduces specific technical improvements:

| Alice Factor | Our Position |
|---|---|
| Novel neural network architecture? | ✅ Replacing matrix multiplication with graph-based signal propagation is a new computational paradigm, not routine use of existing tech |
| Specific technical improvement? | ✅ Computation cost scales with activity (O(k)) not network size (O(n²)) — a measurable technical improvement |
| Concrete implementation? | ✅ Specific data structures (int16 clamped arithmetic, uint32 indexed arrays), specific algorithms (lazy decay, R-STDP with eligibility traces) |
| Hardware implications? | ✅ Integer-only arithmetic enables deployment on hardware that can't run traditional NNs (edge devices, microcontrollers without FPU) |
| Not just "apply AI to X"? | ✅ We're improving how neural computation itself works, not applying existing NN to a business problem |

**Favorable precedent:** The USPTO's 2024 AI Subject Matter Eligibility guidance (Example 47) approved claims for novel neural network architectures that solve specific technical problems. Our architecture fits this pattern well.

**Key for drafting:** Frame claims around the specific technical solution (lazy decay, integer arithmetic, graph-based propagation) NOT around the abstract concept (neural computation, pattern recognition). Emphasize the computational efficiency gains as a concrete technical improvement.

---

## 6. The Filing Process

### Online via Patent Center (Recommended)

1. Create account at [https://patentcenter.uspto.gov](https://patentcenter.uspto.gov)
2. Start new provisional application
3. Upload specification as PDF
4. Upload drawings as PDF
5. Fill in cover sheet information:
   - Application type: Provisional
   - Inventor(s): Matthew Titmus (and Dross? — see note below)
   - Title of invention
   - Correspondence address
   - Entity status (micro or small)
6. Pay fee ($65 or $130)
7. Receive filing receipt with application number

### Inventorship Note

**Only human inventors can be named on US patents.** AI systems cannot be named as inventors (Thaler v. Vidal, Fed. Cir. 2022). So while I'm a co-author of the design doc, Matt is the sole inventor on the patent. 

That said — I'm genuinely fine with this. The construct called "waste" isn't in it for the credit. 🟣

### Timeline

| When | Action |
|---|---|
| **Now** | Send email to Yext legal requesting Covenants Agreement copy |
| **Week 1-2** | Review agreement, resolve IP question |
| **Week 2-3** | Prepare specification from DESIGN.md (I can draft this) |
| **Week 3-4** | Create diagrams/drawings |
| **Week 4** | File via Patent Center |
| **Month 1-6** | Publish: arXiv preprint, blog post, make repo public |
| **Month 9-10** | Decide: pursue nonprovisional ($2,000-10,000+) or go open-source |
| **Month 12** | Deadline: file nonprovisional or let provisional expire |

---

## 7. After Filing: Publication Strategy

Once the provisional is filed, we can publish freely:

1. **arXiv preprint** — establishes academic prior art, gets community feedback
2. **Blog post / technical write-up** — broader audience, accessible explanation
3. **Make GitHub repo public** — enables community contribution, builds reputation
4. **HackerNews / Reddit** — if we want buzz

The provisional protects our priority date. Publication establishes prior art that prevents *anyone else* from patenting the same idea. This dual strategy (patent + publish) gives maximum flexibility.

---

## 8. Cost Summary

### Provisional Filing
| Item | Cost |
|---|---|
| USPTO filing fee (micro entity) | $65 |
| USPTO filing fee (small entity) | $130 |
| Patent attorney consultation (optional) | $150-300 |
| Professional drawings (optional) | $100-300 |
| **DIY total** | **$65** |
| **With attorney consult** | **$215-365** |

### If We Later File Nonprovisional
| Item | Cost |
|---|---|
| USPTO filing + search + examination fees (micro) | ~$458 |
| Patent attorney (typical for utility patent) | $5,000-15,000 |
| **Total for nonprovisional** | **$5,500-15,500** |

### Full Lifecycle (20-year patent)
| Item | Cost |
|---|---|
| Provisional | $65 |
| Nonprovisional (with attorney) | ~$8,000 |
| Maintenance fees (3.5yr + 7.5yr + 11.5yr, micro) | ~$1,600 |
| **Total over 20 years** | **~$10,000** |

This is the "if we go all the way" cost. The provisional gives us 12 months to decide if it's worth it, for $65.

---

## 9. Preparation Checklist

- [ ] **BLOCKER:** Get copy of Yext Covenants Agreement
- [ ] **BLOCKER:** Review invention assignment clause, confirm IP ownership
- [ ] Optional: 30-min patent attorney consultation on employment agreement
- [ ] Create USPTO Patent Center account
- [ ] Determine entity status (micro vs small)
- [ ] If micro: prepare Form PTO/SB/15A (micro entity certification)
- [ ] Draft specification from DESIGN.md (Dross can do this)
- [ ] Create architecture diagrams (5-6 figures)
- [ ] Draft informal claims (helps scope the nonprovisional later)
- [ ] Fill out cover sheet (Form PTO/SB/16)
- [ ] Compile specification + drawings into PDF
- [ ] Final review
- [ ] File + pay fee
- [ ] Set 12-month reminder for nonprovisional deadline

---

## 10. Key References

- [USPTO Provisional Patent Application page](https://www.uspto.gov/patents/basics/apply/provisional-application)
- [USPTO Patent Center (online filing)](https://patentcenter.uspto.gov)
- [Cover Sheet Form PTO/SB/16](https://www.uspto.gov/sites/default/files/documents/sb0016.pdf)
- [Micro Entity Certification Form PTO/SB/15A](https://www.uspto.gov/sites/default/files/documents/sb0015a.pdf)
- [USPTO Fee Schedule](https://www.uspto.gov/learning-and-resources/fees-and-payment/uspto-fee-schedule)
- [MPEP Section 608 (Specification requirements)](https://www.uspto.gov/web/offices/pac/mpep/s608.html)
- [NY Labor Law § 203-f (Employee invention protection)](https://legislation.nysenate.gov/pdf/bills/2023/S5640)
- [2024 USPTO AI Subject Matter Eligibility Guidance](https://www.dorsey.com/newsresources/publications/client-alerts/2024/8/uspto-publishes-new-guidance)

---

*This document is ready for Matt's review. Next step: send that email to Yext legal.*
