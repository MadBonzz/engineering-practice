# Public assessment archetypes and what to borrow

This file is a design library, not a claim that every company currently uses the same process for every role. Hiring loops change by team and year.

Use **official sources** as strong evidence of assessment philosophy/format. Treat **candidate reports** as lower-confidence examples of concrete task shapes. Never reproduce proprietary or NDA-protected questions. Borrow the structure and signal.

---

## 1. Perplexity — practical, multi-part, job-relevant components

**Confidence:** HIGH — official interview portal and released former interview exercise.

Sources:

- Practical Assessment: https://interviewresources.perplexity.ai/practical-assessment/
- Hands-on Coding: https://interviewresources.perplexity.ai/hands-on-coding/
- Released Todo List practice exercise: https://interviewresources.perplexity.ai/hands-on-coding/examples/todo-list/

What the public materials show:

- Practical assessments are intended as realistic previews of technical work.
- Hands-on exercises build practical components/abstractions for systems that power Perplexity.
- Exercises are self-contained and multi-part.
- The released Todo List exercise gives starter code with classes/methods, then adds dependencies and later an LLM-facing rendering requirement.
- Earlier behavior must be preserved as the system evolves.
- Perplexity explicitly values careful requirement translation, abstractions, debugging method, technical judgment, precision, and working code.
- Their team calibrates exercises by having employees attempt them under interview conditions.

**Borrow:**

- one coherent company-like system component;
- meaningful starter classes and fixed method contracts;
- 3–4 cumulative parts;
- part-level testing;
- later requirements that change what a good earlier abstraction looks like;
- substantial written/code context that rewards careful reading;
- calibration under realistic time limits;
- evaluation based on precision + design judgment, not completion alone.

**Do not over-borrow:**

- Do not clone the Todo List question or the user's real Perplexity question.

---

## 2. CodeSignal Industry Coding Framework — progressive project under changing requirements

**Confidence:** HIGH — official CodeSignal framework/support docs.

Sources:

- ICA rules: https://support.codesignal.com/hc/en-us/articles/19116922232983-What-are-the-Industry-Coding-Assessment-ICA-rules
- Industry Coding Framework: https://codesignal.com/resource/industry-coding-framework/
- Framework technical brief PDF: https://discover.codesignal.com/rs/659-AFH-023/images/Industry-Coding-Skills-Evaluation-Framework-CodeSignal-Skills-Evaluation-Lab-Short.pdf

Official structure:

- one project-based question;
- four progressive levels;
- requirements accumulate rather than resetting;
- Level 1: basic implementation + corner cases;
- Level 2: data processing / additional operations;
- Level 3: refactoring/encapsulation to support new features;
- Level 4: extension while preserving backward compatibility;
- later levels intentionally reward reusable earlier design;
- the framework's example is a file-hosting service that grows from basic file operations to search, TTL-aware operations, and rollback.

**Borrow:**

- progressive levels;
- backward-compatibility pressure;
- requirements that make refactoring natural;
- a simple domain with increasing state complexity;
- partial progress as useful signal;
- 60–90 minute default target for the CodeSignal-like variant.

**Important adaptation:**

CodeSignal's general Industry Coding Framework intentionally excludes specialized concurrency/distributed topics. This bench is different: when the user's studied topic *is* distributed computing, concurrency, ML serving, etc., include that specialized content rather than mechanically obeying CodeSignal's generic-role exclusions.

---

## 3. DSLabs — constrained framework + invariants + failure exploration

**Confidence:** HIGH — public teaching repository supplied by the user.

Source:

- https://github.com/emichael/dslabs/tree/handout

Notable design traits:

- students inherit a framework and partially implemented classes;
- TODOs sit inside a larger architecture rather than a blank project;
- labs are staged and later parts depend on previous abstractions;
- README requirements specify protocol/system properties while leaving substantial design to the student;
- tests cover more than examples: unreliable networks, long-running workloads, memory/garbage collection, determinism, safety/liveness, and state-space search;
- some later labs require the student to define messages/timers/state themselves within the framework.

**Borrow:**

- an existing framework that constrains implementation;
- interfaces/classes that must fit together;
- staged functionality;
- invariants stronger than example output matching;
- fault injection / stateful tests when the topic warrants it;
- tests for long-run/resource behavior;
- intentionally incomplete design specification where the student must make engineering choices while still satisfying a precise external contract.

**Do not over-borrow:**

- Do not force a message-passing simulator or model checker onto unrelated topics.

---

## 4. Sourcegraph — zipped project, real workflow, follow-up discussion

**Confidence:** HIGH for historical process — official public handbook, now archived.

Sources:

- Async coding exercise: https://github.com/sourcegraph/handbook/blob/main/content/departments/engineering/dev/hiring/software-engineer-coding-exercise.md
- Pairing exercise: https://github.com/sourcegraph/handbook/blob/main/content/departments/engineering/dev/hiring/software-engineer-pairing-exercise.md

Public examples include:

- a two-hour TypeScript zipped project with instructions and unit tests focused on callbacks/asynchronous execution;
- a distribution exercise involving build/deploy workflow, Go services, local Kubernetes, and an HTTP client;
- a simple API-client pairing task;
- follow-up discussion of submitted code;
- exercises selected to be explainable, time-bound, and extensible with further features/tests.

**Borrow:**

- give the user a realistic local repository, not merely a prompt;
- keep environment reproducible;
- build tasks around job-relevant workflow rather than puzzle syntax;
- include natural extension points;
- post-assessment review should ask the student to explain choices.

---

## 5. Hudson River Trading — timed implementation, edge cases, systems fundamentals

**Confidence:** HIGH — official HRT interview guide; concrete exact questions can vary.

Source:

- https://www.hudsonrivertrading.com/hrtbeat/interview-at-hrt/

HRT says its take-home tests are timed, commonly run through coding platforms, permit language-reference resources, and explicitly warns that sample tests are not comprehensive. The guide emphasizes testing corner cases, systems-level knowledge, decomposing unfamiliar problems, and working methodically.

**Borrow:**

- public tests should not be exhaustive;
- hidden corner cases must be fair and important;
- systems tasks should reward understanding what happens under the hood;
- methodical decomposition and self-testing are part of the signal;
- implementation may be more important than polished production architecture in a short timed screen.

Candidate report (lower confidence) indicating an implementation-heavy OA rather than puzzle-only tasks:

- https://news.ycombinator.com/item?id=42576574

Use candidate reports only as inspiration for workload size/style, not as a source to recreate exact questions.

---

## 6. Sarvam AI — proof-of-work ML component with a real dataset/objective

**Confidence:** MEDIUM — widely shared candidate report, not an official Sarvam assessment specification.

Primary candidate report:

- https://www.reddit.com/r/developersIndia/comments/1u4uf70/my_interview_experience_with_sarvamai_for_ml/

Corroborating public discussion:

- https://www.linkedin.com/posts/thervivek_a-candidate-interviewed-at-sarvam-for-an-activity-7504559966494662656-n9Of

Reported task shape:

- build a Voice Activity Detector from scratch in roughly 2.5 hours;
- a small audio dataset was provided;
- architecture choice was open;
- evaluation included functional quality/accuracy, code quality, and proposed improvements that could not be completed in time.

**Borrow:**

- for genuinely ML-facing topics, provide a small local dataset/workload and objective metric;
- allow more than one valid architecture;
- evaluate both working quality and engineering quality;
- include a post-task "what would you improve with more time?" discussion in review mode;
- choose a task that resembles a real component in the domain.

**Do not over-borrow:**

- Do not make every MLE assessment a modeling contest.
- Do not assume the candidate report describes all Sarvam roles/processes.

---

## 7. Anthropic — independent assessments + progressive system implementation

**Confidence:** MIXED.

Official AI-use guidance confirms that Anthropic uses take-home assessments and generally expects candidates to complete them independently unless AI use is explicitly allowed:

- https://www.anthropic.com/candidate-ai-guidance

Concrete progressive CodeSignal details are candidate-reported / third-party, not official Anthropic specifications. Examples:

- https://www.reddit.com/r/InterviewDB/comments/1vpzcht/anthropic_codesignal_industry_coding_assessment/
- https://www.reddit.com/r/InterviewDB/comments/1wn5ed5/anthropic_codesignal_95100_then_rejected_before/
- https://www.sundeepteki.org/advice/anthropic-codesignal-assessment-guide

Repeatedly reported pattern:

- one in-memory system;
- progressive levels;
- each level extends previous behavior;
- later levels involve harder state/history/entity interactions;
- standard-library implementation rather than specialized libraries.

There are also candidate reports of practical, domain-specific live coding (for example an image-processing pipeline) and MLE take-home/research problems; treat these as anecdotal rather than canonical.

**Borrow:**

- progressive stateful systems;
- late-stage historical/state-merging complications when the concept naturally involves history;
- independent solving mode;
- do not make the assessment dependent on AI assistance.

---

## 8. Tower Research Capital — practical, non-gotcha philosophy + debugging/system depth

**Confidence:** HIGH for philosophy, MEDIUM for specific candidate tasks.

Official careers page:

- https://tower-research.com/careers/

Tower publicly says it aims for a practical process without gotcha questions or overly academic discussion.

Candidate report describing one OA variant with coding, a debugging problem, and OS/networking concepts:

- https://leetcode.com/discuss/post/5620964/Tower-Research-or-or-SDE-Intern-or-or-OA-%2B-Interview/

Another candidate report:

- https://reckadon.github.io/trc-interview-experience/

**Borrow:**

- include debugging existing code as an optional archetype;
- connect questions to systems fundamentals where relevant;
- assess implementation + diagnosis, not trick memorization;
- use realistic failure modes in a pre-existing program.

---

## 9. Dropbox / progressive multi-file engineering assessments

**Confidence:** MEDIUM — CodeSignal relationship is public; concrete historical tasks are largely candidate-reported.

CodeSignal customer story:

- https://codesignal.com/customers/dropbox/

Candidate reports:

- https://www.jointaro.com/interviews/companies/dropbox/experiences/software-engineering-intern-united-states-march-11-2025-no-offer-positive-342e811c/
- https://www.jointaro.com/interviews/companies/dropbox/experiences/software-engineer-united-states-october-11-2023-no-offer-negative-78392bff/
- https://www.jointaro.com/interviews/companies/dropbox/experiences/software-engineer-united-states-november-1-2020-no-offer-neutral-08444655/

Reported patterns include progressive class/function requirements, file-system/in-memory-system tasks, and multi-file scaffolding with tests.

**Borrow:**

- realistic multi-file scaffolding;
- enough existing code that navigation matters;
- progressive requirements around a single stateful component.

**Lesson from negative candidate feedback:**

Do **not** overload a 60-minute task with so many irrelevant hidden files or mocked systems that code navigation becomes the dominant challenge. Existing code should create realism, not needless cognitive tax.

---

## 10. Datadog — small production-style service/monitor + follow-up

**Confidence:** MIXED — Datadog officially discusses job-relevant take-homes; the HTTP log-monitor detail is an older candidate report.

Official current guidance:

- https://careers.datadoghq.com/candidate-experience/interviewing-at-datadog-ai-guidelines/

Historical candidate report:

- https://www.jointaro.com/interviews/companies/datadog/experiences/software-engineer-new-york-ny-february-1-2019-no-offer-negative-f63384bc/

Reported take-home: design an HTTP log monitor.

**Borrow:**

- observability/testing topics can be framed as a small production component consuming a stream and maintaining state/alerts;
- expect the student to defend choices in review mode;
- job relevance matters more than puzzle novelty.

---

## 11. Ramp — practical parsing/system evolution

**Confidence:** MEDIUM — candidate reports.

Examples:

- https://www.jointaro.com/interviews/companies/ramp/experiences/software-engineer-new-york-ny-may-1-2024-no-offer-positive-1914107f/
- https://www.reddit.com/r/InterviewCoderHQ/comments/1t7u9ub/ramp_software_engineer_loop_full_breakdown_of/

Reported tasks include data cleaning/parsing and progressive small systems with new functionality added across levels.

**Borrow:**

- parsing/validation can be a legitimate first stage before the deeper system behavior;
- domain objects can evolve through multiple requirements;
- use realistic data formats instead of abstract integer arrays when appropriate.

---

## 12. Jane Street TDOE — adaptation to unfamiliar concepts and sanity checking

**Confidence:** HIGH — official role page, but for a different role family than MLE/SWE.

Source:

- https://www.janestreet.com/join-jane-street/tdoe/

Jane Street describes a relatively short take-home that tests attention to detail and adaptation to an unfamiliar concept, followed by problem solving where candidates are encouraged to sanity-check answers.

**Borrow:**

- occasionally make the assessment require the user to read and operationalize a small unfamiliar local specification/API;
- reward sanity checks and disciplined verification;
- do not require prior niche domain knowledge when the point is learning/adaptation.

---

## 13. OpenAI — skills-based assessment and engineering quality dimensions

**Confidence:** HIGH — official interview guide.

Source:

- https://openai.com/interview-guide/

OpenAI states that skills-based assessment formats vary by team and may include pair coding, take-home projects, and technical tests. For engineering interviews it highlights well-designed solutions, high-quality code, performance, test coverage, communication, and reasoning.

**Borrow:**

- assessment design may vary with the concept instead of forcing one template;
- include design quality, correctness, appropriate performance, and testing where relevant;
- post-assessment review should examine reasoning and decisions, not only pass/fail.

---

# Synthesis: default house style for this bench

When no topic-specific reason suggests otherwise, combine the best properties above:

1. **Perplexity:** practical component from a realistic system + multi-part progression.
2. **CodeSignal ICA:** cumulative requirements and backward-compatible refactoring pressure.
3. **DSLabs:** existing framework plus invariant/stateful testing.
4. **HRT:** incomplete public tests and important corner cases; methodical implementation.
5. **Sourcegraph:** self-contained local repository and follow-up code discussion.
6. **Sarvam:** for ML-facing tasks, real local data/workload and a functional quality metric.
7. **Tower:** debugging/system understanding when appropriate.
8. **Dropbox lesson:** meaningful scaffolding, but avoid navigation overload.

The goal is not to imitate company branding. The goal is to reproduce the *kind of engineering signal* these formats seek.
