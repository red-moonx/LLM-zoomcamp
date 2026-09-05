# Synthetic Question Ideas (Target Persona)

This document saves the conceptual ideas we brainstormed for the synthetic dataset generation. 
The goal is to ensure the LLM generates **natural, colloquial, "down-to-earth" questions in English** that a 30-something female recreational athlete would ask.

## Example 1: Fasted Cardio (The "Women are not small men" contradiction)
**Concept:** Highlight how fasted cardio blunts fat oxidation in women, contrary to male-centric fitness advice.
* **Bad (Academic):** "How does the fed state impact lipid oxidation compared to the fasted state in female athletes?"
* **Good (Target Persona):** "I do fasted cardio every morning because fitness influencers say it burns more fat, but I feel super bloated and haven't lost weight. Should I keep doing it?"

## Example 2: Hormonal Contraceptives & Muscle Building
**Concept:** The impact of oral contraceptives on strength gains and metabolism.
* **Bad (Academic):** "What is the effect of exogenous sex hormones on skeletal muscle hypertrophy?"
* **Good (Target Persona):** "I've been on the pill for years and just started lifting weights to tone up. Is my birth control going to make it harder for me to build muscle?"

## Example 3: Carbohydrate Loading & Luteal Phase
**Concept:** Progesterone shifts substrate utilization, making carb-loading less effective in the luteal phase.
* **Bad (Academic):** "Does progesterone suppress gluconeogenesis during the luteal phase?"
* **Good (Target Persona):** "I'm running a half marathon this weekend but my period is due in a few days. Usually I carb-load the night before, but will that still work if I feel so sluggish and PMS-y?"

## Example 4: General Fatigue & RED-S
**Concept:** Symptoms of Low Energy Availability (LEA) in recreational athletes.
* **Bad (Academic):** "What are the early indicators of Relative Energy Deficiency in Sport?"
* **Good (Target Persona):** "I'm a 33-year-old woman and I just feel completely drained of energy in the mornings before my workouts. What could be going wrong?"

## Example 5: Caffeine & Intermittent Sports
**Concept:** Ergogenic effects of caffeine in female specific sports.
* **Bad (Academic):** "Does acute caffeine supplementation enhance intermittent sport-specific skills?"
* **Good (Target Persona):** "I play recreational tennis a couple of times a week after work. Will drinking a pre-workout or strong coffee actually make me play better, or is that just marketing?"

---

## 🏆 Gold Standard for "Round 2" (Answer Quality)
**Concept:** When generating the expert answers in Round 2, the LLM should output structured, real-world advice (not just one-liners). The following Q&A pair from Round 1 is considered the "Gold Standard" by the user and should be injected into the Round 2 Prompt as a few-shot example for the `answer` field:

**Question:** "I'm a recreational runner and I train five times a week. Should I focus more on strength training or combining it with other types of training for better results?"
**Answer:** "If you're training five times a week, consider incorporating strength training (ST) alongside your current routine. However, be aware that combining strength training with other modalities without rest (like high-intensity interval training) may lead to smaller improvements in lower-body strength compared to focusing primarily on strength training."
