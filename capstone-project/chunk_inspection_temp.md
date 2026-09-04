# Inspección de Chunks Generados

El archivo `chunks.csv` (y su versión `chunks.json`) contiene nuestra base de datos estructurada. Cada fila representa un "chunk" (un fragmento de texto de un artículo).

## 📊 Estructura de Columnas (Metadatos)

Cada fragmento tiene exactamente **11 columnas** que garantizan que nunca perdamos el contexto de dónde salió la información:

1. **`chunk_id`**: Identificador único (ej. `PMID_42488699_c01`).
2. **`pmid`**: ID de PubMed del artículo original.
3. **`title`**: Título completo del artículo.
4. **`authors`**: Autores del estudio.
5. **`category_name`**: La categoría deportiva/fisiológica (ej. *RED-S & Low Energy Availability*).
6. **`publication_date`**: Año de publicación (ej. *2026*).
7. **`source_type`**: Origen del texto (en nuestro caso, `fulltext_xml`).
8. **`chunk_index`**: El número de orden de este fragmento dentro del artículo original (1, 2, 3...).
9. **`content`**: **EL TEXTO EN SÍ** (Lo que leerá el LLM).
10. **`word_count`**: Cantidad de palabras en este bloque.
11. **`char_count`**: Cantidad de caracteres (máximo ~1000).

---

## 🔍 Ejemplos Reales de Chunks (Primeros 4 fragmentos)

A continuación, inspeccionamos en detalle los primeros 4 fragmentos generados del primer artículo sobre cuidados quirúrgicos y RED-S en atletas de élite.

> [!NOTE]
> Observa cómo el `content` incluye la etiqueta `SECTION:` para darle aún más contexto al modelo de Lenguaje sobre en qué parte del artículo se encuentra.

### 📝 Chunk 1 (Abstract)
- **ID:** `PMID_42488699_c01`
- **Categoría:** RED-S & Low Energy Availability
- **Título:** Optimising gynaecological surgical care for elite female athletes: a narrative review.
- **Contenido del texto:**
  ```text
  SECTION: Abstract
  BackgroundWith the rapid growth of women's elite sport, there is an increasing need to optimise healthcare for female athletes. These athletes can present unique clinical challenges, including high physical demands, altered energy availability, increased rates of pelvic floor dysfunction, and sport-related psychological pressures. Gynaecological surgery may significantly disrupt training and competition schedules, impacting both short- and long-term performance.ObjectivesThis narrative review provides a practical, evidence-based framework to support clinicians in delivering tailored peri-operative care for female athletes undergoing gynaecological surgery.Key findingsDrawing from current literature and multidisciplinary expertise, the review outlines key considerations across the pre-, intra-, and post-operative phases. It emphasises the importance of pre-operative assessment of menstrual health, bone density, nutritional status, and psychological readiness, particula
  ```
- **Tamaño:** 118 palabras (1000 caracteres)

---

### 📝 Chunk 2 (Continuación del Abstract)
- **ID:** `PMID_42488699_c02`
- **Categoría:** RED-S & Low Energy Availability
- **Título:** Optimising gynaecological surgical care for elite female athletes: a narrative review.
- **Contenido del texto:**
  ```text
  It emphasises the importance of pre-operative assessment of menstrual health, bone density, nutritional status, and psychological readiness, particularly in athletes at risk of Relative Energy Deficiency in Sport (RED-S). Intra-operatively, surgical techniques should account for anatomical variations in lean athletes, and measures should be taken to minimise complications such as neuropathy, wound breakdown and delayed recovery. Post-operative rehabilitation requires a coordinated, multidisciplinary approach integrating physiotherapy, nutrition, pain management, and psychological support to facilitate a safe and timely return to sport.ConclusionsThis review highlights the need for athlete-specific surgical strategies that align with the physiological and performance demands of elite sport. Future studies are essential to inform sport-specific guidelines and optimise outcomes for this understudied population.
  ```
- **Tamaño:** 114 palabras (921 caracteres)

---

### 📝 Chunk 3 (Introducción)
- **ID:** `PMID_42488699_c04`
- **Categoría:** RED-S & Low Energy Availability
- **Título:** Optimising gynaecological surgical care for elite female athletes: a narrative review.
- **Contenido del texto:**
  ```text
  SECTION: Introduction
  Elite women's sport is experiencing a period of rapid growth with the number of female athletes steadily increasing over the last two decades (1). This increase is also evident in the expanding media coverage and the growing number of spectators attending women's sporting events.1 With the growing popularity of women's sport, there will be a corresponding increase in elite athletes requiring tailored and specialised gynaecological surgical care.
  Despite these trends, the field of Sports Gynaecology is still in its infancy. Whilst there is a greater understanding and appreciation of the unique physiological changes and gynaecological presentations that can affect elite athletes (2), there is a paucity of guidance or literature on managing the athlete requiring gynaecological surgery (1). Alongside the physical demands of their sport, female athletes face distinct challenges related to gynaecological health that can significantly impact their well-being and performa
  ```
- **Tamaño:** 144 palabras (1000 caracteres)

---

### 📝 Chunk 4 (Desarrollo del Texto - PFD)
- **ID:** `PMID_42488699_c10`
- **Categoría:** RED-S & Low Energy Availability
- **Título:** Optimising gynaecological surgical care for elite female athletes: a narrative review.
- **Contenido del texto:**
  ```text
  SECTION: Pelvic floor dysfunction in athletes
  Pelvic floor dysfunction (PFD) is prevalent among elite female athletes, particularly those involved in high-impact or load-bearing sports (6). A 2018 systematic review and meta-analysis reported the prevalence of urinary incontinence at 36% in this population with peak prevalence reaching 76% (7).
  The pathophysiology of PFD in athletes is multifactorial. Contributing factors include repeated elevations in intra-abdominal pressure, and over-recruitment or fatigue of the pelvic floor. Prolonged exposure to these stressors may predispose athletes to microtrauma, neuromuscular dysfunction, or compensatory patterns that eventually manifest as symptoms (8). Pre-existing PFD should therefore be actively identified before surgery, as it may influence both early post-operative recovery and the sequencing of rehabilitation. Athletes with symptoms such as urinary incontinence, pelvic heaviness, pain, or impaired load tolerance may require early pelvi
  ```
- **Tamaño:** 133 palabras (1000 caracteres)
