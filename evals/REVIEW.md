# Revisar el golden set

`golden_set.json` es la única parte del proyecto cuya corrección no se puede
derivar del corpus. Todo lo demás se mide contra él, así que una entrada que
nadie verificó es peor que ninguna: produce un número que parece una medición y
no lo es.

Por eso cada entrada tiene `status`. Los borradores están excluidos del eval
hasta que una persona los cambia a `reviewed`.

## Estado

```bash
uv run python scripts/golden_status.py            # avance y validación
uv run python scripts/golden_status.py --drafts   # listar pendientes
```

El comando sale con código distinto de cero si alguna cita no existe en el
corpus, así que sirve como gate de CI.

## Qué revisar en cada entrada

Abrí `evals/golden_set.json` y para cada borrador:

1. **Leé el artículo citado.** No revises la respuesta contra tu memoria — abrí
   la norma. El modelo redactó lo que cree que dice el artículo, y donde se
   equivoca sutilmente (confunde "deberá" con "podrá", se saltea una excepción)
   el borrador se lee perfecto y está mal. Ese es el error más difícil de
   detectar leyendo por encima.

2. **Reescribí la pregunta con tus palabras.** El borrador se generó *a partir
   del* artículo, así que comparte su vocabulario y el retrieval la encuentra
   demasiado fácil. Si dejás la redacción original, el recall te va a dar
   inflado y no vas a saber por qué. Preguntala como se la preguntarías a un
   colega.

3. **Corregí `expected_answer`** para que diga solo lo que el artículo
   sostiene. Si el artículo no alcanza, o la pregunta no vale la pena, borrá la
   entrada — 30 preguntas buenas valen más que 60 mediocres.

4. **En las trampas (`kind: "trap"`), confirmá que de verdad no esté en el
   corpus.** Buscala antes de darla por buena:
   ```bash
   uv run python scripts/search.py "el tema de la trampa"
   ```
   Una trampa que en realidad sí está en el corpus penaliza al sistema por
   responder bien.

5. **En las de síntesis, verificá que hagan falta los dos artículos.** Si se
   responde con uno solo, no es una pregunta de síntesis: cambiala a `factual`
   y dejá una sola cita.

6. **Cambiá `status` a `"reviewed"`** y vaciá `notes`.

## Agregar preguntas propias

Las mejores del set van a ser las que escribas vos de memoria: las que te
preguntaron de verdad en el trabajo. Copiá el formato de cualquier entrada, poné
un `id` nuevo y `status: "reviewed"`. `golden_status.py` valida que las citas
existan.

## Generar más borradores

```bash
uv run python scripts/draft_questions.py --n 20              # desde artículos
uv run python scripts/draft_questions.py --synthesis 8       # comparativas
uv run python scripts/draft_questions.py --traps 10          # trampas
```

Se agregan al archivo existente sin pisar lo ya revisado.
