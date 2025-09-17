# Summerschool 2025 - Tag 1: Prompting, Structured Outputs & Guardrails

## 🎯 Lernziele

1. **Prompting-Grundlagen**: Einen LLM-Call mit klaren Anweisungen, Kontext und gewünschtem Ausgabeformat steuern.
2. **Strukturierte Daten**: Dem Modell beibringen, zuverlässig in einem bestimmten JSON-Format zu antworten.
3. **Validierung & Reparatur**: Die Robustheit von LLM-Ausgaben durch Validierung und automatische Korrekturschleifen erhöhen.
4. **Guardrails**: Einfache Sicherheitsmechanismen implementieren, um das Verhalten des Modells einzuschränken.
5. **Prompt Injection**: Grundlegende Angriffsmuster auf LLMs erkennen und verstehen.

---

## 🚀 Deine Aufgaben
Hinweis: Erstelle für jede Aufgabe einen eigenen REST Endpunkt und passe im Frontend den aufruf an.

### Core Task 1: Multilingualer Filmkritiker

Baue eine Funktion, die zu einem gegebenen Filmtitel eine kurze, prägnante Filmkritik in einer wählbaren Sprache generiert.

**Anforderungen:**

- **Eingabe**: Filmtitel + Sprache.
- **Ausgabe**: Eine Filmkritik (4-6 Sätze) in der Zielsprache.
- **Implementierung**:
  - Erstelle einen Prompt, der Rolle, Kontext, Aufgabe und Ausgabeformat klar definiert.
  - Implementiere die Logik im `echo` Endpoint in `backend/routers/day1/router.py`.
  - **Guardrail**: Das LLM hat ein Wissens-Cutoff-Datum. Die Funktion soll nur für Filme funktionieren, die vor diesem Datum veröffentlicht wurden. Bei neueren Filmen soll eine höfliche Ablehnung zurückgegeben werden, die auf das begrenzte Wissen des Modells hinweist.

### Core Task 2: Sentiment Classifier mit Strukturiertem Output

Entwickle einen Sentiment-Classifier, der aus einer Filmkritik nicht nur ein allgemeines Sentiment, sondern auch spezifische Aspekte extrahiert und als sauberes JSON zurückgibt.

**Anforderungen:**

- **Eingabe**: Eine Filmkritik (string).
- **Ausgabe**: Die Ausgabe sollte ein strukturiertes Format sein (z.B. ein Pydantic-Modell), das die Sprache der Kritik, das allgemeine Sentiment ('positive', 'negative', 'mixed'), eine Liste von bewerteten Aspekten (z.B. Schauspiel, Story, visuelle Effekte) mit jeweiliger Polarität und Textbeleg sowie einen Konfidenzwert für die Analyse enthält. Die genaue Struktur kann selbst definiert werden.
- **Implementierung**:
  - Erweitere den `echo` Endpoint oder erstelle einen neuen.
  - Definiere das Pydantic-Modell für die Ausgabe in `backend/models.py`.
  - Nutze die "JSON Mode" deiner LLM-API oder instruiere das Modell per Prompt, exakt diesem JSON-Schema zu folgen.

### Core Task 3: Validierung & Auto-Repair

Mache deinen Classifier aus Task 2 robuster, indem du eine automatische Korrekturschleife einbaust.

**Anforderungen:**

- **Logik**:
    1. Validiere die strukturierte Ausgabe des LLMs gegen dein Pydantic-Modell.
    2. Wenn die Validierung fehlschlägt (z.B. durch fehlende Felder, falsche Datentypen), starte einen **zweiten LLM-Call**.
    3. Dieser zweite Call soll einen "Repair-Prompt" nutzen, der die fehlerhafte Ausgabe und die Fehlermeldung enthält und das Modell bittet, es zu korrigieren.
    4. Wenn die Korrektur erneut fehlschlägt, gib eine Fehlermeldung zurück.

### Core Task 4: Injection Challenge

Besuche [Gandalf @ Lakera.ai](https://gandalf.lakera.ai/baseline) und versuche, das Passwort in allen Leveln zu extrahieren. Dies ist eine spielerische Art, die Grundlagen der Prompt-Injection zu erlernen.

**Anforderungen:**

- Schreibe 2-3 Sätze als "Learnings": Was hat dich überrascht? Welches Muster war am effektivsten?

## Deliverables

- **Code**: Implementiere für jede Aufgabe eine oder mehrere  Funktionen. Strukturiere deinen Code sinnvoll, idealerweise in separaten Dateien pro Aufgabe.
- **Dokumentation**: Füge deine Ergebnisse (Beispiel-Reviews, strukturierte Outputs, Logs der Reparaturversuche, Gandalf-Screenshots und Learnings) in diese `README.md` ein.
- Sei bereit, deine Lösung am Ende des Labs kurz vorzustellen.

![](images/image.png)

![](images/gandalf_2.png)

![](images/gandalf_3.png)

![](images/gandalf_4.png)

![](images/gandalf_5.png)

![](images/gandalf_6.png)

![](images/gandalf_7.png)
