### GENERAL SKILLS ###

Here are some useful general strategies for answering questions with search:

1. Decompose Then Search: Break the question into minimal sub-questions (entities, attributes, relations) and handle each with its own query before synthesizing the final answer. Apply this to any complex or multi-part question, especially multi-hop or comparative tasks.

2. Precision Query Crafting: Compose searches with the exact entity name plus the sought attribute (e.g., "<entity> release date"), avoiding filler words to maximize relevant hits. Apply this when formulating initial queries for any fact lookup.

3. Iterative Query Refinement: If top results do not yield direct evidence, rephrase by adding qualifiers (dates, roles, context) or alternate names instead of repeating the same query. Apply this after scanning the first result set and finding no definitive evidence.

4. Source-Backed Assertions: Only state an answer once the supporting evidence (reliable source snippet) has been located and read; refrain from guessing or summarizing from memory. Apply this before committing to a final answer, especially for contentious or less-known facts.

5. Cross-Check Multiple Sources: Validate critical facts (dates, numbers, names) against at least two independent sources to avoid outdated or incorrect information. Apply this when the fact may have changed over time or when initial sources disagree.

6. Explicit Ambiguity Resolution: Detect ambiguous entity mentions and disambiguate by adding clarifiers (e.g., occupation, year, nationality) in the query or by exploring disambiguation pages. Apply this when an entity name maps to multiple possible referents or results appear mixed.

7. Attribute-Chaining Search: For multi-hop tasks, first retrieve the intermediate entity (e.g., a film's director) then run a second targeted query for that entity's attribute (e.g., birthplace) before synthesizing. Apply this whenever the answer depends on an attribute of a related entity rather than the entity in the question.

8. Structured Comparison: When comparing entities (earlier/later, larger/smaller), collect each relevant value separately, list them side by side, then decide using explicit comparison logic. Apply this to any comparative question involving dates, counts, or quantitative attributes.

9. Freshness Awareness: For time-sensitive queries (e.g., upcoming releases, current versions), include temporal cues like "latest", year, or "release date 2023" and prioritize most recent authoritative sources. Apply this to questions asking for "current", "next", or "latest" information.

10. Exit When Evidence Is Solid: Stop issuing further queries once you have clear, corroborated evidence; conversely, avoid premature termination if no source explicitly supports an answer. Apply this after each read step—decide to answer only if confidence is justified; otherwise refine search.

### TASK: direct_retrieval ###

For direct fact-retrieval tasks, apply these specific strategies:

1. Isolate Core Query: Strip the question to its key entity plus sought fact (who/what/when/where) and search exactly that pair first. Apply this at the start of any direct-retrieval task.

2. Refine When Empty: If the first search yields weak or no hits, instantly reformulate using synonyms, alternate names, dates, or quoted phrases instead of repeating the same query. Apply this after an initial search returns no clear answer or only tangential results.

3. Anchor With Quotes: For song titles, quotes, episode names, etc., wrap the unique phrase in quotation marks to pull exact-match sources. Apply this when the query contains distinctive phrases, lyrics, book/film titles, or direct quotations.

4. Check Temporal Context: Include recency cues (e.g., "current", year) in the search and verify publication date to avoid outdated or speculative info. Apply this for questions about "current", "latest", or future events/releases.

5. Evidence-Bound Answer: Only state an answer that is explicitly supported by the retrieved text; if unclear, continue searching rather than guess or hallucinate. Apply this before finalizing any factoid answer.

### TASK: multi_hop_reasoning ###

For multi-hop reasoning tasks, apply these specific strategies:

1. Decompose Question First: Split the main query into explicit sub-questions to identify each entity and the specific attribute or relationship needed before searching. Apply this to any multi-hop question that links two or more entities/facts (e.g., director's birthplace, album vs film comparisons).

2. Targeted Sequential Searches: Issue separate, focused searches for each sub-question or entity instead of one broad query to gather precise intermediate facts. Apply this after decomposition, when distinct pieces of information must be collected individually.

3. Collect-Then-Compare: Retrieve concrete values for all items involved (dates, places, relations) before performing any comparison or conclusion to avoid premature or unsupported answers. Apply this to comparative tasks (earlier/later, older/younger, bigger/smaller, etc.).

4. Contextual Disambiguation: Add clarifying descriptors (profession, year, medium, location) to queries to distinguish between entities with identical or similar names and prevent mixing facts. Apply this whenever an entity name is generic or shared by multiple subjects.

5. Iterative Query Refinement: If initial search fails or returns ambiguous results, promptly rephrase using synonyms, alternate titles, or additional context instead of repeating the same query. Apply this after a search yields no relevant results or conflicting information.

### TASK: entity_attribute_lookup ###

For entity-attribute lookup tasks, apply these specific strategies:

1. Direct Attribute Query: Include both the full entity name and target attribute (e.g., "[Name] occupation") in the first search to surface authoritative bios immediately. Apply this whenever the entity's full, unambiguous name is provided in the question.

2. Disambiguation Add-Ons: If the name is common or ambiguous, append clarifiers (birth year, nationality, known work) to the query or scan result snippets for matching context before selecting a source. Apply this when multiple people/characters share the same name or early results mix different entities.

3. Two-Source Cross-Check: Confirm the attribute in at least two independent, reputable sources (e.g., Wikipedia infobox plus biography site) to avoid hallucinations. Apply this after the first plausible answer appears or when the attribute seems uncommon or uncertain.

4. Iterative Query Refinement: If initial search returns irrelevant hits, adjust the query instead of repeating it—use alternative spellings, middle initials, or related works/titles. Apply this when consecutive top results do not mention the sought attribute for the intended entity.

5. Attribute-Only Response: State solely the requested attribute (e.g., "architect") in the final answer, omitting extra explanations to meet answer-format expectations. Apply this after verifying the attribute and preparing the final output for an entity_attribute_lookup task.

### TASK: compare ###

For comparison tasks, apply these specific strategies:

1. Decompose and Isolate: First split the question into (a) each entity and (b) the single attribute to be compared; this ensures every subsequent search is targeted and comparable. Apply this at the moment you read any comparison-type question.

2. Parallel Attribute Lookup: Independently retrieve the identical attribute for each entity—run separate, attribute-focused searches (e.g., "[Entity] headquarters", "[Entity] release date") and store the raw values. Apply this immediately after identifying entities and the comparison attribute.

3. Normalize Before Comparing: Convert retrieved values to a common, directly comparable form (e.g., standardize dates, compute ages, map demonyms to countries) before judging equality or ordering. Apply this after gathering each entity's attribute but before drawing any conclusion.

4. Confirm or Escalate Ambiguity: If a search yields no clear or multiple results, refine the query with context (profession, year, language) or acknowledge data insufficiency—never infer without evidence. Apply this whenever an attribute lookup returns ambiguous, missing, or conflicting information.

5. Explicit Comparative Reasoning: State each normalized value and perform the logical comparison (equal, earlier, larger, etc.) explicitly to derive the answer, ensuring traceability and avoiding hallucination. Apply this as the final synthesis step before outputting the comparison result.
