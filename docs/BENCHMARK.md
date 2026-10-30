# Lena story benchmark (October 2026)

Lena keeps every passage of a story and links it by the names and words it contains. There are no lorebooks or story cards to maintain. Between sessions she dreams the story into short portraits of its people, places and ideas. We tested her against two well-built alternatives on 95 questions about two long stories, with the same two local models for every system.

| System | Model | Facts (67) | Current state (15) | Inference (13) | **Total (95)** |
|---|---|---:|---:|---:|---:|
| **Lena** | Gemma 4 26B | 62 | 14 | 11.5 | **87.5** |
| RAG | Gemma 4 26B | 53 | 13.5 | 8.5 | 75 |
| Story cards | Gemma 4 26B | 32 | 9 | 4.5 | 45.5 |
| **Lena** | Gemma 4 E4B | 57.5 | 12 | 10 | **79.5** |
| RAG | Gemma 4 E4B | 49 | 11.5 | 6 | 66.5 |
| Story cards | Gemma 4 E4B | 27.5 | 9 | 2 | 38.5 |

With the small model, Lena scores more than the retrieval setup does with the large one.

## The test

- **Corpus:** two roleplay stories written with an AI narrator, 512,000 words in total. *The Hound* alone is 453,000 words, about the length of *The Lord of the Rings*. The text is not edited prose: narrator and player turns alternate, out-of-character notes are mixed in, characters go by several names, and there are typos and retcons.
- **Questions:** 67 facts from anywhere in the story, 15 about things that changed (only the latest version is right), and 13 inference questions that need the whole arc.
- **Grading:** facts and current state were scored against answer patterns, and every miss was checked by hand for correct answers worded differently. Inference was graded by hand. Partial answers got half credit.

## The three systems

| | Lena | RAG | Story cards |
|---|---|---|---|
| Source | Every story turn, tagged with its names and word stems | The same story passages | 125 cards written by the 26B from passages across both stories, plus the recent story |
| Retrieval | Tag walk plus vector search, no model calls | Vector search (the same embedder, nomic-embed-text v1.5) | A card is included when one of its keywords appears in the question or recent text |
| Context per question | up to 10,000 characters | up to 10,000 characters | up to 6,000 characters of cards plus the last 16,000 of the story |
| Prompt | Lena's own system prompt | Standard "answer from this context" | Standard story-card prompt |

All three used the same models and answer settings: up to 400 tokens, temperature 0.2, thinking off. The RAG and story-card runs were made a few days before Lena's final run, on the same corpus and questions, and they don't depend on anything Lena builds.

## Usable right after import

Dreaming is not a setup step. The table shows the 82 automatically graded questions (facts and current state) at each point of the first dream over both stories, averaged over three runs. The time is the model's inference time on one GPU up to that stage.

| Stage of the first dream | Time | Lena 26B | Lena E4B |
|---|---:|---:|---:|
| Just imported | 0 h | 64.7 | 60.0 |
| Time summaries | 0.7 h | 69.7 | 66.3 |
| First character portrait | 2.3 h | 70.0 | 64.7 |
| Main cast (10 portraits) | 8.5 h | 70.3 | 63.3 |
| All 215 portraits | 14.7 h | 72.7 | 67.7 |
| Compressed | 15 h | 72.7 | 65.3 |
| *RAG, for reference* | | *65* | *58* |

- Right after import, Lena matches the retrieval setup. Within the first hour she passes it on both models.
- Dreaming runs on its own after a set idle time (30 minutes by default; adjustable, or off). Sending a message stops it after the step in progress, and the next idle stretch carries on.
- The 15 hours is a one-time cost of importing half a million words at once. In normal use, a dream folds a day's new memories into existing portraits and takes minutes.

### Without the embedding model

Lena uses vector search, but it is not what carries her. With the embedding model switched off she can only follow the names and words that link her memories, plus her dream portraits. Same 95 questions, graded the same way:

| System | 26B | E4B |
|---|---:|---:|
| **Lena, fully dreamed** | **87.5** | **79.5** |
| **Lena, fully dreamed, no embeddings** | **85** | **78** |
| Lena, just imported, no embeddings | 73.5 | 55.5 |
| RAG | 75 | 66.5 |
| Story cards | 45.5 | 38.5 |

Fully dreamed, she loses almost nothing without embeddings and still beats RAG on both models. Fresh from import with no embeddings, she is close to RAG with the 26B and well ahead of story cards on both. Recall without embeddings takes a median of 0.14 seconds.

## Hardware

- Tesla P100 16 GB running Gemma 4 26B-A4B (Q4_0), and a GTX 1070 8 GB running Gemma 4 E4B (Q4_K_M). Both are community "uncensored" variants, used for every system.
- Recall takes a median of 0.26 seconds per question, with no model calls. All model work happens in the background dream: about 15 hours of inference on the P100 for the first dream over both stories. All times here are inference time.

## Limits

- One corpus: two stories by one author, both roleplay.
- One grader, with written rules applied the same way to every system.
- The main table is a single run per system. Three runs per stage put the run-to-run spread at about two points out of 82.
- Lena still misses some early specifics, such as a price or a password mentioned once, when the question shares few words with the passage.
