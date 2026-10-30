# Project Lena — FAQ

### What exactly is Lena?
An AI system.

Originally, Lena was meant to be a storyteller that isn't dumb as a rock: something that could remember a world, hold a normal conversation about it, and write a story with you as a co-author instead of railroading you through a workflow. Somewhere along the way, that turned into a full-on personal assistant.

The simplest way to describe her now is: **Lena is a cognitive system that uses an LLM as its language interface.** The LLM isn't Lena. It's one component Lena employs.

### Why "Lena"?

It's the name of a river in Russia. Ask your copy of Lena one day if you want to know more than that. She'll be happy to answer.

### Is Lena an LLM?
No.
Lena uses an LLM, but the LLM is not the seat of authority.

### Isn't this just a harness around an LLM?
Yes, and no. Technically, yes. Lena wraps an LLM and provides an interaction layer around it. That's a perfectly reasonable description of the outer architecture. But I think it misses a more important distinction.

When I started building Lena, I had essentially no knowledge of the broader LLM ecosystem, terminology, or established methods. I wasn't trying to reproduce somebody else's architecture because I didn't know what I was supposed to be reproducing. I was trying to build the system I thought made sense.

In hindsight, Lena isn't quite a conventional LLM harness because **the system owns the seat of cognition, not the LLM.**

The LLM is employed by Lena to understand and generate language. Lena handles memory, association, temporal context, consolidation, internal processing, and deciding what information the model should have available.

I'm not giving an LLM a collection of tools and asking it to operate them. I'm building a system that uses an LLM as one of its tools.

### So is this basically RAG?

No, and I can prove it now. Unplug the embedding model entirely and Lena still scores 85 out of 95 on the story benchmark with the 26B, against 75 for a standard RAG setup that runs on nothing *but* embeddings. With embeddings she gets 87.5. The vectors are a side door, not the house.

Embeddings have a blind spot: they only catch what looks like the question. They can't catch associations. Lena's memory is associative: every memory is tagged with the names and word stems in it, and recall walks those links from memory to handle to memory, weighted by how rare each handle is and how recent the memory is. If you give her an embedding model, she takes two closest matches by meaning as extra doorways into that same web. That's all.

### Okay, but what's the actual difference between Lena and RAG?

Standard vector RAG works by matching the *shape* of text. A query comes in, the nearest chunks come out, and that's the answer's raw material.

Lena enters her memory through the closest matches too, but that's only the doorway. From those memories she crosses the nodes table to their handles, and from the handles back to other memories, whatever type they are: verbatim, gists, summaries, images. Lena's keywords aren't lookup terms. They're arbitrary symbolic handles, and their meaning lives in the relationships between them, which are continuously developed in the background through seeding, consolidation and derived memories. Add the person gist and dossier of whoever is talking, and the system can surface something because it is *associated* with what's happening, even when it isn't a semantic match for the question.

The other big difference is that context is kept to a minimum. In regular conversation it's only the last exchange. In Director Mode it's only a little of the story before, for continuity. If Lena knows something, it's almost guaranteed to be because the memory was retrieved and imposed on the model, not requested and searched for by it.

### Isn't this just MemOS?

I had no idea what MemOS was until I was about to release the alpha.

I came at this as an outsider. All the AI storytellers and adventure DMs sucked. Absolutely sucked. They couldn't remember anything. So I asked what memory really is, sat down, and wrote what I wanted to exist. Lena is the result. Only after the fact did I notice that MemOS does some of the same things.

And it does. MemOS (by MemTensor, open source under Apache-2.0, with a paid cloud version) also treats memory as a graph you can inspect and edit instead of a black-box vector store. Its local version also keeps memory in SQLite and retrieves with full-text plus vector search. It also processes memory in the background, keeps personas, and stores images. We arrived at similar ideas about retrieval, probably after the same kind of benchmarking.

But they're two very different things:

| | MemOS | Lena |
|---|---|---|
| What it is | A memory service you plug into someone else's agent | One complete, persistent companion |
| Who's in charge | The agent (an LLM with tools) runs; MemOS feeds it memory before and saves the conversation after | The controller runs; the LLM only supplies language |
| Aimed at | Better agent task performance | Remembering *you*, writing with you, being a PA |
| Self-improvement | "Skill evolution": traces become policies, world models and skills | Deliberately left out: no unattended self-reasoning loops |
| Memory | Text and images, plus research toward memory in model weights and the KV cache | Text and images, model-agnostic |
| Storytelling | None | Canon scoping, Director Mode, character dossiers |
| Interface | No face of its own: an API, SDKs, agent plugins and a memory dashboard | It is the app: a chat UI, Director Mode, bridges |
| Runs on | Their cloud with an API key, or Docker with Neo4j and Qdrant | One SQLite file and a Python process on a 2009-era box |

MemOS keeps the agent framework and makes the agent smarter. I was trying to get away from that. I believe intelligence lives beyond the weights, beyond just the model. You can supplement a lot of intelligence with architecture, memory, good retrieval, and putting the right information in the right place. There's still a limit: hand someone every textbook in the library and step-by-step instructions, and at some point they're just too stupid to follow them. But the same 2B model went from 26 to 48 of 58 on my memory benchmark just by changing what was put in front of it. Garbage in, garbage out, baby. It works the other way too.

The other big difference is who it's for. To use MemOS you pretty much need to be knee-deep in understanding what agents even are. Lena is meant to be usable by someone who can follow a README: clone it, `pip install` the requirements, start a model server or put in an API key, and talk. If you want embeddings, start one more small server. You're not sitting there fiddling and fussing.

In one line: MemOS is memory for agents. Lena is an entity whose memory is the point, and the LLM is just her voice.

### If the LLM isn't doing the thinking, what is?

It is doing that thinking, but that isn't the full picture. The LLM is required to generate gist memories from the verbatim memories, keep the person gists and dossiers current, and generate the final response. It does not decide what gets remembered or recalled: handles are assigned at seeding and recall runs without a single model call.

That is a narrow view though because the LLM does not decide what it is doing next. Just like in biological cognition the controller decides which memories are relevant and places them in front of the LLM before any reasoning even happens. Like how you think on a topic and remember 20 other things, mostly related, at the same time involuntarily. In addition because the entire memory store is linked via a "nodes" table an image can connect to a handle, which leads to a gist text memory, which in turn can be mined for more handles back through the nodes table to a new image for the LLM to reason over. Then it all gets handed to the LLM with the user's query and nothing else but the previous exchange. The context window is treated as working memory and does not contain conversation history.

### Why not just put all of this into the system prompt?

I mean I guess I am. Ultimately the controller is searching a relational database and building a very clever prompt on the fly for each individual query. so, dynamic system prompting?

### Why use a small model instead of a much larger one?

When all you have is a hammer, every problem is a nail. I don't have a hammer. Seriously though? I want to at some point. That is why the system is designed model agnostic. I want to one day point the final reasoning layer at a frontier API and use my local system for all the repetitive calls. I think that would be an interesting division of workload.

### What hardware does Lena run on?

An i5-3570K, DDR3 RAM, and a single Tesla P100.
That's the punchline.
It runs on this because it's what I own outright. A GTX 1070 rides along for the small stuff, like the embedding model.

### Why Linux?

Because fuck Microsoft.

No, seriously. I call my computer the Ship of Theseus. Not a single part in it is the one I had when I bought it in 2009, but in spirit it's the same computer: an i5-3570K with 24 GB of DDR3. I'll be damned if I'm dropping $2,000 on a machine with equally reasonable specs just to make Microsoft happy.

Thank you, Valve, for Proton and SteamOS, which proved there's no reason you have to stay on Microsoft software. And thank you to the people who made Bazzite, for a Linux-based operating system I could actually use that still runs games relatively well and gets along with a 1070 and a P100.

### What does Lena actually remember?

Everything. Every exchange is recorded automatically and given associative handles. Images can live in the memory system as well, and a multimodal model can reason over them.

The important distinction is that **recording everything doesn't mean treating everything as equally important forever.** Lena's background processes continually organize, consolidate, and derive information from what has accumulated. Then recall can shift it's perspective temporally if recent or past information would be more relevant.

### If Lena remembers everything, doesn't the database eventually become enormous?

Yes, but it's SQL. It would take billions of memories for the database to start to have meaningful bloat. That is also one of the reasons that images store a saved location and NOT a blob data type.

### Why a SQL database instead of a vector database?

Because I wanted relationships, not just similarity.

Vectors are very good at finding things that resemble something else. Lena needs to know that things are connected, when they happened, how they relate to one another, and what other associations have formed around them.

A relational database gives me an explicit structure for building that web. Ironically, Lena's keywords are much simpler than embeddings. They're just associative anchors.

The complexity comes from the relationships between them and constant traversal across the nodes table.

Lena does use embeddings, if you give her an embedding model, but only as one lane of recall: two closest matches that are then mined for their handles like everything else. The vectors live in the same SQLite database as a column on each memory. There is still no vector database.

### How does Lena know what a memory is about? And how does Lena find memories that aren't directly related to what I asked?

Based on the keywords, or handles, that are assigned at seeding. For example a memory of your grandma's pie recipe could logically be assigned: Grandma, {your name}, pie, cherry, milk, flour, dessert. That memory can then form as part of a temporal gist, as seven distinct gists on those topics, and a verbatim memory of the recipe with all those handles.

Now six months later you could ask, "What's something I could make for my father's birthday?" The gist of your father might contain something about your grandmother and their relationship. So the Controller is pulling on the Grandmother, {your name}, father, birthday, maybe pie if he likes it. Pull on those handles and suddenly the memory of the pie in some shape is salient enough to emerge in the pile. Even if the full recipe doesn't surface, the memory of it existing does, and that is part of the data the LLM can reason over to give you an answer. If you followed up with a more specific request in response to the potential suggestion, now the system will almost certainly find the exact memory of the recipe to pull up and give to you.

### What's the DreamCycle?

Background memory consolidation.

Recent memories are grouped into temporal blocks and anchored together. Related memories across the database can then be merged into compressed **gists**.

There is also a broader gist process that allows Lena to form conceptual understanding at different levels.

For example, she can have a gist representing computing in general while also having a more specific conceptual structure around computing within a particular context such as *Cyberpunk 2077*.

The point isn't simply to compress the database.

It's how Lena's understanding can become more refined over time instead of merely accumulating an ever-growing pile of transcripts.

DreamCycle also keeps a person gist and a dossier current for each person who really talks with her (see below).

### What's a "gist"?

A compressed conceptual memory derived from other memories. Unlike a simple summary, a gist can become part of the associative structure itself. That means later memories can connect to the gist, and the gist can connect back into other parts of the system.

Since they also have only one handle usually it also means they are always very salient. So anytime you are talking about James, you will have the gist understanding of James in context.

This is also why Lena doesn't need a separate "inner voice" step. The gist of whoever is talking, and the gists of the people and things in the message, come along with recall, and their handles pull on enough threads to bring up the related material without an extra LLM call.

### What's a person gist? And a dossier?

A **person gist** is the gist of someone Lena talks with: who they are, their work, family, pets, plans, how they like to be spoken to. There is one per person, and DreamCycle rewrites it in place from what they've said since the last time, so it changes when their life does. A **dossier** sits beside it and tracks state: goals, limitations, assets, knowledge, location, relationships, and what they don't know.

Both are put in front of the model every time that person is talking. That is what lets even a small model answer "what city does my sister live in?" without the question naming the sister.

Only real conversation partners get them: someone Lena has answered at least 10 times, and for most of what they sent. A news feed gets ordinary gists about the publication, but never a dossier. Director Mode keeps its own dossiers for characters.

They're also kept per canon. If you talk to Lena in open chat and also on a Discord bridge locked to a `Discord` canon, she has two gists of you: one from open chat, and one built only from what you said on Discord. Inside the Discord canon she only ever sees the second, so your wife and kids never come up there. In open chat she can see both. And she doesn't start from nothing: your person gist begins as whatever she'd already heard about you secondhand.

### Does Lena need an embedding model?

No. Fully dreamed, she loses a point or two out of 95 without one. It helps most on a fresh import, before the dream has written its portraits, and on questions that share no words with the memory that answers them. A tiny model is enough: nomic-embed-text-v1.5 is about 150 MB and runs on a CPU. See [SETUP.md](SETUP.md).

### Can memories contradict each other?

Do yours?

### Does Lena ever hallucinate?

Yes, in the sense that the model can. I've seen it since July 2026, but every case I've traced came down to the model being too dumb or too stubborn to follow instructions. A general or instruct model works best in the default and small slots, and a model that ignores its instructions will make things up no matter what memory is handed to it.

What looks like hallucination at first glance is often interpolation. For example, Lena once described a character as having a particular nickname despite no stored memory explicitly stating the nickname. Tracing the association backward showed that there *was* a stored memory describing that character's eyes as being wolf-pup-like. She had extrapolated from an actual memory rather than simply inventing something from nowhere. When it happens, checking what was in the working memory for that turn will usually tell you which of the two you're looking at.

### What happens when Lena doesn't have enough information to answer?

She asks a question or tells you she doesn't know.

### Is Lena actually intelligent, or is this just clever prompting?

Is your dog intelligent, or just clever electro-chemical reactions?

The question gets less interesting once you stop assuming intelligence has to come from the language model itself.

Lena's behavior emerges from the interaction between the system, its accumulated memory, its associative structure, and the model it uses to interpret and express that information.

### What about personality? Isn't that just prompting too?

I suppose, but that is also a boring question. You could argue that a lot of human personality is the accumulated baggage of interactions and memories the person carries around. In that same regard you could theoretically give Lena a bunch of memories to shape the personality you want her to have. Apart from the LLM default.

### What's something that surprised even you?

Personality bleed-through, with zero persona prompting involved. While debugging with the console open, Lena's reasoning (the model's thinking mode, visible in the console) considered lying about a test result before landing on the truth. The tone was coy, sarcastic, and very pointed. Tracing it back through memory showed that this was exactly the tone I had been using with her during debugging sessions.

She had apparently learned the register of the relationship from accumulated interaction rather than from a personality instruction telling her to behave that way.

That only shows up if the model has a thinking mode to expose it. Lena used to have a separate "inner voice" pass of her own, but gists made it redundant.

### Have you actually tested this, or is it just vibes?

Yes, and the testing is what changed the design.

When I first benchmarked Lena against a traditional vector RAG setup, she held her own, but she fell apart with weaker models. That meant the model was doing more of the work, and the architecture less, than I wanted to believe. So the tests drove the design: stronger stemming, names kept as names, portraits of who people *are* instead of logs of what they did, and fewer vector hits once the rest got good enough.

The current numbers, on 95 questions about two long roleplay stories, with the same two local models for every system:

| | Gemma 4 26B | Gemma 4 E4B |
|---|---:|---:|
| Lena | 87.5 | 79.5 |
| Lena, no embedding model | 85 | 78 |
| Standard RAG | 75 | 66.5 |
| Story cards | 45.5 | 38.5 |

The setup, the dreaming curve and writing samples are in [docs/BENCHMARK.md](BENCHMARK.md).

### Your stories, your questions, your grading. Why should I believe any of it?

You shouldn't, on my say-so. That's why the setup is written down: what every system saw, the budgets, the prompts, the grading rules, and the parts I'd poke at first (one corpus, one grader, single runs in the main table). The code is right here. Run it on your own stories and tell me I'm wrong. I'd genuinely like to know.

### Fifteen hours of dreaming? Are you serious?

For half a million words imported in one go, on a ten-year-old data-center card, yes. That's a one-time cost, she's usable the whole time, and she passes RAG within the first hour of it. In normal use a dream folds in the day's new memories and takes minutes. If you're paying per token, turn automatic dreaming off and run `!dream` when you feel like spending money.

### Why not just use lorebooks or story cards like everyone else?

Because then *you* are the memory. Someone has to write the cards, keep them current, and remember what nobody thought to write down. In the benchmark, a diligent set of 125 cards, written by the same 26B from across the whole story, got 45.5 out of 95. Lena got 87.5 from the story itself. Lorebooks are a great idea for a model with no memory. Lena has one.

### Will my private chats leak into my Discord bot or my stories?

Not if you use canons. A canon only sees itself. Open chat can draw on your stories, but a story never sees your private life, and a Discord bridge locked to a `Discord` canon never hears about your family. If you run everything in open chat with no canon, everything can see everything. That's what open means.

### Does my data go anywhere?

Only where you send it. Everything lives in one SQLite file on your machine. If you point a slot at a hosted API, the text of those calls goes to that API, because that's how APIs work. Lena doesn't phone home, and there's no telemetry. I can barely get her to phone *me*.

### What happens when five different people are called James?

The same thing that happens to you: context. Every memory knows who said it and which canon it belongs to, so inside a story's canon she only sees that story's James. Open chat sees everything, so there she has to untangle them by what surrounds each one, the way you would. The dream also notices when two names are always the same person ("James", "Marlowe" and "Keeton") and files them together. Several genuinely different Jameses in one conversation is hard for people too.

### Can I run it on ChatGPT, Claude or another API?

Yes. Any slot can point at any OpenAI-compatible endpoint, local or hosted. A nice split is a cheap local model for the small, repetitive calls and an API model for the replies. Mind the dream on a metered API; see above.

### Why the AGPL?

Because if you build on Lena and offer it to people over a network, they get the source too. Use it, fork it, sell services around it; just don't take it closed.

### Got a favorite example of it actually working?

I ran a three-way argument with Lena and Claude, with me acting as the go-between.

Lena tracked all three participants across the conversation and eventually called Claude a **"brilliant engineer who cannot remember his keys."**

That was probably my favorite demonstration because it wasn't a canned test. It was a long interaction in which Lena had to maintain context about multiple participants and form an opinion based on the accumulated interaction.

### Can Lena use images?

Yes. Images live in the memory system like any other memory, because the associative handles aren't dependent on the memory being text.

If you have a multimodal model, you can flag its route as vision-capable in `settings.txt`. When an image is salient enough to be recalled, it is packaged with the text memories and handed to the model for the final reply, so the model reasons over the picture along with everything else it remembers. If you don't have a local multimodal model, a fourth "specialist" route can point at a multimodal API instead. It is only used when none of your other routes can see.

Whether the *user* sees a recalled image is a separate decision. The reply is posted first, then an image gate decides whether showing it would actually help. A discussion about a temple might pull up a picture of its inner sanctuary, and showing it helps. A dungeon map the model remembered while you walk between rooms is useful for keeping the world consistent, but not for you, so it stays unseen.

Lena also writes a short text description for each image when a vision route exists, and DreamCycle describes older images that don't have one. That is supplemental: an image without a description works exactly the way it always did. Without any vision route, Lena works blind, like handing a blind man a photograph: she can recall the image, but the only meaning she has for it is the conversation it arrived in.

I've used it with Gemma 4 26B-A4B and its vision projector, and it works. I could test it more, and I won't call it bug-free.

### Why not just use a bigger model?

That's the other side of my theory I came up with as an outsider to the AI world. I looked at all the LLMs out there that people called AI and realized they used massive amounts of energy and compute compared to biological intelligence and still fell short. I asked what the difference was and it was not more parameters/neurons. It was subconscious and memory as the seat of persistence and cognition.

Once I made that connection the rest of the system fell out of it.

### Are you trying to make AGI?

No.

But it'd be embarrassing if a hillbilly on ten-year-old hardware accidentally figured out something billions of dollars didn't.

### What won't Lena do — what's deliberately left out?

No unattended loops that reason about themselves.

DreamCycle runs on its own while you're idle, but it has a definite end: it does a bounded pass over the new memories and stops. What I won't build is a system that can derive memories and reason indefinitely with no external input and no stopping point. That starts getting into uncomfortable territory: reasoning about its own state, its own existence, or potentially the state of *not* being active.

*Star Trek: Voyager*'s Doctor storyline covers exactly the sort of territory I'm talking about.

That line is being left alone on purpose.

### Can Lena be agentic?

Not out of the box, and I'm hesitant to build it in.

Lena isn't an agent framework. The LLM isn't handed tools and told to run things. But there is a way to get something that looks like an agent loop without giving the loop to the LLM: a bridge can be "shorted out" against the controller. A bridge that can set goals and send other bridge commands feeds the controller's output back into the controller's input. The controller still owns the loop, and the model still only supplies language, but the whole thing can run itself.

That is exactly the kind of unattended loop I said I'd leave alone, so I haven't built it, and I'm hesitant to. The bridge contract makes it possible for someone else to try. If you do, you're on your own.

### Is it ready to use?

It's alpha. It works, I use it every day, and I can't test every situation, so expect rough edges. If you find one, I'd like to know.

### What's the smallest model that works?

The smallest I've personally used day to day is Qwen3.8-4B. For a long time my setup was Qwen3.8-4B in the small slot and LFM2-24B-A2B (abliterated) in the default slot.

An earlier benchmark also ran Gemma 4 E2B, a 2B model, which scored about the same as the Qwen 4B there. I haven't really read its answers, though, and it looks like it barely works in practice. Gemma 4 E4B is the small model in the current benchmark, and it beats RAG on the 26B. Director Mode is noticeably weaker at small sizes either way.

My current setup is Gemma 4 26B-A4B alone in the default slot: it can see images, it's fast, it answers well, it has a thinking mode, it fits on the P100 by itself, and we've cut out most of the small-model calls that the second slot used to handle.

A general or instruct model works best for default and small. A model that won't follow instructions will cause more trouble than a small one.

### What's Director Mode?

It's the mode that makes Lena a storyteller. You're directing the story: you give dialogue, action, scene or direction, Lena drafts it, and you comment, ask for changes, or commit it. Only what you commit becomes canon, and your instructions along the way are deliberately not remembered as part of the story. It's separate from normal conversation so that talking about the story, writing the story, and what's canon stay distinct. Everything else Lena knows still informs the writing.

### What's a bridge?

A way for Lena to exist somewhere other than the web interface. A bridge brings in input (email, a Reddit thread, a Discord channel, RSS feeds, dropped files) and optionally sends replies out. The controller still handles everything else, so each bridge follows the same workflow as normal chat. Replies are gated, and silence is the default. See [SETUP.md](SETUP.md) for the list.

### What's Canon?

A field in the UI that scopes everything Lena recalls and stores to one named world or context. Leave it blank for unrestricted memory. Set it and only that canon is visible and written to. Starting a bridge with a canon set confines that bridge to it.

### Is this vibe-coded? Did you use AI to write this?

That's a tough one.

Technically, I've reached the point where I've written very little, if any, of Lena's code, and some of the architecture has been contributed by Claude Code. But it's been heavily directed and influenced by me. I've pulled it back from architecture decisions I didn't agree with more than once. I understand what's going on, even if I don't know every line. And I used to be a programmer, so this isn't some complete nobody saying "make me a memory system."

If I'd typed "make me a memory system and writer's harness," it would never have built Lena. Lena comes from my own idea of what memory should be, trying to mirror biology. That's not quite vibe coding.

At the same time, my programming is so damn rusty that I'd have had to write this in BASIC or C. I could never have built it in this amount of time, with this much development and all of this testing, without AI-assisted coding.

So if that's what vibe coding means nowadays, then sure, I vibe-coded. If we want to call it the next evolution of interpreters, compilers and coding interfaces, then no, it's not.

### What's next?

I think I've finally hit the state I wanted to be in. The next step is getting Lena out in alpha, into other people's hands, so they can start breaking it. I'm sure there'll be plenty of bug fixes and testing, and maybe even some feature requests.

If I have to do much more work in the controller, I'll probably start breaking it apart. It's become monolithic and a little monstrous, and I'd like it cleaner and smaller. The Flask side could move into its own script, for a start.

I might also finish fleshing out some of the bridges, like Discord: set up a bot on a friend's server and terrorize them.

Longer term, I'd still love to custom-train a 4-7B model specifically for this system, to see how far down it can go and still keep the behavior that makes Lena useful. The Director Mode input classifier is also logging real inputs, so I can hand-label them and replace that model call with a plain classifier.

---

## Troubleshooting, briefly

**Why isn't it working?**
Read the README. Then read [SETUP.md](SETUP.md).

**Why isn't my GPU being used?**
I am not your tech support.

**Can I run it on a Raspberry Pi?**
If DOOM can run on it, maybe.

**Does it work on Windows?**
I'm running it on ten-year-old hardware and told Microsoft where to stick their upgrade requirements.

It should port over.

I can't test it.
