# Sample AI Policy

Acme Research Lab uses AI assistants to help employees summarize internal documents,
draft technical notes, and answer questions about approved knowledge base content.

## Data handling

Employees must not upload secrets, passwords, private keys, medical records, or customer
personal data into AI systems unless the system has been approved for that data category.
Documents used for retrieval should be reviewed before ingestion.

## Human review

AI-generated answers must be treated as drafts. A human owner is responsible for checking
facts, citations, and business impact before decisions are made.

## Retrieval-augmented generation

RAG systems should return source snippets with answers. Source snippets help users understand
why the assistant produced a response and make it easier to verify whether the retrieved
context is relevant.

## Monitoring

Teams should monitor failed searches, low-confidence answers, and user feedback. Monitoring
helps improve chunking, retrieval quality, prompts, and the document ingestion process.

