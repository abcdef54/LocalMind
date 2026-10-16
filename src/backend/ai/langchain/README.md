# AI tools

Tools let a candidate request document context or web information. The server fixes the conversation scope before the model sees the retrieval tool.

```text
generate() → tools.py → create_retrieval_tool(conversation_id)
        ↓
Model requests tool by name with validated arguments
        ↓
graph_nodes.py → _generate_sample_with_tools() → tool.ainvoke()
        ↓
Document tool → indexing/retrieval.py → ConversationRetriever.search()
        ↓
Read ready document generations from PostgreSQL
        ↓
Chroma search filtered by conversation + ready generations
        ↓
Recheck DB state; remove deleted/replaced results
        ↓
Return context + citation artifact as tool result
        ↓
Model continues generation; winning candidate carries citations

Web tool requests → tavily_search() / fetch_web_page()
```
