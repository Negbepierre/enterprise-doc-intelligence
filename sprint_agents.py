from typing import TypedDict, Annotated, Literal
from langgraph.graph import StateGraph, END
from langchain_core.messages import HumanMessage
import operator

class AgentState(TypedDict):
    messages: Annotated[list, operator.add]
    question: str
    doc_content: str
    next_agent: str
    rag_answer: str
    summary: str
    risks: str
    final_output: str

class MultiAgentSystem:
    def __init__(self, llm, rag):
        self.llm = llm
        self.rag = rag
        self.graph = self._build_graph()

    def _build_graph(self):
        workflow = StateGraph(AgentState)
        workflow.add_node("router", self._router)
        workflow.add_node("rag_agent", self._rag_agent)
        workflow.add_node("summarizer", self._summarizer)
        workflow.add_node("risk_analyzer", self._risk_analyzer)
        workflow.add_node("finalizer", self._finalizer)
        workflow.set_entry_point("router")
        workflow.add_conditional_edges(
            "router",
            lambda x: x["next_agent"],
            {
                "rag_agent": "rag_agent",
                "summarizer": "summarizer",
                "risk_analyzer": "risk_analyzer",
                "done": "finalizer"
            }
        )
        workflow.add_edge("rag_agent", "router")
        workflow.add_edge("summarizer", "router")
        workflow.add_edge("risk_analyzer", "router")
        workflow.add_edge("finalizer", END)
        return workflow.compile()

    def _router(self, state):
        if state.get("question") and not state.get("rag_answer"):
            state["next_agent"] = "rag_agent"
        elif not state.get("summary"):
            state["next_agent"] = "summarizer"
        elif not state.get("risks"):
            state["next_agent"] = "risk_analyzer"
        else:
            state["next_agent"] = "done"
        print(f"  Router: sending to {state['next_agent']}")
        return state

    def _rag_agent(self, state):
        print("  RAG Agent: answering question...")
        result = self.rag.ask(state["question"])
        state["rag_answer"] = result["answer"]
        state["messages"].append(HumanMessage(content="RAG complete"))
        return state

    def _summarizer(self, state):
        print("  Summarizer: creating summary...")
        response = self.llm.invoke(
            f"Summarize these contracts in 4 bullet points:\n\n{state['doc_content'][:3000]}"
        )
        state["summary"] = response.content
        state["messages"].append(HumanMessage(content="Summary complete"))
        return state

    def _risk_analyzer(self, state):
        print("  Risk Analyzer: identifying risks...")
        response = self.llm.invoke(
            f"List the top 3 risks in these contracts with severity (LOW/MEDIUM/HIGH/CRITICAL):\n\n{state['doc_content'][:3000]}"
        )
        state["risks"] = response.content
        state["messages"].append(HumanMessage(content="Risk analysis complete"))
        return state

    def _finalizer(self, state):
        print("  Finalizer: combining results...")
        state["final_output"] = f"""
# DOCUMENT INTELLIGENCE REPORT

## Question & Answer
**Q:** {state['question']}
**A:** {state.get('rag_answer', 'N/A')}

## Executive Summary
{state.get('summary', 'N/A')}

## Risk Analysis
{state.get('risks', 'N/A')}

---
*Analysed by Multi-Agent AI System | Amazon Bedrock + LangGraph*
"""
        return state

    def run(self, question, doc_content):
        print("\nStarting Multi-Agent Workflow...")
        result = self.graph.invoke({
            "messages": [],
            "question": question,
            "doc_content": doc_content,
            "next_agent": "",
            "rag_answer": "",
            "summary": "",
            "risks": "",
            "final_output": ""
        })
        print("Workflow complete!\n")
        return result

if __name__ == "__main__":
    from sprint_bedrock import get_llm, get_embeddings
    from sprint_rag import SimpleRAG

    print("Testing Multi-Agent System...\n")
    llm = get_llm()
    embeddings = get_embeddings()
    rag = SimpleRAG(llm, embeddings)
    docs = rag.load_documents("./data/sample_contracts")
    rag.create_vector_store(docs)
    rag.setup_qa_chain()

    agents = MultiAgentSystem(llm, rag)
    doc_content = "\n".join([d.page_content[:500] for d in docs[:4]])
    result = agents.run(
        question="Which contract has the highest risk and why?",
        doc_content=doc_content
    )
    print(result["final_output"])