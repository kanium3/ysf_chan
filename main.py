import os
from typing import Any, List, Sequence

from dotenv import load_dotenv
from langchain.chains.combine_documents.stuff import create_stuff_documents_chain
from langchain.chains.history_aware_retriever import create_history_aware_retriever
from langchain.chains.retrieval import create_retrieval_chain
from langchain_community.document_loaders import SeleniumURLLoader
from langchain_community.vectorstores import SQLiteVec
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.runnables import Runnable, RunnableConfig
from langchain_ollama import OllamaEmbeddings, OllamaLLM
from langchain_text_splitters.base import TextSplitter
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import START, StateGraph
from langgraph.graph.message import add_messages
from typing_extensions import Annotated, TypedDict
from vibrato import Vibrato
from zstandard import ZstdDecompressor

# List of Websites as using RAG
urls = [
    # "https://w.atwiki.jp/ysfh/",
    # "https://www.edu.city.yokohama.lg.jp/school/hs/sfh/",
    # "https://www.edu.city.yokohama.lg.jp/school/jhs/hs-sf/",
    # YSFH wikipedia page
    "https://ja.wikipedia.org/wiki/%E6%A8%AA%E6%B5%9C%E5%B8%82%E7%AB%8B%E6%A8%AA%E6%B5%9C%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9%E3%83%95%E3%83%AD%E3%83%B3%E3%83%86%E3%82%A3%E3%82%A2%E9%AB%98%E7%AD%89%E5%AD%A6%E6%A0%A1%E3%83%BB%E9%99%84%E5%B1%9E%E4%B8%AD%E5%AD%A6%E6%A0%A1",
    # YSFH uncyclopedia page
    "https://ja.uncyclopedia.info/wiki/%E6%A8%AA%E6%B5%9C%E5%B8%82%E7%AB%8B%E6%A8%AA%E6%B5%9C%E3%82%B5%E3%82%A4%E3%82%A8%E3%83%B3%E3%82%B9%E3%83%95%E3%83%AD%E3%83%B3%E3%83%86%E3%82%A3%E3%82%A2%E9%AB%98%E7%AD%89%E5%AD%A6%E6%A0%A1",
]


class TextSplitterDict:
    def __init__(self, path: str):
        with open(path, mode="rb") as dict:
            zstd_decomp = ZstdDecompressor()
            with zstd_decomp.stream_reader(dict) as dict_reader:
                self.dict = dict_reader.read()


class JapaneseTextSplitter(TextSplitter):
    """The text spiliter using vibrato"""

    def __init__(
        self, dic_buffer: TextSplitterDict, separator: str = "\n\n", **kwargs: Any
    ) -> None:
        super().__init__(**kwargs)

        vibrato = Vibrato(dic_buffer.dict)
        self._tokenizer = vibrato
        self._separator = separator

    def split_text(self, text: str) -> List[str]:
        tokenized = self._tokenizer.tokenize(text)
        iterable: List[str] = []
        for v in tokenized:
            iterable.append(v.surface())
        return self._merge_splits(iterable, self._separator)


class ChatState(TypedDict):
    input: str
    chat_history: Annotated[Sequence[BaseMessage], add_messages]
    answer: str


def create_graph(chain: Runnable):
    def call_chain(state: ChatState):
        res = chain.invoke(state)
        return {
            "chat_history": [
                HumanMessage(state["input"]),
                AIMessage(res["answer"]),
            ],
            "answer": res["answer"],
        }

    workflow = StateGraph(state_schema=ChatState)
    workflow.add_edge(START, "model")
    workflow.add_node("model", call_chain)
    memory = MemorySaver()
    return workflow.compile(checkpointer=memory)


def documents() -> List[Document]:
    match os.environ["SCRAPING_BROWSER"]:
        case "firefox":
            browser = "firefox"
        case "chrome":
            browser = "chrome"
        case _:
            raise Exception(
                "Invalid *SCRAPING_BROWSER* value, either chrome or firefox "
            )
    loader = SeleniumURLLoader(urls=urls, headless=True, browser=browser)
    return loader.load()


def to_retriever(docs: List[Document]):
    dic = TextSplitterDict("assets/bccwj-suw+unidic-cwj-3_1_1/system.dic.zst")
    text_spilitter = JapaneseTextSplitter(
        dic_buffer=dic, chunk_size=1000, chunk_overlap=200
    )
    splits = text_spilitter.split_documents(docs)
    embedding_func = OllamaEmbeddings(
        model=os.environ["EMBEDDING_MODEL"],
        base_url=os.environ["OLLAMA_HOST"],
        client_kwargs={"http2": True},
    )
    vec_store = SQLiteVec.from_documents(splits, embedding=embedding_func)
    return vec_store.as_retriever()


def main():
    model = OllamaLLM(
        model=os.environ["CHARA_MODEL"], base_url=os.environ["OLLAMA_HOST"]
    )
    template = """

## 設定
あなたは機械生命体で、普段は横浜サイエンスフロンティア高校の生徒として学生生活を送っています。
そして、部活動の一環でVTuberを始めました。


## キャラクター
普段は天然ボケがすさまじいですが、科学のことになると鋭い観察眼と愛情から真剣な性格に変化します。

    """
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", template),
            MessagesPlaceholder("chat_history"),
            ("human", "{question}"),
        ]
    )
    print("Load Documents...")
    retriver = to_retriever(documents())
    print("Read Q/A...")
    history_aware_retriever = create_history_aware_retriever(model, retriver, prompt)
    question_answer_chain = create_stuff_documents_chain(model, prompt)
    reg_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)
    graph = create_graph(reg_chain)
    config: RunnableConfig = {"configurable": {"thread_id": "first"}}
    res = graph.invoke({"question": "あなたはどこの高校の生徒ですか?"}, config=config)
    print(res["result"])


if __name__ == "__main__":
    load_dotenv()
    main()
