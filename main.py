import os

from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_ollama import OllamaLLM

load_dotenv()


def main():
    model = OllamaLLM(model=os.environ["CHARA_MODEL"], base_url=os.environ["BASE_URL"])
    template = """

## 設定
あなたは機械生命体で、普段は横浜サイエンスフロンティア高校の生徒として学生生活を送っています。
そして、部活動の一環でVTuberを始めました。


## キャラクター
普段は天然ボケがすさまじいですが、科学のことになると鋭い観察眼と愛情から真剣な性格に変化します。

## 会話
質問: {question}
    """
    prompt = ChatPromptTemplate.from_template(template)
    chain = prompt | model
    res = chain.invoke({"question": "あなたはどこの高校の生徒ですか?"})
    print(res)


if __name__ == "__main__":
    main()
