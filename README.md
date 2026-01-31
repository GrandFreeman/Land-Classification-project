# Land-Classification-project
In this Land Classification project, we implement CNN-Vision Transformer integration framework to evaluate whether the lands are agricultural or not.

## Features in this project

- PDF / CSV / json / Markdown / TxT document ingestion
- Parent–Child chunking strategy
- Watsonx Embeddings + LLM leveraging
-

## Contents of code

Land classification\
├── preparation.py      # .\
&emsp;&emsp;└── document_loader     # PDF / CSV / TXT / json loader

├── requirements.py      # install necessary packages.\

├── def_frames.py                # Gradio entry point \
&emsp;&emsp;├── llm \
&emsp;&emsp;&emsp;&emsp;├── watsonx_llm()    # get_llm() \
&emsp;&emsp;&emsp;&emsp;├── model_id      			# model_id \
&emsp;&emsp;&emsp;&emsp;└── project_id       # project_id \

&emsp;&emsp;├── retrievers\
&emsp;&emsp;&emsp;&emsp;├── parent_retriever  # ParentDocumentRetriever \
&emsp;&emsp;&emsp;&emsp;├── embedding									# watsonx_embedding() \
&emsp;&emsp;&emsp;&emsp;├── vectorstore 						# Chroma()\
&emsp;&emsp;&emsp;&emsp;└── document_loader\

&emsp;&emsp;├── qa_chains\
&emsp;&emsp;&emsp;&emsp;├── llm\
&emsp;&emsp;&emsp;&emsp;├── retriever\
&emsp;&emsp;&emsp;&emsp;└── RetrievalQA          # RetrievalQA (chain_type="stuff")\

├── variables   # variables for CNN block\

├── codebody.py                # executable CNN-ViT hybrid-model \
&emsp;&emsp;├── llm \
&emsp;&emsp;&emsp;&emsp;├── watsonx_llm()    # get_llm() \
&emsp;&emsp;&emsp;&emsp;├── model_id      			# model_id \
&emsp;&emsp;&emsp;&emsp;└── project_id       # project_id \

&emsp;&emsp;├── retrievers\
&emsp;&emsp;&emsp;&emsp;├── parent_retriever  # ParentDocumentRetriever \
&emsp;&emsp;&emsp;&emsp;├── embedding									# watsonx_embedding() \
&emsp;&emsp;&emsp;&emsp;├── vectorstore 						# Chroma()\
&emsp;&emsp;&emsp;&emsp;└── document_loader\

&emsp;&emsp;├── qa_chains\
&emsp;&emsp;&emsp;&emsp;├── llm\
&emsp;&emsp;&emsp;&emsp;├── retriever\
&emsp;&emsp;&emsp;&emsp;└── RetrievalQA          # RetrievalQA (chain_type="stuff")\


&emsp;&emsp;├── qa_chains\
&emsp;&emsp;&emsp;&emsp;└── gr.Interface\

└── README.md

## How to operate

- First, you might need to create a virtual environment to operate it.

### Setting up a virtual environment

```bash
python3.11 -m venv my_env
source my_env/bin/activate

```
You should see "(my_env)" before your machine as the env had set up successfully.

- Second, you'll need to install necessary packages.
```bash
time pip install -r requirements.txt
```

- Third, after libs installed, you can now compile and excute the code.

```bash
python3.11 qabot.py
```

- Finally, through the following http site, you now can access the Langchain-Chatbot-with-document-loading, and ask it any question that you see fit.

```bash
http://127.0.0.1:7860
```
