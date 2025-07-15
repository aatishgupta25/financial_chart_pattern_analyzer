
# Financial Chart Pattern Analyzer 
* Built a ViT-GRU Vision-Language Model in PyTorch to detect candlestick patterns from financial charts
* Generated 5K+ labeled charts from historical OHLC data using rule-based pattern labeling for supervised training
* Integrated the model into a LangChain RAG interface using the gemma3-4b model via Ollama and FAISS retrieval to explain predictions with financial literature

## Pipeline to Run Project
- Clone Repo, cd into repo, make virtual enviroment, install dependencies
- Run `generate_data.py` to get data from yfinance API and it uses __detech_pattern()__ from `patterns.py` to label data
- Run `dataset.py` to initialize custom PyTorch Dataset class
- Run `model.py` to define ViT-GRU-based VLM
- Run `train.py` to train the model created, adjust batch size and no. of epochs according to your device constraints
- Run `rag_setup.py` to loada and chunk text data, then build the FAISS index
### For the next step, ensure to download Ollama, and specifically the model 'gemma3:4b-it-qat' and keep it running for the model inference
- Run  `explain.py` to get the inference and explaination pipeline to the VLM. Adjust image path accordingly in the script
