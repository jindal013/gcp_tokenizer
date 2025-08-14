from datatrove.pipeline.readers import ParquetReader

# limit determines how many documents will be streamed (remove for all)
data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu", glob_pattern="data/*/*.parquet", limit=1000)
# or to fetch a specific dump CC-MAIN-2024-10,  eplace "CC-MAIN-2024-10" with "sample/100BT" to use the 100BT sample
data_reader = ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/CC-MAIN-2024-10", limit=1000) 
for document in data_reader():
    # do something with document
    print(document)

###############################    
# OR for a processing pipeline:
###############################

from datatrove.executor import LocalPipelineExecutor
from datatrove.pipeline.readers import ParquetReader
from datatrove.pipeline.filters import LambdaFilter
from datatrove.pipeline.writers import JsonlWriter

pipeline_exec = LocalPipelineExecutor(
    pipeline=[
        # replace "CC-MAIN-2024-10" with "sample/100BT" to use the 100BT sample
        ParquetReader("hf://datasets/HuggingFaceFW/fineweb-edu/CC-MAIN-2024-10", limit=1000),
        LambdaFilter(lambda doc: "hugging" in doc.text),
        JsonlWriter("some-output-path")
    ],
    tasks=10
)
pipeline_exec.run()
