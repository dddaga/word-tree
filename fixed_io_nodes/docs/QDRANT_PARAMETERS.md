## QDRANT's parameters

#### [Optimizer config](https://qdrant.tech/documentation/concepts/optimizer/)

* ***`deleted_threshold`*** - The minimal fraction of deleted vectors in a segment, required to perform segment optimization. rebuilts the entire segment, including re-indexing. We would want to set this as low as possible.


* `vacuum_min_vector_number` - The minimal number of vectors in a segment required to run segment optimization

* `default_segment_number`  - Target amount of segments optimizer will try to keep.  If set to `0`, it will be automatically selected by the number of available CPUs. We would want to set this higher (chose b/w time taken for searching and indexing)

* `max_segment_size_kb` - Segements will not exceed the size specified by this. Default `None`



* ***`memmap_threshold`*** - The maximum size (in kilobytes) of vectors stored in memory per segment. Vectors beyond this limit will be stored on disk. the vectors to store in memory is handled by OS. The eviction policy is LRU.Default `20000`



* `indexing_threshold_kb` - Maximum size (in kilobytes) of vectors allowed for plain indexing. Exceeding this will enable vector indexing. Default `20000`, set to 0 for disabling vector indexing.

* `indexing_threshold` - similar to  `indexing_threshold_kb` but unit is number of vectors instead of total size of segment.



#### [HNSW Config](https://qdrant.tech/documentation/concepts/indexing/#vector-index)


* `m` - Number of edges per node in the index graph. Larger the value - more accurate the search, more space required.

* `ef_construct` - Number of neighbours to consider during the index building. Larger the value - more accurate the search, more time required to build index
